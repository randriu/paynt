#include "GameAbstractionSolver.h"

#include "src/synthesis/pomdp_family/SmgAbstraction.h"
#include "src/synthesis/smg/smgModelChecker.h"
#include "src/synthesis/translation/componentTranslations.h"
#include "src/synthesis/translation/ItemKeyTranslator.h"
#include "src/synthesis/translation/ItemTranslator.h"

#include <storm/environment/solver/GameSolverEnvironment.h>
#include <storm/environment/solver/NativeSolverEnvironment.h>
#include <storm/models/sparse/Smg.h>
#include <storm/solver/GameSolver.h>
#include <storm/storage/PlayerIndex.h>
#include <storm/utility/builder.h>

#include <queue>

namespace synthesis {

    template<typename ValueType>
    GameAbstractionSolver<ValueType>::GameAbstractionSolver(
        storm::models::sparse::Model<ValueType> const& quotient,
        uint64_t quotient_num_actions,
        std::vector<uint64_t> const& choice_to_action,
        std::shared_ptr<storm::logic::Formula const> formula,
        bool player1_maximizing,
        std::string const& target_label,
        double precision
    ) : quotient(quotient), quotient_num_actions(quotient_num_actions), choice_to_action(choice_to_action),
        player1_maximizing(player1_maximizing) {

        auto quotient_num_states = quotient.getNumberOfStates();
        auto quotient_num_choices = quotient.getNumberOfChoices();

        this->choice_to_destinations.resize(quotient_num_choices);
        for(uint64_t choice=0; choice<quotient_num_choices; choice++) {
            for(auto const &entry: quotient.getTransitionMatrix().getRow(choice)) {
                this->choice_to_destinations[choice].push_back(entry.getColumn());
            }
        }
        this->setupSolverEnvironment(precision);

        std::vector<std::variant<std::string, storm::storage::PlayerIndex>> coalition_vector;
        coalition_vector.emplace_back((storm::storage::PlayerIndex)0);
        storm::logic::PlayerCoalition coalition(coalition_vector);
        this->game_formula = std::make_shared<storm::logic::GameFormula const>(coalition,formula);

        // identify target states
        this->state_is_target = storm::storage::BitVector(quotient_num_states,false);
        for(auto state: quotient.getStateLabeling().getStates(target_label)) {
            this->state_is_target.set(state,true);
        }

        this->solution_state_values = std::vector<double>(quotient_num_states,0);
        this->solution_state_to_player1_action = std::vector<uint64_t>(quotient_num_states,quotient_num_actions);
        this->solution_state_to_quotient_choice = std::vector<uint64_t>(quotient_num_states,quotient_num_choices);
    }

    
    template<typename ValueType>
    void GameAbstractionSolver<ValueType>::setupSolverEnvironment(double precision) {
        this->env.solver().game().setPrecision(storm::utility::convertNumber<storm::RationalNumber>(precision));

        // value iteration
        // this->env.solver().game().setMethod(storm::solver::GameMethod::ValueIteration);
        
        // policy iteration
        this->env.solver().game().setMethod(storm::solver::GameMethod::PolicyIteration);
        this->env.solver().setLinearEquationSolverType(storm::solver::EquationSolverType::Native);
        this->env.solver().native().setMethod(storm::solver::NativeLinearEquationSolverMethod::Jacobi);
        this->env.solver().setLinearEquationSolverPrecision(env.solver().game().getPrecision());
    }

    
    template<typename ValueType>
    storm::OptimizationDirection GameAbstractionSolver<ValueType>::getOptimizationDirection(bool maximizing) {
        return maximizing ? storm::OptimizationDirection::Maximize : storm::OptimizationDirection::Minimize;
    }


    template<typename ValueType>
    void GameAbstractionSolver<ValueType>::solveSg(
        storm::storage::BitVector const& quotient_choice_mask, storm::storage::BitVector const& stuck_choices
    ) {
        if(profiling_enabled) {
            this->timer_total.start();
        }

        uint64_t quotient_num_states = this->quotient.getNumberOfStates();
        uint64_t quotient_num_choices = this->quotient.getNumberOfChoices();
        uint64_t quotient_initial_state = *(this->quotient.getInitialStates().begin());

        ItemTranslator state_to_player1_state(quotient_num_states);
        ItemKeyTranslator<uint64_t> state_action_to_player2_state(quotient_num_states);
        std::vector<std::set<uint64_t>> player1_state_to_actions;
        std::vector<std::vector<uint64_t>> player2_state_to_choices;
        
        std::queue<uint64_t> unexplored_states;
        storm::storage::BitVector state_is_encountered(quotient_num_states);
        unexplored_states.push(quotient_initial_state);
        state_is_encountered.set(quotient_initial_state,true);
        auto const& quotient_row_group_indices = this->quotient.getTransitionMatrix().getRowGroupIndices();
        while(not unexplored_states.empty()) {
            uint64_t state = unexplored_states.front();
            unexplored_states.pop();
            uint64_t player1_state = state_to_player1_state.translate(state);
            player1_state_to_actions.resize(state_to_player1_state.numTranslations());
            for(uint64_t choice = quotient_row_group_indices[state]; choice < quotient_row_group_indices[state+1]; choice++) {
                if(not quotient_choice_mask[choice]) {
                    continue;
                }
                uint64_t action = choice_to_action[choice];
                player1_state_to_actions[player1_state].insert(action);
                uint64_t player2_state = state_action_to_player2_state.translate(state,action);
                player2_state_to_choices.resize(state_action_to_player2_state.numTranslations());
                player2_state_to_choices[player2_state].push_back(choice);
                for(uint64_t state_dst: this->choice_to_destinations[choice]) {
                    if(state_is_encountered[state_dst]) {
                        continue;
                    }
                    unexplored_states.push(state_dst);
                    state_is_encountered.set(state_dst,true);
                }
            }
        }
        uint64_t player1_num_states = state_to_player1_state.numTranslations();
        uint64_t player2_num_states = state_action_to_player2_state.numTranslations();
        
        // add fresh target states
        uint64_t player1_target_state = player1_num_states++;
        uint64_t player2_target_state = player2_num_states++;

        // build the matrix of Player 1
        std::vector<uint64_t> player1_choice_to_action;
        storm::storage::SparseMatrixBuilder<storm::storage::sparse::state_type> player1_matrix_builder(0,0,0,false,true);
        uint64_t player1_num_rows = 0;
        for(uint64_t player1_state=0; player1_state<player1_num_states-1; player1_state++) {
            player1_matrix_builder.newRowGroup(player1_num_rows);
            uint64_t state = state_to_player1_state.retrieve(player1_state);
            for(uint64_t action: player1_state_to_actions[player1_state]) {
                player1_choice_to_action.push_back(action);
                uint64_t player2_state = state_action_to_player2_state.translate(state,action);
                player1_matrix_builder.addNextValue(player1_num_rows,player2_state,1);
                player1_num_rows++;
            }
        }
        // fresh target state of Player 1
        player1_matrix_builder.newRowGroup(player1_num_rows);
        player1_matrix_builder.addNextValue(player1_num_rows,player2_target_state,1);
        player1_num_rows++;
        auto player1_matrix = player1_matrix_builder.build();

        
        // build the matrix of Player 2
        std::vector<uint64_t> player2_choice_to_quotient_choice;
        storm::storage::SparseMatrixBuilder<ValueType> player2_matrix_builder(0,0,0,false,true);
        // build the reward vector
        std::vector<double> player2_row_rewards;
        uint64_t player2_num_rows = 0;
        for(uint64_t player2_state=0; player2_state<player2_num_states-1; player2_state++) {
            player2_matrix_builder.newRowGroup(player2_num_rows);
            auto [state,action] = state_action_to_player2_state.retrieve(player2_state);
            if(state_is_target[state]) {
                // target state, transition to the target state of Player 1 and gain unit reward
                player2_matrix_builder.addNextValue(player2_num_rows,player1_target_state,1);
                player2_choice_to_quotient_choice.push_back(quotient_num_choices);
                player2_row_rewards.push_back(1);
                player2_num_rows++;
            } else {
                // state is not target
                bool may_get_stuck = false;
                for(auto choice: player2_state_to_choices[player2_state]) {
                    // transition to the corresponding states of Player 1 and gain zero reward
                    for(auto const& entry: this->quotient.getTransitionMatrix().getRow(choice)) {
                        auto state_dst = entry.getColumn();
                        auto player1_state_dst = state_to_player1_state.translate(state_dst);
                        player2_matrix_builder.addNextValue(player2_num_rows,player1_state_dst,entry.getValue());
                    }
                    player2_choice_to_quotient_choice.push_back(choice);
                    player2_row_rewards.push_back(0);
                    player2_num_rows++;
                    may_get_stuck = may_get_stuck or stuck_choices[choice];
                }
                if(may_get_stuck) {
                    // some member does not offer the action: stuck, with a unit reward (the target) if Player 1 minimizes and none if it maximizes; reported
                    // as the first choice of the action
                    player2_matrix_builder.addNextValue(player2_num_rows,player1_target_state,1);
                    player2_choice_to_quotient_choice.push_back(player2_state_to_choices[player2_state].front());
                    player2_row_rewards.push_back(this->player1_maximizing ? 0 : 1);
                    player2_num_rows++;
                }
            }
        }
        // fresh target state of Player 2: transition to the target state of Player 1 with zero reward
        player2_matrix_builder.newRowGroup(player2_num_rows);
        player2_matrix_builder.addNextValue(player2_num_rows,player1_target_state,1);
        player2_choice_to_quotient_choice.push_back(quotient_num_choices);
        player2_row_rewards.push_back(0);
        player2_num_rows++;
        auto player2_matrix = player2_matrix_builder.build();

        // solve the game
        auto solver = storm::solver::GameSolverFactory<ValueType>().create(env, player1_matrix, player2_matrix);
        solver->setTrackSchedulers(true);
        auto player1_direction = this->getOptimizationDirection(this->player1_maximizing);
        auto player2_direction = this->getOptimizationDirection(not this->player1_maximizing);
        std::vector<double> player1_state_values(player1_num_states,0);
        if(profiling_enabled) {
            this->timer_game_solving.start();
        }
        this->env.solver().game().setMaximalNumberOfIterations(this->max_policy_iterations);
        this->solution_converged = solver->solveGame(this->env, player1_direction, player2_direction, player1_state_values, player2_row_rewards);
        if(profiling_enabled) {
            this->timer_game_solving.stop();
        }
        auto player1_choices = solver->getPlayer1SchedulerChoices();
        auto player2_choices = solver->getPlayer2SchedulerChoices();

        // collect all the results
        std::fill(this->solution_state_values.begin(),this->solution_state_values.end(),0);
        std::fill(this->solution_state_to_player1_action.begin(),this->solution_state_to_player1_action.end(),quotient_num_actions);
        std::fill(this->solution_state_to_quotient_choice.begin(),this->solution_state_to_quotient_choice.end(),quotient_num_choices);

        auto const& player1_matrix_row_group_indices = player1_matrix.getRowGroupIndices();
        auto const& player2_matrix_row_group_indices = player2_matrix.getRowGroupIndices();

        for(uint64_t player1_state=0; player1_state<player1_num_states-1; player1_state++) {
            auto state = state_to_player1_state.retrieve(player1_state);
            this->solution_state_values[state] = player1_state_values[player1_state];

            // get action selected by Player 1
            auto player1_choice = player1_matrix_row_group_indices[player1_state] + player1_choices[player1_state];
            auto player1_action = player1_choice_to_action[player1_choice];
            this->solution_state_to_player1_action[state] = player1_action;

            if(this->state_is_target[state]) {
                auto state_only_choice = quotient_row_group_indices[state];
                this->solution_state_to_quotient_choice[state] = state_only_choice;
                continue;
            }

            // get action selected by Player 2 and map it to the quotient choice
            auto player2_state = state_action_to_player2_state.translate(state,player1_action);
            auto player2_choice = player2_matrix_row_group_indices[player2_state]+player2_choices[player2_state];
            auto quotient_choice = player2_choice_to_quotient_choice[player2_choice];
            this->solution_state_to_quotient_choice[state] = quotient_choice;
        }

        if(profiling_enabled) {
            this->timer_total.stop();
        }

        this->solution_value = this->solution_state_values[quotient_initial_state];
    }

    template<typename ValueType>
    void GameAbstractionSolver<ValueType>::solveSmg(storm::storage::BitVector const& quotient_choice_mask) {
        if(profiling_enabled) {
            this->timer_total.start();
        }

        SmgAbstraction<ValueType> abstraction = synthesis::SmgAbstraction<ValueType>(
            quotient,quotient_num_actions,choice_to_action,quotient_choice_mask
        );

        // solve the game
        if(profiling_enabled) {
            this->timer_game_solving.start();
        }
        std::shared_ptr<storm::modelchecker::CheckResult> result = synthesis::modelCheckSmg(*(abstraction.smg),this->game_formula,false,true,this->env);
        if(profiling_enabled) {
            this->timer_game_solving.stop();
        }
        storm::modelchecker::ExplicitQuantitativeCheckResult<ValueType> const& result_quan = result->asExplicitQuantitativeCheckResult<ValueType>();
        storm::storage::Scheduler<ValueType> const& scheduler = result_quan.getScheduler();

        std::fill(this->solution_state_values.begin(),this->solution_state_values.end(),0);
        std::fill(this->solution_state_to_player1_action.begin(),this->solution_state_to_player1_action.end(),this->quotient_num_actions);
        std::fill(this->solution_state_to_quotient_choice.begin(),this->solution_state_to_quotient_choice.end(),this->quotient.getNumberOfChoices());
        std::vector<uint64_t> const& game_row_groups = abstraction.smg->getTransitionMatrix().getRowGroupIndices();

        // get actions selected by Player 0
        for(uint64_t game_state = 0; game_state < abstraction.smg->getNumberOfStates(); ++game_state) {
            if(abstraction.smg->getPlayerOfState(game_state) != 0) {
                continue;
            }
            auto [state,_] = abstraction.state_to_quotient_state_action[game_state];
            this->solution_state_values[state] = result_quan[game_state];
            uint64_t game_choice = game_row_groups[game_state]+scheduler.getChoice(game_state).getDeterministicChoice();
            uint64_t player1_action = choice_to_action[abstraction.choice_to_quotient_choice[game_choice]];
            this->solution_state_to_player1_action[state] = player1_action;
        }
        // get action variants selected by Player 1 in corresponding states
        for(uint64_t game_state = 0; game_state < abstraction.smg->getNumberOfStates(); ++game_state) {
            if(abstraction.smg->getPlayerOfState(game_state) != 1) {
                continue;
            }
            auto [state,action] = abstraction.state_to_quotient_state_action[game_state];
            if(action !=  this->solution_state_to_player1_action[state]) {
                continue;
            }
            uint64_t game_choice = game_row_groups[game_state]+scheduler.getChoice(game_state).getDeterministicChoice();
            this->solution_state_to_quotient_choice[state] = abstraction.choice_to_quotient_choice[game_choice];
        }

        if(profiling_enabled) {
            this->timer_total.stop();
        }

        uint64_t quotient_initial_state = *(this->quotient.getInitialStates().begin());
        this->solution_value = this->solution_state_values[quotient_initial_state];
        this->solution_converged = true;
    }



    template <typename ValueType>
    void GameAbstractionSolver<ValueType>::enableProfiling(bool enable) {
        profiling_enabled = enable;
    }
 
    template <typename ValueType>
    void GameAbstractionSolver<ValueType>::printProfiling() {
        std::cout << "[s] total: " << this->timer_total << std::endl;
        std::cout << "[s]     game solving: " << this->timer_game_solving << std::endl;
    }


    template class GameAbstractionSolver<double>;
}