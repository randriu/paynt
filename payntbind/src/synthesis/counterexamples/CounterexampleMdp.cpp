#include "CounterexampleMdp.h"

#include <storm/storage/BitVector.h>
#include <storm/exceptions/InvalidArgumentException.h>
#include <storm/exceptions/NotImplementedException.h>

#include <storm/storage/sparse/JaniChoiceOrigins.h>
#include <storm/storage/valuations/Valuations.h>
#include <storm/utility/builder.h>
#include <storm/storage/SparseMatrix.h>
#include <storm/storage/sparse/ModelComponents.h>
#include <storm/models/sparse/StateLabeling.h>
#include <storm/solver/OptimizationDirection.h>
#include <storm/api/verification.h>
#include <storm/logic/Bound.h>
#include <storm/modelchecker/CheckTask.h>
#include <storm/modelchecker/hints/ExplicitModelCheckerHint.h>
#include <storm/environment/Environment.h>
#include <storm/environment/solver/SolverEnvironment.h>

namespace synthesis {



// labelStates
template <typename ValueType, typename StateType>
std::shared_ptr<storm::modelchecker::ExplicitQualitativeCheckResult<ValueType>> CounterexampleGeneratorMdp<ValueType,StateType>::labelStates(
    storm::models::sparse::Mdp<ValueType> const& mdp,
    storm::logic::Formula const& label
) {
    std::shared_ptr<storm::models::sparse::Mdp<ValueType>> mdp_shared = std::make_shared<storm::models::sparse::Mdp<ValueType>>(mdp);
    bool onlyInitialStatesRelevant = false;
    storm::modelchecker::CheckTask<storm::logic::Formula, ValueType> task(label, onlyInitialStatesRelevant);
    std::unique_ptr<storm::modelchecker::CheckResult> result_ptr = storm::api::verifyWithSparseEngine<ValueType>(mdp_shared, task);
    std::shared_ptr<storm::modelchecker::ExplicitQualitativeCheckResult<ValueType>> mdp_target = std::make_shared<storm::modelchecker::ExplicitQualitativeCheckResult<ValueType>>(result_ptr->asExplicitQualitativeCheckResult<ValueType>());
    return mdp_target;
}


// constructor
template <typename ValueType, typename StateType>
CounterexampleGeneratorMdp<ValueType,StateType>::CounterexampleGeneratorMdp (
    storm::models::sparse::Mdp<ValueType> const& quotient_mdp,
    uint64_t hole_count,
    std::vector<std::set<uint64_t>> const& quotient_holes,
    std::vector<std::shared_ptr<storm::logic::Formula const>> const& formulae
    ) : quotient_mdp(quotient_mdp), hole_count(hole_count), quotient_holes(quotient_holes) {

    // create label formulae for our own labels
    std::shared_ptr<storm::logic::Formula const> const& target_label_formula = std::make_shared<storm::logic::AtomicLabelFormula>(this->target_label);
    std::shared_ptr<storm::logic::Formula const> const& until_label_formula = std::make_shared<storm::logic::AtomicLabelFormula>(this->until_label);

    // process all formulae
    for(auto formula: formulae) {

        // store formula type and optimality type
        assert(formula->isOperatorFormula());
        storm::logic::OperatorFormula const& of = formula->asOperatorFormula();

        assert(of.hasOptimalityType());
        storm::solver::OptimizationDirection ot = of.getOptimalityType();
        bool is_safety = ot == storm::solver::OptimizationDirection::Minimize;
        this->formula_safety.push_back(is_safety);

        bool is_reward = formula->isRewardOperatorFormula();
        this->formula_reward.push_back(is_reward);
        if(!is_reward) {
            this->formula_reward_name.push_back("");
        } else {
            STORM_LOG_THROW(formula->asRewardOperatorFormula().hasRewardModelName(), storm::exceptions::InvalidArgumentException, "Name of the reward model must be specified.");
            this->formula_reward_name.push_back(formula->asRewardOperatorFormula().getRewardModelName());
        }

        // extract predicate for until and target states and identify such states
        storm::logic::Formula const& osf = of.getSubformula();
        if(!osf.isUntilFormula() && !osf.isEventuallyFormula()) {
            throw storm::exceptions::NotImplementedException() << "Only until or reachability formulae supported.";
        }

        std::shared_ptr<storm::logic::Formula const> modified_subformula;
        if(osf.isUntilFormula()) {
            storm::logic::UntilFormula const& uf = osf.asUntilFormula();

            auto mdp_until = this->labelStates(this->quotient_mdp,uf.getLeftSubformula());
            this->mdp_untils.push_back(mdp_until);

            auto mdp_target = this->labelStates(this->quotient_mdp, uf.getRightSubformula());
            this->mdp_targets.push_back(mdp_target);

            modified_subformula = std::make_shared<storm::logic::UntilFormula>(until_label_formula, target_label_formula);
        } else if(osf.isEventuallyFormula()) {
            storm::logic::EventuallyFormula const& ef = osf.asEventuallyFormula();

            this->mdp_untils.push_back(NULL);

            auto mdp_target = this->labelStates(this->quotient_mdp,ef.getSubformula());
            // std::cout << (*mdp_target).getTruthValuesVector() << std::endl;
            this->mdp_targets.push_back(mdp_target);

            modified_subformula = std::make_shared<storm::logic::EventuallyFormula>(target_label_formula, ef.getContext());
            // std::cout << (*modified_subformula).asEventuallyFormula().getSubformula() << std::endl;
        }

        // integrate formula into original context
        std::shared_ptr<storm::logic::Formula> modified_formula;
        if(!is_reward) {
            modified_formula = std::make_shared<storm::logic::ProbabilityOperatorFormula>(modified_subformula, of.getOperatorInformation());
        } else {
            modified_formula = std::make_shared<storm::logic::RewardOperatorFormula>(modified_subformula, this->formula_reward_name.back(), of.getOperatorInformation());
        }
        this->formula_modified.push_back(modified_formula);
    }
}

// prepareMdp
template <typename ValueType, typename StateType>
void CounterexampleGeneratorMdp<ValueType,StateType>::prepareMdp(
    storm::models::sparse::Mdp<ValueType> const& mdp,
    std::vector<uint64_t> const& state_map
    // storm::storage::BitVector initial_expand
    ) {

    // Get MDP info
    this->mdp = std::make_shared<storm::models::sparse::Mdp<ValueType>>(mdp);
    this->state_map = state_map;
    uint64_t mdp_states = this->mdp->getNumberOfStates();
    StateType initial_state = *(this->mdp->getInitialStates().begin());

    // Clear up previous MDP exploration metadata
    while(!this->state_horizon.empty()) {
        this->state_horizon.pop();
    }
    this->hole_wave.clear();
    this->wave_states.clear();
    this->state_horizon_blocking.clear();
    this->unregistered_holes_count = std::vector<uint64_t>(mdp_states);
    this->mdp_holes = std::vector<std::set<uint64_t>>(mdp_states);
    this->current_wave = 0;
    this->reachable_flag = storm::storage::BitVector(mdp_states, false);
    this->blocking_candidate_set = false;


    // Mark all simple holes registered and rest unregistered
    this->hole_wave.resize(this->hole_count,0);

    // Associate states of a MDP with relevant holes and store their count
    for(StateType state = 0; state < mdp_states; state++) {
        this->mdp_holes[state] = this->quotient_holes[state_map[state]];
        for(uint64_t hole : this->mdp_holes[state]) {
            // Hole is unregistered
            if(this->hole_wave[hole] == 0) {
                unregistered_holes_count[state]++;
            }
        }
    }

    // Round 0: encounter initial state first (important)
    this->wave_states.push_back(std::vector<StateType>());
    reachable_flag.set(initial_state);
    if(unregistered_holes_count[initial_state] == 0) {
        // non-blocking
        state_horizon.push(initial_state);
    } else {
        // blocking
        state_horizon_blocking.push_back(initial_state);
        blocking_candidate_set = true;
        blocking_candidate = initial_state;
    }
}

template <typename ValueType, typename StateType>
bool CounterexampleGeneratorMdp<ValueType,StateType>::exploreWave () {

    storm::storage::SparseMatrix<ValueType> const& transition_matrix = this->mdp->getTransitionMatrix();
    std::vector<uint64_t> row_group_indices = transition_matrix.getRowGroupIndices();
    uint64_t mdp_states = this->mdp->getNumberOfStates();

    // Expand the non-blocking horizon
    while(!state_horizon.empty()) {
        StateType state = state_horizon.top();
        state_horizon.pop();
        this->wave_states.back().push_back(state);

        // Reach successors
        for(uint64_t row_index = row_group_indices[state]; row_index < row_group_indices[state+1]; row_index++) {
            for(auto entry : transition_matrix.getRow(row_index)) {
                // TODO CHECK
                StateType successor = entry.getColumn();
                if(reachable_flag[successor]) {
                    // already reached
                    continue;
                }
                // new state reached
                reachable_flag.set(successor);
                if(unregistered_holes_count[successor] == 0) {
                    // non-blocking
                    state_horizon.push(successor);
                } else {
                    // blocking
                    state_horizon_blocking.push_back(successor);
                    if(!blocking_candidate_set || unregistered_holes_count[successor] < unregistered_holes_count[blocking_candidate]) {
                        // new blocking candidate
                        blocking_candidate_set = true;
                        blocking_candidate = successor;
                    }
                }
            }
        }
    }

    // Non-blocking horizon exhausted
    if(!blocking_candidate_set) {
        // fully explored - nothing more to expand
        return true;
    }

    // Start a new wave
    this->current_wave++;
    this->wave_states.push_back(std::vector<StateType>());
    blocking_candidate_set = false;

    // Register all unregistered holes of this blocking state
    for(uint64_t hole: mdp_holes[blocking_candidate]) {
        if(this->hole_wave[hole] == 0) {
            hole_wave[hole] = current_wave;
            // std::cout << "[storm] hole " << hole << " expanded in wave " << current_wave << std::endl;
        }
    }

    // Recompute number of unregistered holes in each state
    for(StateType state = 0; state < mdp_states; state++) {
        unregistered_holes_count[state] = 0;
        for(uint64_t hole: mdp_holes[state]) {
            if(this->hole_wave[hole] == 0) {
                unregistered_holes_count[state]++;
            }
        }
    }

    // Unblock the states from the blocking horizon
    std::vector<StateType> old_blocking_horizon;
    old_blocking_horizon.swap(state_horizon_blocking);
    for(StateType state: old_blocking_horizon) {
        if(unregistered_holes_count[state] == 0) {
            // state unblocked
            state_horizon.push(state);
        } else {
            // still blocking
            state_horizon_blocking.push_back(state);
            if(!blocking_candidate_set || unregistered_holes_count[state] < unregistered_holes_count[blocking_candidate]) {
                // new blocking candidate
                blocking_candidate_set = true;
                blocking_candidate = state;
            }
        }
    }

    // not fully explored
    return false;
}

// prepareSubmdp
template <typename ValueType, typename StateType>
void CounterexampleGeneratorMdp<ValueType,StateType>::prepareSubmdp (
    uint64_t formula_index,
    std::shared_ptr<storm::modelchecker::ExplicitQuantitativeCheckResult<ValueType> const> mdp_bounds,
    std::vector<StateType> const& mdp_quotient_state_map,
    std::vector<std::vector<StormRow>> & matrix_submdp,
    storm::models::sparse::StateLabeling & labeling_submdp,
    std::unordered_map<std::string,storm::models::sparse::StandardRewardModel<ValueType>> & reward_models_submdp
    ) {

    // Get MDP info
    StateType mdp_states = mdp->getNumberOfStates();
    storm::storage::SparseMatrix<ValueType> const& transition_matrix = this->mdp->getTransitionMatrix();

    // Introduce expanded state space
    uint64_t sink_state_false = mdp_states;
    uint64_t sink_state_true = mdp_states+1;

    // Label target states of a MDP
    std::shared_ptr<storm::modelchecker::ExplicitQualitativeCheckResult<ValueType> const> mdp_target = this->mdp_targets[formula_index];
    std::shared_ptr<storm::modelchecker::ExplicitQualitativeCheckResult<ValueType> const> mdp_until = this->mdp_untils[formula_index];
    labeling_submdp.addLabel(this->target_label);
    labeling_submdp.addLabel(this->until_label);
    for(StateType state = 0; state < mdp_states; state++) {
        StateType mdp_state = this->state_map[state];
        if((*mdp_target)[mdp_state]) {
            labeling_submdp.addLabelToState(this->target_label, state);
        }
        if(mdp_until != NULL && (*mdp_until)[mdp_state]) {
            labeling_submdp.addLabelToState(this->until_label, state);
        }
    }
    // Associate true sink with the target label
    labeling_submdp.addLabelToState(this->target_label, sink_state_true);

    // Map MDP bounds onto the state space of a quotient MDP
    bool have_bounds = mdp_bounds != NULL;
    std::vector<ValueType> quotient_mdp_bounds;
    if(have_bounds) {
        auto const& mdp_values = mdp_bounds->getValueVector();
        quotient_mdp_bounds.resize(this->quotient_mdp.getNumberOfStates());
        uint64_t mdp_states = mdp_values.size();
        for(StateType state = 0; state < mdp_states; state++) {
            quotient_mdp_bounds[mdp_quotient_state_map[state]] = mdp_values[state];
        }
    }

    // Construct transition matrix (as well as the reward model) for the submdp
    if(!this->formula_reward[formula_index]) {
        // Probability formula: no reward models
        double default_bound = this->formula_safety[formula_index] ? 0 : 1;
        for(StateType state = 0; state < mdp_states; state++) {
            // matrix_submdp.push_back(std::vector<StormRow>());
            StateType mdp_state = this->state_map[state];

            // FIXME later? - multiple same actions
            uint64_t state_actions = transition_matrix.getRowGroupSize(state);
            for(uint64_t action = 0; action < state_actions; action++) {
                StormRow r;
                double probability = have_bounds ? quotient_mdp_bounds[mdp_state] : default_bound;
                r.emplace_back(sink_state_false, 1-probability);
                r.emplace_back(sink_state_true, probability);
                matrix_submdp[state].push_back(r);
            }
        }
    } else {
        // Reward formula: one reward model
        assert(mdp->hasRewardModel(this->formula_reward_name[formula_index]));

        std::optional<std::vector<ValueType>> state_rewards;
        std::vector<ValueType> state_action_rewards_submdp(transition_matrix.getRowCount()+2);
        double default_reward = 0;
        uint64_t row_index = 0;
        for(StateType state = 0; state < mdp_states; state++) {
            StateType mdp_state = this->state_map[state];
            // matrix_submdp.push_back(std::vector<StormRow>());
            // FIXME - quotient_mdp_bounds
            double reward = have_bounds ? quotient_mdp_bounds[mdp_state] : default_reward;

            // FIXME later? - multiple same actions
            uint64_t state_actions = transition_matrix.getRowGroupSize(state);
            for(uint64_t action = 0; action < state_actions; action++) {
                state_action_rewards_submdp[row_index] = reward;
                row_index++;
                StormRow r;
                r.emplace_back(sink_state_true, 1);
                matrix_submdp[state].push_back(r);
            }


        }
        storm::models::sparse::StandardRewardModel<ValueType> reward_model_submdp(state_rewards, state_action_rewards_submdp);
        reward_models_submdp.emplace(this->formula_reward_name[formula_index], reward_model_submdp);
    }

    // Add self-loops to sink states
    for(StateType state = sink_state_false; state <= sink_state_true; state++) {
        StormRow r;
        r.emplace_back(state, 1);
        matrix_submdp[state].push_back(r);
    }
}

// expandAndCheck
template <typename ValueType, typename StateType>
std::pair<bool,bool> CounterexampleGeneratorMdp<ValueType,StateType>::expandAndCheck (
    uint64_t formula_index,
    ValueType formula_bound,
    bool strict,
    // std::vector<std::vector<std::pair<StateType,ValueType>>> & matrix_submdp,
    std::vector<std::vector<StormRow>> & matrix_submdp,
    storm::models::sparse::StateLabeling const& labeling_submdp,
    std::unordered_map<std::string,storm::models::sparse::StandardRewardModel<ValueType>> & reward_models_submdp,
    storm::Environment const& env
) {
    // result.first - wave exploration finish status
    // result.second - formula satisfied
    std::pair<bool,bool> result(false, true);

    // explore one row_indexrow_index
    // FIXME later? - we don't need to explore last wave
    bool fully_explored = exploreWave();
    result.first = fully_explored;

    // all waves explored
    if(fully_explored) {
        return result;
    }


    // Get MDP info
    uint64_t mdp_states = this->mdp->getNumberOfStates();
    storm::storage::SparseMatrix<ValueType> const& transition_matrix = this->mdp->getTransitionMatrix();
    std::vector<uint64_t> row_group_indices = transition_matrix.getRowGroupIndices();
    StateType initial_state = *(this->mdp->getInitialStates().begin());
    std::vector<StateType> to_expand = this->wave_states[current_wave-1];
    // Expand states from the new wave:
    // - expand transition probabilities
    for(StateType state : to_expand) {
        matrix_submdp[state].clear();

        for(uint64_t row_index = row_group_indices[state]; row_index < row_group_indices[state+1]; row_index++) {
            StormRow r;
            for(auto entry : transition_matrix.getRow(row_index)) {
                r.emplace_back(entry.getColumn(), entry.getValue());
            }
            matrix_submdp[state].push_back(r);
        }
    }

    if(this->formula_reward[formula_index]) {
        // - expand state rewards
        storm::models::sparse::StandardRewardModel<ValueType> const& reward_model_mdp = mdp->getRewardModel(this->formula_reward_name[formula_index]);
        assert(reward_model_mdp.hasStateActionRewards());
        storm::models::sparse::StandardRewardModel<ValueType> & reward_model_submdp = (reward_models_submdp.find(this->formula_reward_name[formula_index]))->second;

        for(StateType state : to_expand) {
            for(uint64_t row_index = row_group_indices[state]; row_index < row_group_indices[state+1]; row_index++) {
                ValueType reward = reward_model_mdp.getStateActionReward(row_index);
                reward_model_submdp.setStateActionReward(row_index, reward);
            }
        }
    }

    // Construct sub-MDP
    storm::storage::SparseMatrixBuilder<ValueType> transitionMatrixBuilder(0, 0, 0, false, true, row_group_indices.size());
    uint64_t row_index = 0;
    for(StateType state = 0; state < mdp_states+2; state++) {
        transitionMatrixBuilder.newRowGroup(row_index);
        for(auto row: matrix_submdp[state]) {
            for(auto row_entry: row) {
                transitionMatrixBuilder.addNextValue(row_index, row_entry.first, row_entry.second);
            }
            row_index++;
        }
    }
    storm::storage::SparseMatrix<ValueType> sub_matrix = transitionMatrixBuilder.build();
    // std::cout << sub_matrix << std::endl;
    // zero tolerance is too strict for floating-point row sums (a row can be a few ULPs off from 1);
    // match storm's own default --precision (1e-6, see GeneralSettings::precisionOptionName)
    assert(sub_matrix.isProbabilistic(storm::utility::convertNumber<ValueType>(1e-6)));
    storm::storage::sparse::ModelComponents<ValueType> components(sub_matrix, labeling_submdp, reward_models_submdp);
    std::shared_ptr<storm::models::sparse::Model<ValueType>> submdp = storm::utility::builder::buildModelFromComponents(storm::models::ModelType::Mdp, std::move(components));
    // std::cout << "[storm] sub-mdp has " << submdp->getNumberOfStates() << " states" << std::endl;


    // Construct MC task
    bool onlyInitialStatesRelevant = false;
    storm::modelchecker::CheckTask<storm::logic::Formula, ValueType> task(*(this->formula_modified[formula_index]), onlyInitialStatesRelevant);
    if(this->hint_result != NULL) {
        // Add hints from previous wave
        storm::modelchecker::ExplicitModelCheckerHint<ValueType> hint;
        hint.setComputeOnlyMaybeStates(false);
        hint.setResultHint(boost::make_optional(this->hint_result->template asExplicitQuantitativeCheckResult<ValueType>().getValueVector()));
        task.setHint(std::make_shared<storm::modelchecker::ExplicitModelCheckerHint<ValueType>>(hint));
    }
    // Model check
    // std::unique_ptr<storm::modelchecker::CheckResult> result_ptr = storm::api::verifyWithSparseEngine<ValueType>(submdp, task);
    // storm::modelchecker::ExplicitQuantitativeCheckResult<ValueType>& result = result_ptr->asExplicitQuantitativeCheckResult<ValueType>();
    this->timer_model_check.start();
    this->hint_result = storm::api::verifyWithSparseEngine<ValueType>(env, submdp, task);
    this->timer_model_check.stop();
    storm::modelchecker::ExplicitQuantitativeCheckResult<ValueType>& model_check_result = this->hint_result->template asExplicitQuantitativeCheckResult<ValueType>();

    // the formulae have no bound, so the comparison comes from the caller
    ValueType value = model_check_result[initial_state];
    if(this->formula_safety[formula_index]) {
        result.second = strict ? value < formula_bound : value <= formula_bound;
    } else {
        result.second = strict ? value > formula_bound : value >= formula_bound;
    }

    return result;
}


// constructConflict
template <typename ValueType, typename StateType>
std::vector<uint64_t> CounterexampleGeneratorMdp<ValueType,StateType>::constructConflict (
    uint64_t formula_index,
    ValueType formula_bound,
    bool strict,
    // TODO: revise rest of the args
    std::shared_ptr<storm::modelchecker::ExplicitQuantitativeCheckResult<ValueType> const> mdp_bounds, // placeholders for Hybrid
    std::vector<StateType> const& mdp_quotient_state_map, // maybe not needed as we don't need to quotient MDP
    storm::Environment const& env
) {
    this->timer_conflict.start();
    StateType mdp_states = this->mdp->getNumberOfStates();

    // Prepare to construct sub-MDPs
    std::vector<std::vector<StormRow>> matrix_submdp(mdp_states+2);
    storm::models::sparse::StateLabeling labeling_submdp(mdp_states+2);
    std::unordered_map<std::string, storm::models::sparse::StandardRewardModel<ValueType>> reward_models_submdp;
    this->prepareSubmdp(
        formula_index, mdp_bounds, mdp_quotient_state_map, matrix_submdp, labeling_submdp, reward_models_submdp
    );

    bool violated = false;
    while(true) {
        std::pair<bool,bool> result = this->expandAndCheck(
            formula_index, formula_bound, strict, matrix_submdp, labeling_submdp, reward_models_submdp, env
        );
        bool last_wave = result.first;
        bool satisfied = result.second;

        if(!satisfied) {
            violated = true;
            break;
        }
        if(last_wave) {
            break;
        }
    }

    // Return a set of critical holes
    // exploreWave registers the holes of the next wave before the current one is expanded, so a violated check has not seen the current wave's holes
    uint64_t conflict_wave = violated ? current_wave - 1 : current_wave;
    std::vector<uint64_t> critical_holes;
    for(uint64_t hole = 0; hole < this->hole_count; hole++) {
        uint64_t wave_registered = this->hole_wave[hole];
        if(wave_registered > 0 && wave_registered <= conflict_wave) {
            critical_holes.push_back(hole);
        }
    }
    this->timer_conflict.stop();

    return critical_holes;
}

// printProfiling
template <typename ValueType, typename StateType>
void CounterexampleGeneratorMdp<ValueType,StateType>::printProfiling() {
    std::cout << "[s] conflict: " << this->timer_conflict << std::endl;
    std::cout << "[s]     model checking: " << this->timer_model_check << std::endl;
}

 // Explicitly instantiate functions and classes.
template class CounterexampleGeneratorMdp<double, uint64_t>;

}
