#include "ColoringSmt.h"

#include "src/synthesis/translation/choiceTransformation.h"

#include <storm/storage/valuations/Valuations.h>
#include <storm/exceptions/UnexpectedException.h>

#include <algorithm>
#include <sstream>

namespace synthesis {


template<typename ValueType>
ColoringSmt<ValueType>::ColoringSmt(
    std::vector<uint64_t> const& row_groups,
    std::vector<uint64_t> const& choice_to_action,
    uint64_t num_actions,
    uint64_t dont_care_action,
    storm::storage::sparse::Valuations const& state_valuations,
    BitVector const& state_is_relevant,
    std::vector<std::string> const& variable_name,
    std::vector<std::vector<int64_t>> const& variable_domain,
    std::vector<std::tuple<uint64_t,uint64_t,uint64_t>> const& tree_list,
    bool enable_harmonization
) : row_groups(row_groups), choice_to_action(choice_to_action), num_actions(num_actions), dont_care_action(dont_care_action),
    state_is_relevant(state_is_relevant),
    variable_name(variable_name), variable_domain(variable_domain),
    solver(ctx), harmonizing_variable(ctx), enable_harmonization(enable_harmonization) {

    // timers[__FUNCTION__].start();
    // timers["ColoringSmt::0"].start();

    for(uint64_t state = 0; state < numStates(); ++state) {
        for(uint64_t choice = row_groups[state]; choice < row_groups[state+1]; ++choice) {
            choice_to_state.push_back(state);
        }
    }

    // identify available actions
    for(uint64_t state = 0; state < numStates(); ++state) {
        BitVector action_available(num_actions,false);
        for(uint64_t choice = row_groups[state]; choice < row_groups[state+1]; ++choice) {
            action_available.set(this->choice_to_action[choice],true);
        }
        this->state_available_actions.push_back(action_available);
    }

    // extract variables in the order of variable_name
    std::vector<storm::expressions::Variable> program_variables;
    auto const& all_variables = state_valuations.getAllVariables();
    for(std::string const& name: variable_name) {
        bool variable_found = false;
        for(auto const& program_variable: all_variables) {
            if(program_variable.getName() == name) {
                program_variables.push_back(program_variable);
                variable_found = true;
                break;
            }
        }
        STORM_LOG_THROW(variable_found, storm::exceptions::UnexpectedException, "Unexpected variable name.");
    }

    // create the tree
    uint64_t num_nodes = tree_list.size();
    this->num_actions = *std::max_element(choice_to_action.begin(),choice_to_action.end())+1;
    for(uint64_t node = 0; node < num_nodes; ++node) {
        auto [parent,child_true,child_false] = tree_list[node];
        STORM_LOG_THROW(
            (child_true != num_nodes) == (child_false != num_nodes), storm::exceptions::UnexpectedException,
            "Inner node has only one child."
        );
        if(child_true != num_nodes) {
            tree.push_back(std::make_shared<InnerNode>(node,ctx,this->variable_name,this->variable_domain));
        } else {
            tree.push_back(std::make_shared<TerminalNode>(node,ctx,this->variable_name,this->variable_domain,this->num_actions));
        }
    }
    getRoot()->createTree(tree_list,tree);
    getRoot()->createHoles(family);
    harmonizing_variable = ctx.int_const("__harm__");
    getRoot()->createPaths(harmonizing_variable);

    for(uint64_t state = 0; state < numStates(); ++state) {
        state_path_enabled.push_back(BitVector(numPaths()));
    }

    // store state valuations in terms of hole options
    state_valuation.resize(numStates());
    for(uint64_t state: state_is_relevant) {
        for(uint64_t variable = 0; variable < variable_name.size(); ++variable) {
            storm::expressions::Variable const& program_variable = program_variables[variable];
            int64_t value;
            if(program_variable.hasBooleanType()) {
                value = (int64_t)state_valuations.getBooleanValue(state,program_variable);
            } else {
                value = state_valuations.getInt64Value(state,program_variable);
            }
            bool domain_option_found = false;
            for(uint64_t domain_option = 0; domain_option < variable_domain[variable].size(); ++domain_option) {
                if(variable_domain[variable][domain_option] == value) {
                    state_valuation[state].push_back(domain_option);
                    domain_option_found = true;
                    break;
                }
            }
            STORM_LOG_THROW(domain_option_found, storm::exceptions::UnexpectedException, "Hole option not found.");
        }
    }
    // timers["ColoringSmt::0"].stop();

    // create choice colors
    // timers["ColoringSmt::1 create choice colors"].start();
    // std::cout << "ColoringSmt::1 create choice colors" << std::endl << std::flush;

    for(std::vector<bool> const& path: getRoot()->paths) {
        path_action_hole.push_back(getRoot()->getPathActionHole(path));
    }
    choice_path_label.resize(numChoices());
    for(uint64_t state: state_is_relevant) {
        for(uint64_t choice = row_groups[state]; choice < row_groups[state+1]; ++choice) {
            for(uint64_t path = 0; path < numPaths(); ++path) {
                std::string label = "p" + std::to_string(choice) + "_" + std::to_string(path);
                choice_path_label[choice].push_back(label);
            }
        }
    }

    std::vector<const TerminalNode*> terminals;
    for(uint64_t path = 0; path < numPaths(); ++path) {
        terminals.push_back(getRoot()->getTerminal(getRoot()->paths[path]));
    }

    // allocate array for path expressions
    uint64_t longest_path = 0;
    for(uint64_t path = 0; path < numPaths(); ++path) {
        longest_path = std::max(longest_path,(uint64_t)getRoot()->paths[path].size());
    }
    z3::expr_vector state_valuation_int(ctx);
    z3::array<Z3_ast> clause_array(longest_path-1+num_actions);

    getRoot()->substituteActionExpressions();
    choice_path_expresssion.resize(numChoices());
    for(uint64_t state = 0; state < numStates(); ++state) {
        if(not state_is_relevant[state]) {
            continue;
        }

        for(uint64_t value: state_valuation[state]) {
            state_valuation_int.push_back(ctx.int_val(value));
        }
        // timers["ColoringSmt::1-2 createPrefixSubstitutions"].start();
        getRoot()->createPrefixSubstitutions(state_valuation[state], state_valuation_int);
        // timers["ColoringSmt::1-2 createPrefixSubstitutions"].stop();
        state_valuation_int.resize(0);

        // timers["ColoringSmt::1-3"].start();
        for(uint64_t path = 0; path < numPaths(); ++path) {
            // timers["ColoringSmt::1-3-1"].start();
            getRoot()->substitutePrefixExpression(getRoot()->paths[path], clause_array);
            // timers["ColoringSmt::1-3-1"].stop();

            // timers["ColoringSmt::1-3-2"].start();
            for(uint64_t choice = row_groups[state]; choice < row_groups[state+1]; ++choice) {
                uint64_t num_clauses = getRoot()->paths[path].size()-1;
                uint64_t action = choice_to_action[choice];
                clause_array[num_clauses++] = terminals[path]->action_expression[action];
                if(action == dont_care_action) {
                    for(uint64_t unavailable_action: ~state_available_actions[state]) {
                        clause_array[num_clauses++] = terminals[path]->action_expression[unavailable_action];
                    }
                }
                choice_path_expresssion[choice].push_back(z3::expr(ctx, Z3_mk_or(ctx, num_clauses, clause_array.ptr())));
                // choice_path_expresssion[choice].push_back(Z3_mk_or(ctx, num_clauses, clause_array.ptr()));
            }
            // timers["ColoringSmt::1-3-2"].stop();
        }
        // timers["ColoringSmt::1-3"].stop();
    }
    // timers["ColoringSmt::1 create choice colors"].stop();

    if(not this->enable_harmonization) {
        // timers[__FUNCTION__].stop();
        return;
    }

    // timers["ColoringSmt::2 create harmonizing variants"].start();
    // std::cout << "ColoringSmt::2 create harmonizing variants" << std::endl << std::flush;

    getRoot()->substituteActionExpressionsHarmonizing(harmonizing_variable);
    choice_path_expresssion_harm.resize(numChoices());
    for(uint64_t state = 0; state < numStates(); ++state) {
        if(not state_is_relevant[state]) {
            continue;
        }

        for(uint64_t value: state_valuation[state]) {
            state_valuation_int.push_back(ctx.int_val(value));
        }
        // timers["ColoringSmt::2-2 createPrefixSubstitutionsHarmonizing"].start();
        getRoot()->createPrefixSubstitutionsHarmonizing(state_valuation[state], state_valuation_int, harmonizing_variable);
        // timers["ColoringSmt::2-2 createPrefixSubstitutionsHarmonizing"].stop();
        state_valuation_int.resize(0);

        // timers["ColoringSmt::2-3"].start();
        for(uint64_t path = 0; path < numPaths(); ++path) {
            // timers["ColoringSmt::2-3-1"].start();
            getRoot()->substitutePrefixExpressionHarmonizing(getRoot()->paths[path], clause_array);
            // timers["ColoringSmt::2-3-1"].stop();

            // timers["ColoringSmt::2-3-2"].start();
            for(uint64_t choice = row_groups[state]; choice < row_groups[state+1]; ++choice) {
                uint64_t action = choice_to_action[choice];
                uint64_t num_clauses = getRoot()->paths[path].size()-1;
                clause_array[num_clauses++] = terminals[path]->action_expression_harmonizing[action];
                if(action == dont_care_action) {
                    for(uint64_t unavailable_action: ~state_available_actions[state]) {
                        clause_array[num_clauses++] = terminals[path]->action_expression_harmonizing[unavailable_action];
                    }
                }
                choice_path_expresssion_harm[choice].push_back(z3::expr(ctx, Z3_mk_or(ctx, num_clauses, clause_array.ptr())));
                // choice_path_expresssion_harm[choice].push_back(Z3_mk_or(ctx, num_clauses, clause_array.ptr()));
            }
            // timers["ColoringSmt::2-3-2"].stop();
        }
        // timers["ColoringSmt::2-3"].stop();
    }
    // timers["ColoringSmt::2 create harmonizing variants"].stop();

    getRoot()->clearCache();

    // timers[__FUNCTION__].stop();
}

template<typename ValueType>
ColoringSmt<ValueType>::~ColoringSmt() {
    tree.clear();
}

template<typename ValueType>
void ColoringSmt<ValueType>::enableStateExploration(storm::models::sparse::NondeterministicModel<ValueType> const& model) {
    this->state_exploration_enabled = true;
    this->initial_state = *model.getInitialStates().begin();
    this->choice_destinations = synthesis::computeChoiceDestinations(model);
}

template<typename ValueType>
const uint64_t ColoringSmt<ValueType>::numStates() const {
    return row_groups.size()-1;
}

template<typename ValueType>
const uint64_t ColoringSmt<ValueType>::numChoices() const {
    return row_groups.back();
}

template<typename ValueType>
const bool ColoringSmt<ValueType>::dontCareActionDefined() const {
    return dont_care_action < num_actions;
}

template<typename ValueType>
const uint64_t ColoringSmt<ValueType>::numVariables() const {
    return variable_name.size();
}

template<typename ValueType>
uint64_t ColoringSmt<ValueType>::numNodes() const {
    return tree.size();
}

template<typename ValueType>
std::shared_ptr<TreeNode> ColoringSmt<ValueType>::getRoot() {
    return tree[0];
}

template<typename ValueType>
uint64_t ColoringSmt<ValueType>::numPaths() {
    return getRoot()->paths.size();
}

template<typename ValueType>
bool ColoringSmt<ValueType>::check() {
    // timers[__FUNCTION__].start();
    bool sat = solver.check() == z3::sat;
    // timers[__FUNCTION__].stop();
    return sat;
}

template<typename ValueType>std::vector<std::tuple<uint64_t,std::string,std::string>> ColoringSmt<ValueType>::getFamilyInfo() {
    std::vector<std::tuple<uint64_t,std::string,std::string>> hole_info(family.numHoles());
    getRoot()->loadHoleInfo(hole_info);
    return hole_info;
}

template<typename ValueType>
void ColoringSmt<ValueType>::visitChoice(uint64_t choice, BitVector & state_reached, std::queue<uint64_t> & unexplored_states) {
    if(not this->state_exploration_enabled) {
        return;
    }
    for(uint64_t dst: choice_destinations[choice]) {
        if(not state_reached[dst]) {
            unexplored_states.push(dst);
            state_reached.set(dst,true);
        }
    }
}

template<typename ValueType>
BitVector ColoringSmt<ValueType>::selectCompatibleChoices(Family const& subfamily) {
    return selectCompatibleChoices(subfamily,BitVector(numChoices(),true));
}

template<typename ValueType>
BitVector ColoringSmt<ValueType>::selectCompatibleChoices(Family const& subfamily, BitVector const& base_choices) {
    // timers[__FUNCTION__].start();

    if(CHECK_FAMILY_CONSISTENCE) {
        // check if the subfamily itself satisfies hole restrictions
        // timers["selectCompatibleChoices::1 is family sat"].start();
        solver.push();
        getRoot()->addFamilyEncoding(subfamily,solver);
        bool subfamily_sat = check();
        solver.pop();
        // timers["selectCompatibleChoices::1 is family sat"].stop();
        STORM_LOG_THROW(
            subfamily_sat, storm::exceptions::UnexpectedException,
            "family is UNSAT (?)"
        );
    }

    // check individual choices
    // timers["selectCompatibleChoices::2 state exploration"].start();

    BitVector choice_selection(numChoices(),false);
    std::queue<uint64_t> unexplored_states;
    BitVector state_reached(numStates(),false);
    if(this->state_exploration_enabled) {
        unexplored_states.push(initial_state);
        state_reached.set(initial_state,true);
    } else {
        for(uint64_t state = 0; state < numStates(); ++state) {
            unexplored_states.push(state);
        }
    }

    while(not unexplored_states.empty()) {
        uint64_t state = unexplored_states.front(); unexplored_states.pop();
        if(not state_is_relevant[state]) {
            // enable the first choice
            for(uint64_t choice = row_groups[state]; choice < row_groups[state+1]; ++choice) {
                if(not base_choices[choice]) {
                    continue;
                }
                choice_selection.set(choice,true);
                visitChoice(choice,state_reached,unexplored_states);
                break;
            }
            continue;
        }
        state_path_enabled[state].clear();
        getRoot()->arePathsEnabledInState(subfamily,state_valuation[state]);
        for(uint64_t path = 0; path < numPaths(); ++path) {
            bool path_enabled = getRoot()->isPathEnabled(getRoot()->paths[path]);
            state_path_enabled[state].set(path,path_enabled);
        }

        bool any_choice_enabled = false;
        for(uint64_t choice = row_groups[state]; choice < row_groups[state+1]; ++choice) {
            uint64_t action = choice_to_action[choice];
            if(not base_choices[choice]) {
                continue;
            }
            bool choice_enabled = false;
            for(uint64_t path: state_path_enabled[state]) {
                uint64_t path_hole = path_action_hole[path];
                // enable the choice if this action is the family
                choice_enabled = subfamily.holeContains(path_hole,action);
                if(not choice_enabled and action == this->dont_care_action) {
                    // don't-care action can also be enabled if any unavailable action is in the family
                    for(uint64_t unavailable_action: ~state_available_actions[state]) {
                        if(subfamily.holeContains(path_hole,unavailable_action)) {
                            choice_enabled = true;
                            break;
                        }
                    }
                }
                if(choice_enabled) {
                    break;
                }
            }
            if(choice_enabled) {
                any_choice_enabled = true;
                choice_selection.set(choice,true);
                visitChoice(choice,state_reached,unexplored_states);
            }
        }
        STORM_LOG_THROW(any_choice_enabled, storm::exceptions::UnexpectedException, "no choice is available in the sub-MDP");
    }

    // timers["selectCompatibleChoices::2 state exploration"].stop();

    /*if(CHECK_CONSISTENT_SCHEDULER_EXISTENCE) {
        // check selected choices simultaneously
        solver.push();
        getRoot()->addFamilyEncoding(subfamily,solver);
        for(uint64_t state: state_reached) {
            z3::expr_vector enabled_choices(ctx);
            for(uint64_t choice: state_enabled_choices[state]) {
                // enabled_choices.push_back(z3::mk_and(choice_path_expresssion[choice]));
                enabled_choices.push_back(choice_path_expresssion_expr[choice]);
            }
            solver.add(z3::mk_or(enabled_choices));
        }
        bool consistent_scheduler_exists = check();
        if(not consistent_scheduler_exists) {
            STORM_LOG_WARN_COND(not subfamily.isAssignment(), "Hole assignment does not induce a DTMC.");
            choice_selection.clear();
        }
        solver.pop();
    }*/

    // timers[__FUNCTION__].stop();
    return choice_selection;
}

template<typename ValueType>
void ColoringSmt<ValueType>::loadUnsatCore(z3::expr_vector const& unsat_core_expr, Family const& subfamily) {
    // timers[__FUNCTION__].start();
    this->unsat_core.clear();
    for(z3::expr expr: unsat_core_expr) {
        std::istringstream iss(expr.decl().name().str());
        char prefix; iss >> prefix;
        if(prefix == 'h' or prefix == 'z') {
            // uint64_t hole; iss >> prefix; iss >> hole;
            continue;
        }
        // prefix == 'p'
        uint64_t choice,path; iss >> choice; iss >> prefix; iss >> path;
        this->unsat_core.emplace_back(choice,path);
        if(PRINT_UNSAT_CORE) {
            bool action_enabled = subfamily.holeContains(path_action_hole[path],choice_to_action[choice]);
            std::cout << "choice = " << choice << ", path = " << path << ", enabled = " << action_enabled << std::endl;
            std::cout << choice_path_expresssion[choice][path] << std::endl;
        }
    }
    // timers[__FUNCTION__].stop();
    return;

    /*for(uint64_t index = 0; index < this->unsat_core.size()-1; ++index) {
        auto [choice,path] = this->unsat_core[index];
        solver.push();
        solver.add(choice_path_expresssion[choice][path]);
    }
    while(true) {
        if(check()) {
            // pop remaining expressions
            for(uint64_t index = 0; index < this->unsat_core.size()-1; ++index) {
                solver.pop();
            }
            break;
        }
        this->unsat_core.pop_back();
        solver.pop();
    }
    // timers[__FUNCTION__].stop();*/
}

template<typename ValueType>
void ColoringSmt<ValueType>::loadUnsatCore(z3::expr_vector const& unsat_core_expr, Family const& subfamily, BitVector const& choices) {
    // timers[__FUNCTION__].start();
    this->unsat_core.clear();
    std::set<uint64_t> critical_states;
    for(z3::expr expr: unsat_core_expr) {
        std::istringstream iss(expr.decl().name().str());
        char prefix; iss >> prefix;
        if(prefix == 'h' or prefix == 'z') {
            // uint64_t hole; iss >> prefix; iss >> hole;
            continue;
        }
        // prefix == 'p'
        uint64_t choice,path; iss >> choice; iss >> prefix; iss >> path;
        uint64_t state = this->choice_to_state[choice];
        critical_states.insert(state);
    }
    for(uint64_t state: critical_states) {
        for(uint64_t choice = row_groups[state]; choice < row_groups[state+1]; ++choice) {
            if(not choices[choice]) {
                continue;
            }
            for(uint64_t path: state_path_enabled[state]) {
                this->unsat_core.emplace_back(choice,path);
            }
        }
    }

    // timers[__FUNCTION__].stop();
    return;
}

template<typename ValueType>
std::pair<bool,std::vector<std::vector<uint64_t>>> ColoringSmt<ValueType>::areChoicesConsistent(BitVector const& choices, Family const& subfamily) {
    // timers[__FUNCTION__].start();
    std::vector<std::vector<uint64_t>> hole_options_vector(family.numHoles());

    // timers["areChoicesConsistent::1 is scheduler consistent?"].start();
    solver.push();
    getRoot()->addFamilyEncoding(subfamily,solver);
    solver.push();
    for(uint64_t choice: choices) {
        uint64_t state = choice_to_state[choice];
        if(not state_is_relevant[state]) {
            continue;
        }
        for(uint64_t path: state_path_enabled[state]) {
            const char *label = choice_path_label[choice][path].c_str();
            solver.add(choice_path_expresssion[choice][path], label);
            // Z3_solver_assert_and_track(ctx, solver.operator Z3_solver(), choice_path_expresssion[choice][path], choice_path_label_expr[choice][path]);
        }
    }
    // timers["areChoicesConsistent::1 (check)"].start();
    bool consistent = check();
    // timers["areChoicesConsistent::1 (check)"].stop();
    // timers["areChoicesConsistent::1 is scheduler consistent?"].stop();

    if(consistent) {
        z3::model model = solver.get_model();
        solver.pop();
        solver.pop();
        getRoot()->loadHoleAssignmentFromModel(model,hole_options_vector);
        // timers[__FUNCTION__].stop();
        return std::make_pair(true,hole_options_vector);
    }

    if(not this->enable_harmonization) {
        solver.pop();
        solver.pop();
        // timers[__FUNCTION__].stop();
        return std::make_pair(false,hole_options_vector);
    }

    solver.pop();

    // timers["areChoicesConsistent::2 better unsat core"].start();
    solver.push();
    std::queue<uint64_t> unexplored_states;
    unexplored_states.push(initial_state);
    BitVector state_reached(numStates(),false);
    state_reached.set(initial_state,true);
    consistent = true;
    while(consistent) {
        STORM_LOG_THROW(not unexplored_states.empty(), storm::exceptions::UnexpectedException, "all states explored but UNSAT core not found");
        uint64_t state = unexplored_states.front(); unexplored_states.pop();
        for(uint64_t choice = row_groups[state]; choice < row_groups[state+1]; ++choice) {
            if(not choices[choice]) {
                continue;
            }
            if(state_is_relevant[state]) {
                for(uint64_t path: state_path_enabled[state]) {
                    const char *label = choice_path_label[choice][path].c_str();
                    solver.add(choice_path_expresssion[choice][path], label);
                }
                // timers["areChoicesConsistent::2 (check)"].start();
                consistent = check();
                // timers["areChoicesConsistent::2 (check)"].stop();
            }
            visitChoice(choice,state_reached,unexplored_states);
            break;
        }
    }
    z3::expr_vector unsat_core_expr = solver.unsat_core();
    solver.pop();
    // loadUnsatCore(unsat_core_expr,subfamily);
    loadUnsatCore(unsat_core_expr,subfamily,choices);
    // timers["areChoicesConsistent::2 better unsat core"].stop();

    if(PRINT_UNSAT_CORE)
        std::cout << "-- unsat core start --" << std::endl;
    // timers["areChoicesConsistent::3 unsat core analysis"].start();
    for(auto [choice,path]: this->unsat_core) {
        const char *label = choice_path_label[choice][path].c_str();
        solver.add(choice_path_expresssion_harm[choice][path], label);
    }

    z3::model model(ctx);

    /*z3::expr_vector harmonizing_variable_domain(ctx);
    for(uint64_t hole: harmonizing_hole_hint) {
        harmonizing_variable_domain.push_back(harmonizing_variable == (int)hole);
    }
    solver.add(z3::mk_or(harmonizing_variable_domain), "harmonizing_domain");*/

    /*bool harmonizing_hole_found = false;
    for(uint64_t hole = 0; hole < family.numHoles(); ++hole) {
        solver.push();
        solver.add(harmonizing_variable == (int)hole, "harmonizing_domain");
        if(check()) {
            harmonizing_hole_found = true;
            model = solver.get_model();
            solver.pop();
            break;
        }
        solver.pop();
    }
    STORM_LOG_THROW(harmonizing_hole_found, storm::exceptions::UnexpectedException, "harmonized UNSAT core is not SAT");*/

    solver.add(0 <= harmonizing_variable and harmonizing_variable < (int)(family.numHoles()), "harmonizing_domain");
    // timers["areChoicesConsistent::3 (check)"].start();
    consistent = check();
    // timers["areChoicesConsistent::3 (check)"].stop();
    if(consistent) {
        model = solver.get_model();
        uint64_t harmonizing_hole = model.eval(harmonizing_variable).get_numeral_uint64();
        getRoot()->loadHoleAssignmentFromModel(model,hole_options_vector);
        getRoot()->loadHoleAssignmentFromModelHarmonizing(model,hole_options_vector,harmonizing_hole);
        if(hole_options_vector[harmonizing_hole][0] > hole_options_vector[harmonizing_hole][1]) {
            uint64_t tmp = hole_options_vector[harmonizing_hole][0];
            hole_options_vector[harmonizing_hole][0] = hole_options_vector[harmonizing_hole][1];
            hole_options_vector[harmonizing_hole][1] = tmp;
        }
    } else {
        STORM_LOG_THROW(consistent, storm::exceptions::UnexpectedException, "harmonized UNSAT core is not SAT");
    }
    solver.pop();

    if(PRINT_UNSAT_CORE)
        std::cout << "-- unsat core end --" << std::endl;

    // timers["areChoicesConsistent::3 unsat core analysis"].stop();

    // timers[__FUNCTION__].stop();
    return std::make_pair(false,hole_options_vector);
}

// template<typename ValueType>
// std::map<std::string,storm::utility::Stopwatch> ColoringSmt<ValueType>::timers;

template class ColoringSmt<>;

}