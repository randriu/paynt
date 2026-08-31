#include "src/synthesis/synthesis.h"

#include "JaniChoices.h"
#include "Family.h"
#include "Coloring.h"
#include "ColoringSmt.h"
#include "src/synthesis/translation/componentTranslations.h"

#include <storm/storage/expressions/ExpressionManager.h>
#include <storm/storage/expressions/Variable.h>
#include <storm/storage/valuations/Valuations.h>
#include <storm/storage/valuations/ValuationDescriptionBuilder.h>
#include <storm/storage/valuations/ValuationsStorage.h>
#include <storm/utility/builder.h>

#include <storm/exceptions/InvalidModelException.h>
#include <storm/models/sparse/Mdp.h>
#include <storm/models/sparse/NondeterministicModel.h>
#include <storm/storage/BitVector.h>
#include <storm/storage/sparse/JaniChoiceOrigins.h>

#include <storm/storage/Scheduler.h>

#include <storm/adapters/RationalNumberAdapter.h>

#include <algorithm>
#include <map>

#include <z3++.h>

namespace synthesis {

template<typename ValueType>
std::shared_ptr<storm::models::sparse::Model<ValueType>> addStateValuations(
    storm::models::sparse::Model<ValueType> const& model,
    std::vector<std::vector<std::pair<std::string,int64_t>>> const& state_valuations
) {
    STORM_LOG_THROW(
        model.getNumberOfStates() == state_valuations.size(), storm::exceptions::InvalidModelException,
        "Number of state valuations does not match the state count."
    );
    std::shared_ptr<storm::expressions::ExpressionManager> em(new storm::expressions::ExpressionManager());
    std::map<std::string,storm::expressions::Variable> name_to_variable;
    std::map<std::string,int64_t> variable_lower_bound;
    std::map<std::string,int64_t> variable_upper_bound;
    for(auto const& state_valuation: state_valuations) {
        for(auto const& [variable_name,value]: state_valuation) {
            auto it = name_to_variable.find(variable_name);
            if(it == name_to_variable.end()) {
                name_to_variable.emplace(variable_name,em->declareIntegerVariable(variable_name));
                variable_lower_bound[variable_name] = value;
                variable_upper_bound[variable_name] = value;
            } else {
                variable_lower_bound[variable_name] = std::min(variable_lower_bound[variable_name],value);
                variable_upper_bound[variable_name] = std::max(variable_upper_bound[variable_name],value);
            }
        }
    }
    storm::storage::sparse::ValuationDescriptionBuilder desc_builder(em);
    for(auto const& [variable_name,variable]: name_to_variable) {
        desc_builder.addIntegerVariable(variable, variable_lower_bound[variable_name], variable_upper_bound[variable_name]);
    }
    storm::storage::sparse::Valuations valuations(desc_builder.buildClassDescription(), em, model.getNumberOfStates());
    for(uint64_t state = 0; state < model.getNumberOfStates(); ++state) {
        for(auto const& [variable_name,value]: state_valuations[state]) {
            valuations.getStorage().writeValue<int64_t>(state, name_to_variable.at(variable_name), value);
        }
    }
    storm::storage::sparse::ModelComponents<ValueType> components = synthesis::componentsFromModel(model);
    components.stateValuations = std::move(valuations);
    return storm::utility::builder::buildModelFromComponents<ValueType>(model.getType(),std::move(components));
}

template<typename ValueType>
std::pair<storm::storage::BitVector,std::vector<std::vector<std::pair<uint64_t,uint64_t>>>> janiMapChoicesToHoleAssignments(
    storm::models::sparse::Mdp<ValueType> const& mdp,
    Family const& family,
    std::map<uint64_t,std::vector<std::pair<uint64_t,uint64_t>>> edge_to_hole_assignment
) {

    uint64_t num_choices = mdp.getNumberOfChoices();
    storm::storage::BitVector choice_is_valid(num_choices,true);
    std::vector<std::vector<std::pair<uint64_t,uint64_t>>> choice_to_hole_assignment(num_choices);
    for(uint64_t choice = 0; choice < num_choices; ++choice) {
        std::vector<bool> hole_set(family.numHoles(),false);
        std::vector<uint64_t> hole_option(family.numHoles());
        bool valid_choice = true;
        for(auto const& edge: mdp.getChoiceOrigins()->asJaniChoiceOrigins().getEdgeIndexSet(choice)) {
            auto hole_assignment = edge_to_hole_assignment.find(edge);
            if(hole_assignment == edge_to_hole_assignment.end()) {
                continue;
            }
            for(auto const& [hole,option]: hole_assignment->second) {
                if(not hole_set[hole]) {
                    hole_option[hole] = option;
                    hole_set[hole] = true;
                } else if(hole_option[hole] != option) {
                    valid_choice = false;
                    break;
                }
            }
            if(not valid_choice) {
                break;
            }
        }
        if(not valid_choice) {
            choice_is_valid.set(choice,false);
            continue;
        }
        for(uint64_t hole = 0; hole < family.numHoles(); ++hole) {
            if(not hole_set[hole]) {
                continue;
            }
            choice_to_hole_assignment[choice].push_back(std::make_pair(hole,hole_option[hole]));
        }
    }
    return std::make_pair(choice_is_valid,choice_to_hole_assignment);
}

template<typename ValueType>
std::vector<uint64_t> schedulerToStateToGlobalChoice(
    storm::storage::Scheduler<ValueType> const& scheduler, storm::models::sparse::Mdp<ValueType> const& sub_mdp,
    std::vector<uint64_t> choice_to_global_choice
) {
    uint64_t num_states = sub_mdp.getNumberOfStates();
    std::vector<uint64_t> state_to_choice(num_states);
    auto const& nci = sub_mdp.getNondeterministicChoiceIndices();
    for(uint64_t state=0; state<num_states; ++state) {
        uint64_t choice = nci[state] + scheduler.getChoice(state).getDeterministicChoice();
        uint64_t choice_global = choice_to_global_choice[choice];
        state_to_choice[state] = choice_global;
    }
    return state_to_choice;
}

std::map<uint64_t,double> computeInconsistentHoleVariance(
    Family const& family,
    std::vector<uint64_t> const& row_groups, std::vector<uint64_t> const& choice_to_global_choice,
    std::vector<double> const& choice_to_value,
    Coloring const& coloring, std::map<uint64_t,std::vector<uint64_t>> const& hole_to_inconsistent_options,
    std::vector<double> const& state_to_expected_visits
) {

    auto num_holes = family.numHoles();
    std::vector<BitVector> hole_to_inconsistent_options_mask(num_holes);

    for(uint64_t hole=0; hole<num_holes; ++hole) {
        hole_to_inconsistent_options_mask[hole] = BitVector(family.holeNumOptionsTotal(hole));
    }
    BitVector inconsistent_holes(num_holes);
    for(auto const& [hole,options]: hole_to_inconsistent_options) {
        inconsistent_holes.set(hole);
        for(auto option: options) {
            hole_to_inconsistent_options_mask[hole].set(option);
        }
    }

    std::vector<double> hole_difference_avg(num_holes,0);
    std::vector<uint64_t> hole_states_affected(num_holes,0);
    auto const& choice_to_assignment = coloring.getChoiceToAssignment();

    std::vector<bool> hole_set(num_holes);
    std::vector<double> hole_min(num_holes);
    std::vector<double> hole_max(num_holes);
        
    auto num_states = row_groups.size()-1;
    for(uint64_t state=0; state<num_states; ++state) {

        for(uint64_t choice=row_groups[state]; choice<row_groups[state+1]; ++choice) {
            auto value = choice_to_value[choice];
            auto choice_global = choice_to_global_choice[choice];

            for(auto const& [hole,option]: choice_to_assignment[choice_global]) {
                if(not  hole_to_inconsistent_options_mask[hole][option]) {
                    continue;
                }

                if(not hole_set[hole]) {
                    hole_min[hole] = value;
                    hole_max[hole] = value;
                    hole_set[hole] = true;
                } else {
                    if(value < hole_min[hole]) {
                        hole_min[hole] = value;
                    }
                    if(value > hole_max[hole]) {
                        hole_max[hole] = value;
                    }
                }
            }
        }

        for(auto hole: inconsistent_holes) {
            if(not hole_set[hole]) {
                continue;
            }
            double difference = (hole_max[hole]-hole_min[hole])*state_to_expected_visits[state];
            hole_states_affected[hole] += 1;
            hole_difference_avg[hole] += (difference-hole_difference_avg[hole]) / hole_states_affected[hole];
        }
        std::fill(hole_set.begin(), hole_set.end(), false);
    }

    std::map<uint64_t,double> inconsistent_hole_variance;
    for(auto hole: inconsistent_holes) {
        inconsistent_hole_variance[hole] = hole_difference_avg[hole];
    }

    return inconsistent_hole_variance;
}


/*storm::storage::BitVector keepReachableChoices(
    storm::storage::BitVector enabled_choices, uint64_t initial_state,
    std::vector<uint64_t> const& row_groups, std::vector<std::vector<uint64_t>> const& choice_destinations
) {

    uint64_t num_states = row_groups.size()-1;
    uint64_t num_choices = enabled_choices.size();

    storm::storage::BitVector reachable_choices(num_choices,false);
    storm::storage::BitVector state_visited(num_states,false);

    std::queue<uint64_t> state_queue;
    state_visited.set(initial_state,true);
    state_queue.push(initial_state);
    while(not state_queue.empty()) {
        auto state = state_queue.front();
        state_queue.pop();
        for(uint64_t choice = row_groups[state]; choice < row_groups[state+1]; ++choice) {
            if(not enabled_choices[choice]) {
                continue;
            }
            reachable_choices.set(choice,true);
            for(auto dst: choice_destinations[choice]) {
                if(not state_visited[dst]) {
                    state_visited.set(dst,true);
                    state_queue.push(dst);
                }
            }
        }
    }
    return reachable_choices;
}*/

// RA: I don't even understand why this needs to be optimized, but it does
storm::storage::BitVector policyToChoicesForFamily(
    std::vector<uint64_t> const& policy_choices,
    storm::storage::BitVector const& family_choices
) {
    storm::storage::BitVector choices(family_choices.size(),false);
    for(auto choice : policy_choices) {
        choices.set(choice,true);
    }
    return choices & family_choices;
}


/*std::pair<std::vector<uint64_t>,storm::storage::BitVector> fixPolicyForFamily(
    std::vector<uint64_t> const& policy, uint64_t invalid_action,
    storm::storage::BitVector const& family_choices,
    uint64_t initial_state, uint64_t num_choices,
    std::vector<std::vector<uint64_t>> const& state_to_actions,
    std::vector<std::vector<std::vector<uint64_t>>> const& state_action_choices,
    std::vector<std::vector<uint64_t>> const& choice_destinations
) {

    uint64_t num_states = state_to_actions.size();

    std::vector<uint64_t> policy_fixed(num_states,invalid_action);
    storm::storage::BitVector choice_mask(num_choices,false);

    storm::storage::BitVector state_visited(num_states,false);
    state_visited.set(initial_state,true);

    std::queue<uint64_t> state_queue;
    state_queue.push(initial_state);
    while(not state_queue.empty()) {
        auto state = state_queue.front();
        state_queue.pop();
        // get action executed in the state
        auto action = policy[state];
        if(action == invalid_action) {
            action = state_to_actions[state][0];
        }
        policy_fixed[state] = action;
        // expand through the choices that correspond to this action
        for(auto choice: state_action_choices[state][action]) {
            if(not family_choices[choice]) {
                continue;
            }
            choice_mask.set(choice,true);
            for(auto dst: choice_destinations[choice]) {
                if(not state_visited[dst]) {
                    state_visited.set(dst,true);
                    state_queue.push(dst);
                }
            }   
        }
    }
    return std::make_pair(policy_fixed,choice_mask);
}*/

}


template <typename ValueType>
void bindings_coloring_vt(py::module& m, std::string const& vtSuffix) {

    m.def(("addStateValuations" + vtSuffix).c_str(), &synthesis::addStateValuations<ValueType>);
    m.def(("janiMapChoicesToHoleAssignments" + vtSuffix).c_str(), &synthesis::janiMapChoicesToHoleAssignments<ValueType>);
    m.def(("addChoiceLabelsFromJani" + vtSuffix).c_str(), &synthesis::addChoiceLabelsFromJani<ValueType>);

    m.def(("schedulerToStateToGlobalChoice" + vtSuffix).c_str(), &synthesis::schedulerToStateToGlobalChoice<ValueType>);
}

void bindings_coloring(py::module& m) {
    bindings_coloring_vt<double>(m, "");
    bindings_coloring_vt<storm::RationalNumber>(m, "Exact");

    m.def("computeInconsistentHoleVariance", &synthesis::computeInconsistentHoleVariance);

    m.def("policyToChoicesForFamily", &synthesis::policyToChoicesForFamily);

    py::classh<synthesis::Family>(m, "Family")
        .def(py::init<>())
        .def(py::init<synthesis::Family const&>())
        .def("numHoles", &synthesis::Family::numHoles)
        .def("addHole", &synthesis::Family::addHole)
        
        .def("holeOptions", &synthesis::Family::holeOptions)
        .def("holeOptionsMask", &synthesis::Family::holeOptionsMask)
        .def("holeSetOptions", py::overload_cast<uint64_t, std::vector<uint64_t> const&>(&synthesis::Family::holeSetOptions))
        .def("holeSetOptions", py::overload_cast<uint64_t, storm::storage::BitVector const&>(&synthesis::Family::holeSetOptions))
        .def("holeNumOptions", &synthesis::Family::holeNumOptions)
        .def("holeNumOptionsTotal", &synthesis::Family::holeNumOptionsTotal)
        .def("holeContains", &synthesis::Family::holeContains)
        ;

    py::classh<synthesis::Coloring>(m, "Coloring")
        .def(py::init<
            synthesis::Family const&,
            std::vector<uint64_t> const&,
            std::vector<std::vector<std::pair<uint64_t,uint64_t>>>
        >())
        .def("getChoiceToAssignment", &synthesis::Coloring::getChoiceToAssignment)
        .def("getStateToHoles", &synthesis::Coloring::getStateToHoles)
        .def("selectCompatibleChoices", &synthesis::Coloring::selectCompatibleChoices)
        .def("collectHoleOptions", &synthesis::Coloring::collectHoleOptions)
        ;

    py::classh<synthesis::ColoringSmt<>>(m, "ColoringSmt")
        .def(py::init<
            std::vector<uint64_t> const&,
            std::vector<uint64_t> const&,
            uint64_t, uint64_t,
            storm::storage::sparse::Valuations const&,
            storm::storage::BitVector const&,
            std::vector<std::string> const&,
            std::vector<std::vector<int64_t>> const&,
            std::vector<std::tuple<uint64_t,uint64_t,uint64_t>> const&,
            bool
        >())
        .def("enableStateExploration", &synthesis::ColoringSmt<>::enableStateExploration)
        .def("getFamilyInfo", &synthesis::ColoringSmt<>::getFamilyInfo)
        .def("selectCompatibleChoices", py::overload_cast<synthesis::Family const&>(&synthesis::ColoringSmt<>::selectCompatibleChoices))
        .def("selectCompatibleChoices", py::overload_cast<synthesis::Family const&, storm::storage::BitVector const&>(&synthesis::ColoringSmt<>::selectCompatibleChoices))
        .def("areChoicesConsistent", &synthesis::ColoringSmt<>::areChoicesConsistent)
        // .def_property_readonly("unsat_core", [](synthesis::ColoringSmt<>& coloring) {return coloring.unsat_core;})
        // .def("getProfilingInfo", &synthesis::ColoringSmt<>::getProfilingInfo)
        ;
}
