#include "Conditional.h"

#include <storm/adapters/RationalNumberAdapter.h>
#include <storm/exceptions/NotSupportedException.h>
#include <storm/modelchecker/prctl/helper/SparseDtmcPrctlHelper.h>
#include <storm/modelchecker/results/ExplicitQuantitativeCheckResult.h>
#include <storm/models/sparse/ChoiceLabeling.h>
#include <storm/models/sparse/StandardRewardModel.h>
#include <storm/models/sparse/StateLabeling.h>
#include <storm/storage/SparseMatrix.h>
#include <storm/storage/memorystructure/MemoryStructure.h>
#include <storm/storage/sparse/ModelComponents.h>
#include <storm/solver/SolveGoal.h>
#include <storm/utility/ExtendedNumber.h>
#include <storm/utility/graph.h>
#include <storm/utility/macros.h>

#include <algorithm>
#include <deque>
#include <limits>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>

namespace synthesis {

    namespace {

        // the phase of a path that has entered both sets: its choices no longer matter
        uint64_t const DECIDED = 3;

        uint64_t phaseAfterEntering(
            uint64_t phase, uint64_t state, storm::storage::BitVector const& target, storm::storage::BitVector const& condition
        ) {
            bool inTarget = target.get(state);
            bool inCondition = condition.get(state);
            switch (phase) {
                case 0:
                    if (inCondition) {
                        return inTarget ? DECIDED : 1;
                    }
                    return inTarget ? 2 : 0;
                case 1:
                    return inTarget ? DECIDED : 1;
                case 2:
                    return inCondition ? DECIDED : 2;
                default:
                    return DECIDED;
            }
        }

        // the phase of a state of the product: phases 1 and 2 are kept once a path enters the other set too
        uint64_t productPhaseAfterEntering(
            uint64_t phase, uint64_t state, storm::storage::BitVector const& target, storm::storage::BitVector const& condition
        ) {
            if (phase != 0) {
                return phase;
            }
            if (condition.get(state)) {
                return 1;
            }
            return target.get(state) ? 2 : 0;
        }

    }

    template<typename ValueType>
    bool someSchedulerReaches(storm::models::sparse::Model<ValueType> const& model, storm::storage::BitVector const& states) {
        storm::storage::BitVector all(model.getNumberOfStates(), true);
        storm::storage::BitVector reachable = storm::utility::graph::getReachableStates(model.getTransitionMatrix(), model.getInitialStates(), all, states);
        return !(reachable & states).empty();
    }

    template<typename ValueType>
    bool someSchedulerAvoids(storm::models::sparse::Model<ValueType> const& model, storm::storage::BitVector const& states) {
        auto const& matrix = model.getTransitionMatrix();
        storm::storage::SparseMatrix<ValueType> backward = model.getBackwardTransitions();
        storm::storage::BitVector all(model.getNumberOfStates(), true);
        storm::storage::BitVector avoiding;
        if (model.isNondeterministicModel()) {
            avoiding = storm::utility::graph::performProb0E(matrix, matrix.getRowGroupIndices(), backward, all, states);
        } else {
            avoiding = ~storm::utility::graph::performProbGreater0(backward, all, states);
        }
        return !(avoiding & model.getInitialStates()).empty();
    }

    template<typename ValueType>
    std::map<uint64_t, uint64_t> conditionalSchedulerChoices(
        storm::models::sparse::Mdp<ValueType> const& mdp, storm::storage::Scheduler<ValueType> const& scheduler,
        storm::storage::BitVector const& target, storm::storage::BitVector const& condition, std::vector<uint64_t> const& choice_map
    ) {
        STORM_LOG_THROW(scheduler.isDeterministicScheduler(), storm::exceptions::NotSupportedException, "expected a deterministic scheduler");
        auto const& matrix = mdp.getTransitionMatrix();
        auto const& row_groups = mdp.getNondeterministicChoiceIndices();
        storm::storage::SparseMatrix<ValueType> backward = mdp.getBackwardTransitions();
        storm::storage::BitVector all(mdp.getNumberOfStates(), true);
        storm::storage::BitVector reaching_target = storm::utility::graph::performProbGreater0E(backward, all, target);
        storm::storage::BitVector reaching_condition = storm::utility::graph::performProbGreater0E(backward, all, condition);

        auto const& memory = scheduler.getMemoryStructure();
        uint64_t num_memory_states = scheduler.getNumberOfMemoryStates();
        // Storm (2.0) gives a state that reaches condition (target) almost surely, but is not in it, a choice that merely reaches it: its value is that of the
        // choice it takes once condition (target) is reached, which is the one to follow (a choice is then the same in both phases)
        std::optional<uint64_t> memory_after_condition;
        std::optional<uint64_t> memory_after_target;
        if (memory) {
            auto const& memory_labeling = memory->getStateLabeling();
            if (memory_labeling.containsLabel("condition_reached") && memory_labeling.containsLabel("target_reached")) {
                memory_after_condition = *memory_labeling.getStates("condition_reached").begin();
                memory_after_target = *memory_labeling.getStates("target_reached").begin();
            }
        }
        storm::storage::BitVector reaching_condition_surely = storm::utility::graph::performProb1A(matrix, row_groups, backward, all, condition);
        storm::storage::BitVector reaching_target_surely = storm::utility::graph::performProb1A(matrix, row_groups, backward, all, target);
        // a (state, memory state, phase) triple is visited once: a memoryless scheduler can reach a state in several phases
        auto index = [&](uint64_t state, uint64_t memory_state, uint64_t phase) { return (state * num_memory_states + memory_state) * (DECIDED + 1) + phase; };
        storm::storage::BitVector visited(mdp.getNumberOfStates() * num_memory_states * (DECIDED + 1), false);

        struct Visit {
            uint64_t state;
            uint64_t memory_state;
            uint64_t phase;
            uint64_t depth;
        };
        std::deque<Visit> queue;
        uint64_t initial_index = 0;
        for (auto state : mdp.getInitialStates()) {
            uint64_t memory_state = memory ? memory->getInitialMemoryStates()[initial_index] : 0;
            uint64_t phase = phaseAfterEntering(0, state, target, condition);
            visited.set(index(state, memory_state, phase));
            queue.push_back({state, memory_state, phase, 0});
            ++initial_index;
        }

        std::map<uint64_t, uint64_t> choice_depth;
        while (!queue.empty()) {
            Visit visit = queue.front();
            queue.pop_front();
            if (visit.phase == DECIDED) {
                continue;
            }
            storm::storage::BitVector const& reaching_needed = visit.phase == 1 ? reaching_target : reaching_condition;
            if (!reaching_needed.get(visit.state)) {
                continue;
            }
            uint64_t choice_memory_state = visit.memory_state;
            if (visit.phase == 0 && memory_after_condition && reaching_condition_surely.get(visit.state)) {
                choice_memory_state = *memory_after_condition;
            } else if (visit.phase == 0 && memory_after_target && reaching_target_surely.get(visit.state)) {
                choice_memory_state = *memory_after_target;
            }
            auto const& scheduler_choice = scheduler.getChoice(visit.state, choice_memory_state);
            if (!scheduler_choice.isDefined()) {
                continue;
            }
            uint64_t choice = row_groups[visit.state] + scheduler_choice.getDeterministicChoice();
            // breadth-first: the first visit is along a shortest path
            choice_depth.emplace(choice_map[choice], visit.depth);
            for (auto entry = matrix.begin(choice); entry != matrix.end(choice); ++entry) {
                uint64_t successor = entry->getColumn();
                uint64_t successor_memory_state = memory ? memory->getSuccessorMemoryState(visit.memory_state, entry - matrix.begin()) : 0;
                uint64_t successor_phase = phaseAfterEntering(visit.phase, successor, target, condition);
                uint64_t successor_index = index(successor, successor_memory_state, successor_phase);
                if (!visited.get(successor_index)) {
                    visited.set(successor_index);
                    queue.push_back({successor, successor_memory_state, successor_phase, visit.depth + 1});
                }
            }
        }
        return choice_depth;
    }

    template<typename ValueType>
    std::shared_ptr<storm::modelchecker::CheckResult> conditionalProbabilities(
        storm::Environment const& env, storm::models::sparse::Dtmc<ValueType> const& dtmc, storm::storage::BitVector const& target,
        storm::storage::BitVector const& condition
    ) {
        using Helper = storm::modelchecker::helper::SparseDtmcPrctlHelper<ValueType>;
        auto const& matrix = dtmc.getTransitionMatrix();
        storm::storage::SparseMatrix<ValueType> backward = dtmc.getBackwardTransitions();
        uint64_t num_states = dtmc.getNumberOfStates();
        storm::storage::BitVector all(num_states, true);
        std::vector<ValueType> reaching_condition =
            Helper::computeUntilProbabilities(env, storm::solver::SolveGoal<ValueType>(), matrix, backward, all, condition, false).values;
        std::vector<ValueType> reaching_target =
            Helper::computeUntilProbabilities(env, storm::solver::SolveGoal<ValueType>(), matrix, backward, all, target, false).values;

        // P(F target & F condition): a path enters one of the two sets first and then has to reach the other one, so the chain stops there and moves to
        // 'both' with the probability of reaching the other set, to 'neither' otherwise (a sum of probabilities, not a difference that could cancel out)
        uint64_t const both = num_states;
        uint64_t const neither = num_states + 1;
        ValueType const one = storm::utility::one<ValueType>();
        storm::storage::SparseMatrixBuilder<ValueType> builder(num_states + 2, num_states + 2, 0, true, false);
        for (uint64_t state = 0; state < num_states; ++state) {
            if (condition.get(state) || target.get(state)) {
                // iterative solvers may overshoot
                ValueType reaching_other = std::min(condition.get(state) ? reaching_target[state] : reaching_condition[state], one);
                if (!storm::utility::isZero(reaching_other)) {
                    builder.addNextValue(state, both, reaching_other);
                }
                if (!storm::utility::isOne(reaching_other)) {
                    builder.addNextValue(state, neither, one - reaching_other);
                }
            } else {
                for (auto const& entry : matrix.getRow(state)) {
                    builder.addNextValue(state, entry.getColumn(), entry.getValue());
                }
            }
        }
        builder.addNextValue(both, both, one);
        builder.addNextValue(neither, neither, one);
        storm::storage::SparseMatrix<ValueType> stopped = builder.build();
        storm::storage::BitVector both_state(num_states + 2, false);
        both_state.set(both);
        std::vector<ValueType> reaching_both = Helper::computeUntilProbabilities(
            env, storm::solver::SolveGoal<ValueType>(), stopped, stopped.transpose(), storm::storage::BitVector(num_states + 2, true), both_state, false
        ).values;

        std::vector<storm::utility::ExtendedValueType<ValueType>> values(num_states);
        for (uint64_t state = 0; state < num_states; ++state) {
            if (storm::utility::isZero(reaching_condition[state])) {
                values[state] = storm::utility::positiveInfinity<ValueType>();
            } else {
                values[state] = storm::utility::ExtendedValueType<ValueType>(reaching_both[state] / reaching_condition[state]);
            }
        }
        return std::make_shared<storm::modelchecker::ExplicitQuantitativeCheckResult<ValueType>>(std::move(values));
    }

    template<typename ValueType>
    std::tuple<std::shared_ptr<storm::models::sparse::Mdp<ValueType>>, std::vector<uint64_t>, std::vector<uint64_t>> unfoldConditionalPhases(
        storm::models::sparse::Mdp<ValueType> const& mdp, storm::storage::BitVector const& target, storm::storage::BitVector const& condition
    ) {
        auto const& matrix = mdp.getTransitionMatrix();
        auto const& row_groups = mdp.getNondeterministicChoiceIndices();
        uint64_t num_states = mdp.getNumberOfStates();
        STORM_LOG_THROW(mdp.getInitialStates().getNumberOfSetBits() == 1, storm::exceptions::NotSupportedException, "expected a single initial state");
        uint64_t initial_state = *mdp.getInitialStates().begin();

        // the (state, phase) pairs reachable from the initial state, breadth-first: the initial one is the first
        uint64_t const none = std::numeric_limits<uint64_t>::max();
        std::vector<uint64_t> pair_to_product_state(num_states * 3, none);
        std::vector<uint64_t> state_prototype;
        std::vector<uint64_t> state_phase;
        auto product_state = [&](uint64_t state, uint64_t phase) {
            uint64_t& product = pair_to_product_state[state * 3 + phase];
            if (product == none) {
                product = state_prototype.size();
                state_prototype.push_back(state);
                state_phase.push_back(phase);
            }
            return product;
        };
        product_state(initial_state, productPhaseAfterEntering(0, initial_state, target, condition));
        uint64_t num_product_choices = 0;
        for (uint64_t product = 0; product < state_prototype.size(); ++product) {
            uint64_t state = state_prototype[product];
            uint64_t phase = state_phase[product];
            num_product_choices += row_groups[state + 1] - row_groups[state];
            for (uint64_t choice = row_groups[state]; choice < row_groups[state + 1]; ++choice) {
                for (auto const& entry : matrix.getRow(choice)) {
                    product_state(entry.getColumn(), productPhaseAfterEntering(phase, entry.getColumn(), target, condition));
                }
            }
        }
        uint64_t num_product_states = state_prototype.size();

        storm::storage::SparseMatrixBuilder<ValueType> builder(num_product_choices, num_product_states, 0, true, true, num_product_states);
        std::vector<uint64_t> choice_map;
        std::vector<uint64_t> choice_phase;
        choice_map.reserve(num_product_choices);
        choice_phase.reserve(num_product_choices);
        std::vector<std::pair<uint64_t, ValueType>> entries;
        uint64_t row = 0;
        for (uint64_t product = 0; product < num_product_states; ++product) {
            uint64_t state = state_prototype[product];
            uint64_t phase = state_phase[product];
            builder.newRowGroup(row);
            for (uint64_t choice = row_groups[state]; choice < row_groups[state + 1]; ++choice) {
                entries.clear();
                for (auto const& entry : matrix.getRow(choice)) {
                    uint64_t successor_phase = productPhaseAfterEntering(phase, entry.getColumn(), target, condition);
                    entries.emplace_back(pair_to_product_state[entry.getColumn() * 3 + successor_phase], entry.getValue());
                }
                std::sort(entries.begin(), entries.end(), [](auto const& a, auto const& b) { return a.first < b.first; });
                for (auto const& [column, value] : entries) {
                    builder.addNextValue(row, column, value);
                }
                choice_map.push_back(choice);
                choice_phase.push_back(phase);
                ++row;
            }
        }

        storm::models::sparse::StateLabeling labeling(num_product_states);
        for (auto const& label : mdp.getStateLabeling().getLabels()) {
            storm::storage::BitVector const& states = mdp.getStateLabeling().getStates(label);
            storm::storage::BitVector product_states(num_product_states, false);
            if (label == "init") {
                product_states.set(0);
            } else {
                for (uint64_t product = 0; product < num_product_states; ++product) {
                    product_states.set(product, states.get(state_prototype[product]));
                }
            }
            labeling.addLabel(label, std::move(product_states));
        }

        std::unordered_map<std::string, storm::models::sparse::StandardRewardModel<ValueType>> reward_models;
        for (auto const& [name, reward_model] : mdp.getRewardModels()) {
            STORM_LOG_THROW(!reward_model.hasTransitionRewards(), storm::exceptions::NotSupportedException, "transition rewards are not supported");
            std::optional<std::vector<ValueType>> state_rewards;
            std::optional<std::vector<ValueType>> state_action_rewards;
            if (reward_model.hasStateRewards()) {
                state_rewards = std::vector<ValueType>(num_product_states);
                for (uint64_t product = 0; product < num_product_states; ++product) {
                    (*state_rewards)[product] = reward_model.getStateReward(state_prototype[product]);
                }
            }
            if (reward_model.hasStateActionRewards()) {
                state_action_rewards = std::vector<ValueType>(num_product_choices);
                for (uint64_t choice = 0; choice < num_product_choices; ++choice) {
                    (*state_action_rewards)[choice] = reward_model.getStateActionReward(choice_map[choice]);
                }
            }
            reward_models.emplace(name, storm::models::sparse::StandardRewardModel<ValueType>(std::move(state_rewards), std::move(state_action_rewards)));
        }

        storm::storage::sparse::ModelComponents<ValueType> components(builder.build(), std::move(labeling), std::move(reward_models));
        if (mdp.hasChoiceLabeling()) {
            storm::models::sparse::ChoiceLabeling choice_labeling(num_product_choices);
            for (auto const& label : mdp.getChoiceLabeling().getLabels()) {
                storm::storage::BitVector const& choices = mdp.getChoiceLabeling().getChoices(label);
                storm::storage::BitVector product_choices(num_product_choices, false);
                for (uint64_t choice = 0; choice < num_product_choices; ++choice) {
                    product_choices.set(choice, choices.get(choice_map[choice]));
                }
                choice_labeling.addLabel(label, std::move(product_choices));
            }
            components.choiceLabeling = std::move(choice_labeling);
        }
        auto product = std::make_shared<storm::models::sparse::Mdp<ValueType>>(std::move(components));
        return {product, choice_map, choice_phase};
    }

    template bool someSchedulerReaches<double>(storm::models::sparse::Model<double> const&, storm::storage::BitVector const&);
    template bool someSchedulerReaches<storm::RationalNumber>(storm::models::sparse::Model<storm::RationalNumber> const&, storm::storage::BitVector const&);
    template bool someSchedulerAvoids<double>(storm::models::sparse::Model<double> const&, storm::storage::BitVector const&);
    template bool someSchedulerAvoids<storm::RationalNumber>(storm::models::sparse::Model<storm::RationalNumber> const&, storm::storage::BitVector const&);
    template std::map<uint64_t, uint64_t> conditionalSchedulerChoices<double>(
        storm::models::sparse::Mdp<double> const&, storm::storage::Scheduler<double> const&, storm::storage::BitVector const&,
        storm::storage::BitVector const&, std::vector<uint64_t> const&
    );
    template std::map<uint64_t, uint64_t> conditionalSchedulerChoices<storm::RationalNumber>(
        storm::models::sparse::Mdp<storm::RationalNumber> const&, storm::storage::Scheduler<storm::RationalNumber> const&, storm::storage::BitVector const&,
        storm::storage::BitVector const&, std::vector<uint64_t> const&
    );
    template std::shared_ptr<storm::modelchecker::CheckResult> conditionalProbabilities<double>(
        storm::Environment const&, storm::models::sparse::Dtmc<double> const&, storm::storage::BitVector const&, storm::storage::BitVector const&
    );
    template std::shared_ptr<storm::modelchecker::CheckResult> conditionalProbabilities<storm::RationalNumber>(
        storm::Environment const&, storm::models::sparse::Dtmc<storm::RationalNumber> const&, storm::storage::BitVector const&,
        storm::storage::BitVector const&
    );
    template std::tuple<std::shared_ptr<storm::models::sparse::Mdp<double>>, std::vector<uint64_t>, std::vector<uint64_t>> unfoldConditionalPhases<double>(
        storm::models::sparse::Mdp<double> const&, storm::storage::BitVector const&, storm::storage::BitVector const&
    );
    template std::tuple<std::shared_ptr<storm::models::sparse::Mdp<storm::RationalNumber>>, std::vector<uint64_t>, std::vector<uint64_t>>
    unfoldConditionalPhases<storm::RationalNumber>(
        storm::models::sparse::Mdp<storm::RationalNumber> const&, storm::storage::BitVector const&, storm::storage::BitVector const&
    );

}
