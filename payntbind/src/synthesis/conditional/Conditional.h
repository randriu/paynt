#pragma once

#include <storm/models/sparse/Dtmc.h>
#include <storm/models/sparse/Mdp.h>
#include <storm/models/sparse/Model.h>
#include <storm/storage/BitVector.h>
#include <storm/storage/scheduler/Scheduler.h>

#include <cstdint>
#include <map>
#include <memory>
#include <tuple>
#include <vector>

/**
 * Helpers for a conditional reachability property P[F target || F condition]: the probability of reaching target given that condition is reached, undefined
 * where condition is unreachable.
 *
 * Its best scheduler needs to remember which of the two sets a path has entered first, so it is followed in phases: a path is in phase 0 until it enters
 * condition (phase 1, it then still needs target) or target (phase 2, it then still needs condition), whichever comes first. The phase switches on entering
 * a state, so the choice in a state of condition already belongs to phase 1, as in Storm's scheduler, whose memory switches on leaving a state.
 */
namespace synthesis {

    /** Whether some scheduler of the model reaches the states from its initial state (for a Markov chain: whether it does). */
    template<typename ValueType>
    bool someSchedulerReaches(storm::models::sparse::Model<ValueType> const& model, storm::storage::BitVector const& states);

    /** Whether some scheduler of the model avoids the states from its initial state: reaches them with probability 0 (for a Markov chain: whether it does). */
    template<typename ValueType>
    bool someSchedulerAvoids(storm::models::sparse::Model<ValueType> const& model, storm::storage::BitVector const& states);

    /**
     * The choices that a scheduler for P[F target || F condition] takes where they matter, from the initial state on: following the scheduler and its memory
     * along the phases, with the length of a shortest path to each choice.
     *
     * A choice does not matter where its path has entered both sets, or cannot reach what its phase still needs under any scheduler (condition in phases 0
     * and 2, target in phase 1). Neither does one the scheduler leaves undefined. In phase 0, a state that reaches condition (target) almost surely takes
     * the choice of the next phase, where Storm's scheduler has the one that attains the value.
     *
     * @param choice_map maps the choices of the MDP to those of the MDP it was built from
     * @return {mapped choice: the length of a shortest path to it}
     */
    template<typename ValueType>
    std::map<uint64_t, uint64_t> conditionalSchedulerChoices(
        storm::models::sparse::Mdp<ValueType> const& mdp, storm::storage::Scheduler<ValueType> const& scheduler,
        storm::storage::BitVector const& target, storm::storage::BitVector const& condition, std::vector<uint64_t> const& choice_map
    );

    /**
     * The Markov chain as an MDP with one choice per state, checked by Storm's procedure for MDPs: its procedure for conditional properties of Markov chains
     * (Storm 2.0) fails on some.
     */
    template<typename ValueType>
    std::shared_ptr<storm::models::sparse::Mdp<ValueType>> dtmcAsMdp(storm::models::sparse::Dtmc<ValueType> const& dtmc);

    /**
     * The product of an MDP with the phases of P[F target || F condition]: each state paired with the phase of the paths that enter it. Phases 1 and 2 are
     * kept once a path enters the other set too. Only the pairs reachable from the initial state are kept; labels, choice labels and reward models carry over.
     *
     * @return the product, and for each of its choices: the choice of the MDP it copies, and its phase
     */
    template<typename ValueType>
    std::tuple<std::shared_ptr<storm::models::sparse::Mdp<ValueType>>, std::vector<uint64_t>, std::vector<uint64_t>> unfoldConditionalPhases(
        storm::models::sparse::Mdp<ValueType> const& mdp, storm::storage::BitVector const& target, storm::storage::BitVector const& condition
    );

}
