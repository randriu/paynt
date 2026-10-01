#pragma once

#include "src/synthesis/coloring/Family.h"
#include "src/synthesis/coloring/TreeNode.h"

#include <storm/models/sparse/NondeterministicModel.h>
#include <storm/storage/BitVector.h>
// #include <storm/utility/Stopwatch.h>

#include <cstdint>
#include <queue>
#include <vector>
#include <memory>

#include <z3++.h>

namespace synthesis {

using BitVector = storm::storage::BitVector;

template<typename ValueType = double>
class ColoringSmt {
public:

    /**
     * Construct an SMT coloring.
     * @param row groups (nondeterministic choice indices) of an underlying MDP
     * @param choice_to_action for every choice, the corresponding action
     * @param num_actions total number of actions in the MDP
     * @param dont_care_action index of the don't-care action; if the MDP has no such action, a value equal to
     *  \p num_actions can be given
     * @param state_valuations state valuation of the underlying MDP
     * @param variable_name list of variable names
     * @param variable_domain list of possible variable values
     * @param tree_list a decision tree template encoded via a list of (parent id, left child id, right child id) triples
     * @param enable_harmonization if true, the object will be set up to enable CE generation via harmonization
     */
    ColoringSmt(
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
    );

    ~ColoringSmt();

    /**
     * Enable efficient state exploration of reachable states.
     * @note this is required for harmonization
     */
    void enableStateExploration(storm::models::sparse::NondeterministicModel<ValueType> const& model);

    /** For each hole, get a list of name-type pairs.  */
    std::vector<std::tuple<uint64_t,std::string,std::string>> getFamilyInfo();

    /**
     * Get a mask of choices compatible with the family. For irrelevant states, only the first choice will be enabled.
     */
    BitVector selectCompatibleChoices(Family const& subfamily);
    /**
     * Get a mask of sub-choices of \p base_choices compatible with the family. If a relevant state has no enabled
     * actions, it last action will be enabled. We assume here that this last action is the one executing random action.
     */
    BitVector selectCompatibleChoices(Family const& subfamily, BitVector const& base_choices);

    /**
     * Verify whether the given choices have an associated non-conficting hole assignment (within the given family).
     * @return a pair (A,B) where A is true iff the choice selection is consistent; B is either a satisfying hole
     * assignment if A holds, or a list of inconsistent hole assignments to be used for splitting.
     */
    std::pair<bool,std::vector<std::vector<uint64_t>>> areChoicesConsistent(
        BitVector const& choices, Family const& subfamily
    );

    // static std::map<std::string,storm::utility::Stopwatch> timers;
    // std::vector<std::pair<std::string,double>> getProfilingInfo() {
    //     std::vector<std::pair<std::string,double>> profiling;
    //     for(auto const& [method,timer]: timers) {
    //         profiling.emplace_back(method, (double)(timer.getTimeInMilliseconds())/1000);
    //     }
    //     return profiling;
    // }

    /** A list of choice-path indices appeared in the last UNSAT core. */
    std::vector<std::pair<uint64_t,uint64_t>> unsat_core;

protected:

    /** Whether a check for an admissible family member is done before choice selection. */
    const bool CHECK_FAMILY_CONSISTENCE = true;

    /** Row groups of the quotient. */
    const std::vector<uint64_t> row_groups;
    /** For each choice, the state it comes from. */
    std::vector<uint64_t> choice_to_state;
    /** Number of states in the quotient. */
    const uint64_t numStates() const;
    /** Number of choices in the quotient. */
    const uint64_t numChoices() const;

    /** For each choice, its unique action. */
    const std::vector<uint64_t> choice_to_action;
    /** Number of MDP actions. */
    uint64_t num_actions;
    /** Index of the don't care action; equal to \num_actions if no such action exists. */
    uint64_t dont_care_action;
    /** For every state, a list of available actions. */
    std::vector<BitVector> state_available_actions;
    /** Whether the don't-care action is present in the MDP. */
    const bool dontCareActionDefined() const;

    /** Valuation of each state. */
    std::vector<std::vector<uint64_t>> state_valuation;
    /** Only relevant states are taken into account when checking consistency. */
    const BitVector state_is_relevant;

    /** For each variable, its name. */
    const std::vector<std::string> variable_name;
    /** For each variable, a list of its options. */
    const std::vector<std::vector<int64_t>> variable_domain;
    /** Number of (relevant) program variables. */
    const uint64_t numVariables() const;

    /** A list of tree nodes. */
    std::vector<std::shared_ptr<TreeNode>> tree;
    /** Number of tree nodes. */
    uint64_t numNodes() const;
    /** Retreive tree root. */
    std::shared_ptr<TreeNode> getRoot();
    /** Number of tree paths. */
    uint64_t numPaths();

    /** SMT solver context. */
    z3::context ctx;
    /** SMT solver. */
    z3::solver solver;
    /** Unrefined family. */
    Family family;

    /** Whether efficient state exploration has been enabled. */
    bool state_exploration_enabled = false;
    /** The initial state. */
    uint64_t initial_state;
    /** For each state, a list of target states. */
    std::vector<std::vector<uint64_t>> choice_destinations;

    /** Check the current SMT formula. */
    bool check();

    /** For each path, an index of the hole that occurs at its end. */
    std::vector<uint64_t> path_action_hole;
    /** For each choice and path, a label passed to SMT solver. */
    std::vector<std::vector<std::string>> choice_path_label;
    /** For each choice, its color expressed as a conjunction of all path implications. */
    std::vector<std::vector<z3::expr>> choice_path_expresssion;
    // std::vector<std::vector<Z3_ast>> choice_path_expresssion;


    /** Whether harmonization is required. */
    const bool enable_harmonization;
    /** SMT variable refering to harmonizing hole. */
    z3::expr harmonizing_variable;
    /** For each choice, its color expressed as a conjunction of all path implications. */
    std::vector<std::vector<z3::expr>> choice_path_expresssion_harm;
    // std::vector<std::vector<Z3_ast>> choice_path_expresssion_harm;

    /** For each state, whether (in the last subfamily) the path is enabled. */
    std::vector<BitVector> state_path_enabled;

    /** Add unexplored destinations of the given choice to the queue and mark them as reached. */
    void visitChoice(uint64_t choice, BitVector & state_reached, std::queue<uint64_t> & unexplored_states);

    bool PRINT_UNSAT_CORE = false;
    void loadUnsatCore(z3::expr_vector const& unsat_core_expr, Family const& subfamily);
    void loadUnsatCore(z3::expr_vector const& unsat_core_expr, Family const& subfamily, BitVector const& choices);

};

}