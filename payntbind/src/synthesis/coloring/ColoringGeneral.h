#pragma once

#include "src/synthesis/coloring/Coloring.h"
#include "src/synthesis/coloring/Family.h"

#include <storm/storage/BitVector.h>

#include <cstdint>
#include <memory>
#include <queue>
#include <tuple>
#include <utility>
#include <vector>

#include <z3++.h>

namespace synthesis {

using BitVector = storm::storage::BitVector;

/**
 * A general coloring: each choice of the quotient MDP is colored by a quantifier-free formula over the parameters,
 * and an assignment to parameters enables a choice iff it satisfies the formula.
 *
 * Representation. The colors are a node table built in Python by paynt/utils/coloring_builder.py; the constructor lists
 * the operands of each NodeOp. Colors are written in z3py, which links a potentially different Z3 than this library, 
 * so they arrive as plain data. Data columns (a row per state and per choice) let one formula color many choices:
 * a choice stores only the root of its color (choice_root) and its data row, so choices whose colors differ only in
 * data share all their nodes.
 * Invariants: children have smaller ids than their parents; parameter values are option indices; 
 * choice_root[c] == -1 means that c is uncolored, i.e. always enabled.
 *
 * Three readers implement the meaning of the table and must agree with the z3py formula: the Kleene evaluation
 * (selectCompatibleChoices, relevantParameters), the Z3 translation (areChoicesConsistent, exact mode) and
 * paynt.utils.coloring.choice_formula. A new NodeOp needs a case in the constructor's scope/support pass,
 * computeFormula or computeTerm, buildFormula or buildTerm, unionReasonTrue/False, the builder and choice_formula.
 *
 * Kleene evaluation. Under a given assignment to the parameters, a formula evaluates to T (every member satisfies it), 
 * F (no member does) or U (unknown), and a term to an interval that contains its value in every member. 
 * A choice is selected unless its formula is evaluated to F.
 * Every step over-approximates, so a choice that some member enables is never dropped, and on a full assignment the
 * result is exact. On a partial assignment it is exact if every comparison is between a bare parameter and data or constants
 * (InSetF and InBitsF are of this kind) and, in addition, every parameter occurs at most once in the color, or the color
 * has no negation and every And joins parts over different parameters (decision trees, lifted standard colorings).
 * Otherwise it may keep a choice that no member enables, e.g. `And(p >= 1, p <= 2)` or `p + 1 == 3 for p in {0, 3};` that
 * only enlarges the sub-MDP, meaning it is still a safe abstraction. 
 * Deciding this exactly is NP-complete in general! Exact mode does it with Z3, much more slowly.
 *
 * A data column node reads its column in the row of the state or choice being evaluated, so a node is evaluated once
 * per query, per state or per choice, depending on whether it reads (itself or through its children) no data, state
 * data or choice data; epoch counters invalidate these caches. The caches and the query context are members: one query
 * at a time, not thread-safe.
 */
class ColoringGeneral {
public:

    enum class NodeOp : uint8_t {
        // terms (integer-valued nodes)
        ConstTerm, ParamTerm, StateColTerm, ChoiceColTerm, AddTerm, SubTerm, MulTerm, IteTerm,
        // formulas (boolean-valued nodes)
        TrueF, FalseF, NotF, AndF, OrF, ImpliesF, IteF,
        EqF, NeF, LtF, LeF, GtF, GeF,
        InSetF, InBitsF,
    };

    /**
     * @param row_groups nondeterministic choice indices (row groups) of the underlying quotient MDP
     * @param num_parameters number of parameters (holes) of the parameter space this coloring is defined over
     * @param node_op, node_a, node_b, node_c a flat node table (a small hash-consed IR/DAG): node i is
     *   (node_op[i], node_a[i], node_b[i], node_c[i]). The meaning of a/b/c depends on node_op:
     *     ConstTerm: a = the constant value.
     *     ParamTerm: a = parameter index.
     *     StateColTerm / ChoiceColTerm: a = column index into state_data / choice_data.
     *     AddTerm / SubTerm / MulTerm: a, b = child term node ids.
     *     IteTerm: a = condition (formula node id), b = then (term node id), c = else (term node id).
     *     TrueF / FalseF: unused.
     *     NotF: a = child formula node id.
     *     AndF / OrF / ImpliesF: a, b = child formula node ids.
     *     IteF: a = condition, b = then, c = else (all formula node ids).
     *     EqF / NeF / LtF / LeF / GtF / GeF: a, b = child TERM node ids.
     *     InSetF: a = a term node id that must be a bare ParamTerm; b = index into option_sets.
     *     InBitsF: a = a term node id that must be a bare ParamTerm; b = a term node id evaluating to a 64-bit
     *       bitmask (e.g. a data column); true iff the parameter's value is a set bit. The bitmask must not depend on
     *       a parameter (checked when evaluated); options >= 64 are never set.
     * @param option_sets for InSetF nodes, a pool of constant option sets (each a sorted vector of distinct
     *   option indices; the order is not checked)
     * @param choice_root for each choice, the id of its color's root FORMULA node, or -1 if the choice is
     *   uncolored (always enabled under any assignment, matching Coloring's own convention)
     * @param state_data, choice_data per-state / per-choice data columns, row-major ([entity][column])
     * @param state_is_relevant irrelevant states are never evaluated (see selectCompatibleChoices). A BitVector whose
     *   size is not the number of states, e.g. an empty one, makes every state relevant.
     */
    ColoringGeneral(
        std::vector<uint64_t> const& row_groups,
        uint64_t num_parameters,
        std::vector<uint8_t> const& node_op,
        std::vector<int64_t> const& node_a,
        std::vector<int32_t> const& node_b,
        std::vector<int32_t> const& node_c,
        std::vector<std::vector<uint64_t>> const& option_sets,
        std::vector<int32_t> const& choice_root,
        std::vector<std::vector<int64_t>> const& state_data,
        std::vector<std::vector<int64_t>> const& choice_data,
        BitVector const& state_is_relevant
    );

    /**
     * Lift a plain Coloring (a conjunction of (hole,option) pairs per choice) into a ColoringGeneral, built as a
     * conjunction of exact equality atoms. Used for differential testing against Coloring. 
     * It has no data columns, and every state is relevant.
     */
    static std::shared_ptr<ColoringGeneral> fromColoring(
        Family const& family, std::vector<uint64_t> const& row_groups, Coloring const& coloring
    );

    /**
     * Enable reachability-restricted selection (BFS from initial_state via choice_destinations), matching
     * ColoringSmt::enableStateExploration.
     */
    void enableStateExploration(uint64_t initial_state, std::vector<std::vector<uint64_t>> const& choice_destinations);

    /** Decide the colors that the Kleene evaluation leaves unknown with Z3, one solver per such choice: 
     * exact, but orders of magnitude slower. */
    void setExactMode(bool exact);

    /**
     * The choices that some member of the family may enable: at a relevant state, every uncolored choice and every
     * colored one whose color is not F; at an irrelevant state, its first choice, whatever its color. With state
     * exploration, only the states reached from the initial state get choices. A relevant state where no member of
     * the family enables a choice keeps none.
     */
    BitVector selectCompatibleChoices(Family const& family);
    /**
     * As above, but a colored choice must also be in base_choices, and an irrelevant state gets its first choice in
     * base_choices (none, without an error, if there is none). Sound when family narrows the family that produced
     * base_choices.
     */
    BitVector selectCompatibleChoices(Family const& family, BitVector const& base_choices);

    /** Static per-state parameter supports: the parameters in the colors of the state's choices, whatever the family, sorted. */
    std::vector<std::vector<uint64_t>> const& getStateToHoles() const;

    /**
     * Per-query refinement of getStateToHoles, for conflict minimization (Theorem 6 of the SMPMC paper): a set P of
     * parameters such that widening the domain of any parameter outside P (keeping those in P as in family) selects
     * exactly the same choices at each of the given (underlying-MDP) states, as selectCompatibleChoices selects them.
     * Irrelevant states contribute nothing.
     *
     * It is the union, over every choice of those states whose color is certainly FALSE under family, of a Kleene
     * "reason": parameters that alone keep that color FALSE. A choice that is selected needs no reason, since Kleene
     * evaluation is monotone under domain-widening (a selected choice stays selected). In exact mode a color that is
     * undecided may still have been excluded by the exact check, and that has no Kleene reason, so its whole support
     * counts. Hence, for the reachable states of the sub-MDP induced by family, the sub-MDP -- and any refutation
     * drawn from it -- is unchanged by such a widening.
     *
     * Unlike getStateToHoles, for a decision-tree coloring this is only the decisions and thresholds actually tested
     * and the leaves actually reached, not every parameter of the tree.
     */
    BitVector relevantParameters(Family const& family, std::vector<uint64_t> const& states);

    /**
     * Verify whether the given choices have a common satisfying parameter assignment (within the given family), i.e.
     * whether the conjunction of their colors is satisfiable. Choices at irrelevant states and uncolored choices
     * impose nothing. Decided by Z3, without the Kleene evaluation; no split is proposed for an inconsistent choice set.
     * @return (A,B): A iff the choices are consistent; if A holds, B is a satisfying assignment (one option for every
     *   parameter); otherwise B is empty (an empty list per parameter).
     */
    std::pair<bool,std::vector<std::vector<uint64_t>>> areChoicesConsistent(BitVector const& choices, Family const& family);

    /**
     * The node table the coloring was built from, as the constructor takes it: (node_op, node_a, node_b, node_c, option_sets). Read-only access for rebuilding
     * the formula of a choice outside the engine, together with getChoiceColor (see paynt/utils/coloring.py).
     */
    std::tuple<std::vector<uint8_t>, std::vector<int64_t>, std::vector<int32_t>, std::vector<int32_t>, std::vector<std::vector<uint64_t>>> getNodeTable() const;

    /**
     * What the color of a choice is made of: (root, state_row, choice_row, state_is_relevant, is_first_of_state).
     * root is the formula node of the color in the node table, or -1 if the choice is uncolored. state_row and choice_row are the data rows (the columns
     * that the formula reads) of the choice's state and of the choice itself. selectCompatibleChoices evaluates the color only at a relevant state; at an
     * irrelevant one it enables the first choice only, which is_first_of_state tells apart (with base_choices, the first one in base_choices instead).
     * @throws std::out_of_range if there is no such choice
     */
    std::tuple<int32_t, std::vector<int64_t>, std::vector<int64_t>, bool, bool> getChoiceColor(uint64_t choice) const;

protected:

    const std::vector<uint64_t> row_groups;
    uint64_t numStates() const;
    uint64_t numChoices() const;
    const uint64_t num_parameters;
    uint64_t numNodes() const;

    // the node table
    std::vector<NodeOp> node_op;
    std::vector<int64_t> node_a;
    std::vector<int32_t> node_b;
    std::vector<int32_t> node_c;
    std::vector<std::vector<uint64_t>> option_sets;
    std::vector<int32_t> choice_root;
    std::vector<std::vector<int64_t>> state_data;
    std::vector<std::vector<int64_t>> choice_data;
    const BitVector state_is_relevant;

    // build-time derived info
    // the cache of a node: it reads no data (Query), state data (State) or choice data (Choice)
    enum class Scope : uint8_t { Query = 0, State = 1, Choice = 2 };
    std::vector<Scope> node_scope;
    std::vector<BitVector> node_support; // per node, the parameters (transitively) referenced
    std::vector<std::vector<uint64_t>> state_to_holes_; // static per-state supports, computed once at construction

    bool exact_mode = false;
    bool state_exploration_enabled = false;
    uint64_t initial_state = 0;
    std::vector<std::vector<uint64_t>> choice_destinations;
    void visitChoice(uint64_t choice, BitVector& state_reached, std::queue<uint64_t>& unexplored) const;

    // Kleene evaluation

    enum class Tri : uint8_t { F = 0, U = 1, T = 2 };
    struct Interval { int64_t lo; int64_t hi; };

    // current query context (set by the public entry points below)
    Family const* family = nullptr;
    uint64_t current_state = 0;
    uint64_t current_choice = 0;

    // stamped per-scope memo tables: valid iff stamp[node] == the scope's current epoch
    uint64_t query_epoch = 0, state_epoch = 0, choice_epoch = 0;
    std::vector<uint64_t> query_stamp, state_stamp, choice_stamp;
    std::vector<Tri> query_tri, state_tri, choice_tri;
    std::vector<Interval> query_iv, state_iv, choice_iv;

    static Tri triNot(Tri t);
    Tri evalFormula(int32_t node);
    Interval evalTerm(int32_t node);
    Tri computeFormula(int32_t node);
    Interval computeTerm(int32_t node);
    Tri evalComparison(NodeOp op, int32_t a, int32_t b);
    Tri evalInSet(int32_t param_term, int32_t set_idx);
    Tri evalInBits(int32_t param_term, int32_t data_term);
    // read a just-evaluated node's cached Tri (same context as the eval that produced it)
    Tri cachedTri(int32_t node) const;

    // Kleene "reason" (see relevantParameters): assumes evalFormula(node) was just called in the same context
    void unionReasonTrue(int32_t node, BitVector& out) const;
    void unionReasonFalse(int32_t node, BitVector& out) const;

    // exact (Z3) mode and areChoicesConsistent

    // One Z3 context and one integer constant per parameter, for the object's whole lifetime (built once, in the
    // constructor); the family's domain constraints and the choices' formulas are added per query.
    z3::context ctx;
    std::vector<z3::expr> param_vars;

    struct Z3BuildContext {
        z3::context& ctx;
        std::vector<z3::expr> const& param_vars;
        std::vector<int64_t> const* state_row;
        std::vector<int64_t> const* choice_row;
    };
    z3::expr buildTerm(int32_t node, Z3BuildContext const& bc) const;
    z3::expr buildFormula(int32_t node, Z3BuildContext const& bc) const;
    z3::expr domainConstraint(uint64_t parameter, z3::expr const& var, Family const& family, z3::context& ctx) const;
    bool exactSat(int32_t node, Family const& family, std::vector<int64_t> const& state_row, std::vector<int64_t> const& choice_row);
    // evaluates a data-only term (Const/StateCol/ChoiceCol/Add/Sub/Mul, no parameters) to a concrete value, without
    // touching the mutable Kleene-eval caches -- used by buildFormula's InBitsF case to resolve the bitmask word
    // against the row explicitly passed in via Z3BuildContext, independent of current_state/current_choice.
    int64_t constTermValue(int32_t node, std::vector<int64_t> const& state_row, std::vector<int64_t> const& choice_row) const;

    std::vector<std::vector<uint64_t>> extractAssignment(z3::model const& model) const;

};

}
