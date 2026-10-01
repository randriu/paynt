#pragma once

#include "src/synthesis/coloring/Coloring.h"
#include "src/synthesis/coloring/Family.h"

#include <storm/storage/BitVector.h>

#include <cstdint>
#include <memory>
#include <queue>
#include <utility>
#include <vector>

#include <z3++.h>

namespace synthesis {

using BitVector = storm::storage::BitVector;

/**
 * A general coloring (SMPMC paper, arXiv:2511.08078, Definition 2): every choice is colored by an arbitrary
 * quantifier-free formula over a finite parameter space, rather than Coloring's conjunction of (hole,option) pairs.
 * A full parameter assignment theta enables a choice c iff theta satisfies c's color.
 *
 * Choices sharing a formula shape ("template") are distinguished by per-state/per-choice data columns, so one
 * formula (an entry in a small, hash-consed node table, built once by the caller -- see
 * paynt/utils/coloring_builder.py) can color arbitrarily many choices without one Z3-sized formula per choice.
 *
 * Unlike ColoringSmt (specialized and tuned for decision-tree synthesis; must not change), this class is
 * deliberately generic: it knows nothing about trees or any other coloring kind, and never needs touching to
 * support a new one -- new colorings are authored entirely in Python against the node-table format below.
 *
 * Compatible-choice selection is a sound three-valued (Kleene) evaluation by default (never wrongly excludes a
 * choice, but may keep one that turns out incompatible on every concrete assignment); an optional exact mode
 * resolves undecided colors with an incremental Z3 check, at extra cost per query.
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
     *       bitmask (e.g. a data column); true iff the parameter's value is a set bit.
     * @param option_sets for InSetF nodes, a pool of constant option sets (each a sorted vector of distinct
     *   option indices)
     * @param choice_root for each choice, the id of its color's root FORMULA node, or -1 if the choice is
     *   uncolored (always enabled under any assignment, matching Coloring's own convention)
     * @param state_data, choice_data per-state / per-choice data columns, row-major ([entity][column])
     * @param state_is_relevant irrelevant states are never evaluated: selectCompatibleChoices enables only their
     *   first choice present in base_choices, exactly like ColoringSmt. Defaults to "every state is relevant" when
     *   given an empty BitVector.
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
     * conjunction of exact equality atoms. Used for differential testing against Coloring, and to run the generic
     * engines (CEGIS/OneByOne/SMPMC) on a colored MDP built with the standard Coloring.
     */
    static std::shared_ptr<ColoringGeneral> fromColoring(
        Family const& family, std::vector<uint64_t> const& row_groups, Coloring const& coloring
    );

    /**
     * Enable reachability-restricted selection (BFS from initial_state via choice_destinations), matching
     * ColoringSmt::enableStateExploration.
     */
    void enableStateExploration(uint64_t initial_state, std::vector<std::vector<uint64_t>> const& choice_destinations);

    /** Enable exact resolution of Kleene-undecided colors via an incremental Z3 check (extra cost per query). */
    void setExactMode(bool exact);

    /** Get a mask of choices compatible with the family (kept unless certainly FALSE). */
    BitVector selectCompatibleChoices(Family const& family);
    /**
     * As above, restricting the colored-choice search to base_choices. Sound whenever family is a narrowing of
     * whatever family produced base_choices -- see Coloring::selectCompatibleChoices for the argument, identical
     * here. Uncolored choices are always included, exactly as in Coloring.
     */
    BitVector selectCompatibleChoices(Family const& family, BitVector const& base_choices);

    /** Static (family-independent) per-state parameter supports, in Coloring's own format. */
    std::vector<BitVector> const& getStateToHoles() const;

    /**
     * Per-query refinement of getStateToHoles, for conflict minimization (Theorem 6 of the SMPMC paper): a set P of
     * parameters such that widening the domain of any parameter outside P (keeping those in P as in \p family) selects
     * exactly the same choices at each of the given (underlying-MDP) states.
     *
     * It is the union, over every choice of those states whose color is certainly FALSE under \p family, of a Kleene
     * "reason": parameters that alone keep that color FALSE. A choice that is selected needs no reason, since Kleene
     * evaluation is monotone under domain-widening (a selected choice stays selected). In exact mode a color that is
     * undecided may still have been excluded by the exact check, and that has no Kleene reason, so its whole support
     * counts. Hence, for the reachable states of the sub-MDP induced by \p family, the sub-MDP -- and any refutation
     * drawn from it -- is unchanged by such a widening.
     *
     * Unlike getStateToHoles, for a decision-tree coloring this is only the decisions and thresholds actually tested
     * and the leaves actually reached, not every parameter of the tree.
     */
    BitVector relevantParameters(Family const& family, std::vector<uint64_t> const& states);

    /**
     * Verify whether the given choices have a common satisfying parameter assignment (within the given family), i.e.
     * whether the conjunction of their colors is satisfiable. Choices at irrelevant states and uncolored choices
     * impose nothing. This is ColoringSmt::areChoicesConsistent without harmonization: no split is proposed for an
     * inconsistent choice set.
     * @return (A,B): A iff the choices are consistent; if A holds, B is a satisfying assignment (one option per
     *   parameter); otherwise B is empty (an empty list per parameter).
     */
    std::pair<bool,std::vector<std::vector<uint64_t>>> areChoicesConsistent(BitVector const& choices, Family const& family);

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
    enum class Scope : uint8_t { Query = 0, State = 1, Choice = 2 };
    std::vector<Scope> node_scope;
    std::vector<BitVector> node_support; // per node, the parameters (transitively) referenced
    std::vector<BitVector> state_to_holes_; // static per-state supports, computed once at construction

    bool exact_mode = false;
    bool state_exploration_enabled = false;
    uint64_t initial_state = 0;
    std::vector<std::vector<uint64_t>> choice_destinations;
    void visitChoice(uint64_t choice, BitVector& state_reached, std::queue<uint64_t>& unexplored) const;

    // -- Kleene evaluation --------------------------------------------------------------------------------------

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

    // -- exact (Z3) mode and areChoicesConsistent ----------------------------------------------------------------

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
