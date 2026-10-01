#include "ColoringGeneral.h"

#include <storm/exceptions/UnexpectedException.h>
#include <storm/utility/macros.h>

#include <algorithm>
#include <map>

namespace synthesis {

// ==================================================================================================================
// construction
// ==================================================================================================================

ColoringGeneral::ColoringGeneral(
    std::vector<uint64_t> const& row_groups,
    uint64_t num_parameters,
    std::vector<uint8_t> const& node_op_raw,
    std::vector<int64_t> const& node_a,
    std::vector<int32_t> const& node_b,
    std::vector<int32_t> const& node_c,
    std::vector<std::vector<uint64_t>> const& option_sets,
    std::vector<int32_t> const& choice_root,
    std::vector<std::vector<int64_t>> const& state_data,
    std::vector<std::vector<int64_t>> const& choice_data,
    BitVector const& state_is_relevant_arg
) : row_groups(row_groups), num_parameters(num_parameters),
    node_a(node_a), node_b(node_b), node_c(node_c), option_sets(option_sets),
    choice_root(choice_root), state_data(state_data), choice_data(choice_data),
    state_is_relevant(state_is_relevant_arg.size() == row_groups.size()-1 ? state_is_relevant_arg : BitVector(row_groups.size()-1,true))
{
    STORM_LOG_THROW(
        node_a.size() == node_op_raw.size() and node_b.size() == node_op_raw.size() and node_c.size() == node_op_raw.size(),
        storm::exceptions::UnexpectedException, "node table arrays have inconsistent lengths"
    );
    STORM_LOG_THROW(
        choice_root.size() == numChoices(), storm::exceptions::UnexpectedException,
        "choice_root must have one entry per choice"
    );
    STORM_LOG_THROW(
        state_data.size() == numStates(), storm::exceptions::UnexpectedException, "state_data must have one row per state"
    );
    STORM_LOG_THROW(
        choice_data.size() == numChoices(), storm::exceptions::UnexpectedException, "choice_data must have one row per choice"
    );

    node_op.reserve(node_op_raw.size());
    for(auto op: node_op_raw) {
        node_op.push_back(static_cast<NodeOp>(op));
    }

    for(auto root: choice_root) {
        STORM_LOG_THROW(root < (int32_t)numNodes(), storm::exceptions::UnexpectedException, "choice_root references an invalid node");
    }

    // validate & compute per-node scope and (transitive) parameter support bottom-up; the node table is a DAG with
    // edges only to smaller ids (built bottom-up by the caller), so a single forward pass suffices
    node_scope.resize(numNodes());
    node_support.resize(numNodes());
    for(uint64_t n = 0; n < numNodes(); ++n) {
        BitVector support(num_parameters,false);
        Scope scope = Scope::Query;
        auto mergeChild = [&](int32_t child) {
            STORM_LOG_THROW(child >= 0 and (uint64_t)child < n, storm::exceptions::UnexpectedException, "node table is not a bottom-up DAG");
            support |= node_support[child];
            scope = std::max(scope, node_scope[child]);
        };
        switch(node_op[n]) {
            case NodeOp::ConstTerm: break;
            case NodeOp::ParamTerm: {
                uint64_t p = (uint64_t)node_a[n];
                STORM_LOG_THROW(p < num_parameters, storm::exceptions::UnexpectedException, "ParamTerm references an invalid parameter");
                support.set(p,true);
                break;
            }
            case NodeOp::StateColTerm: scope = Scope::State; break;
            case NodeOp::ChoiceColTerm: scope = Scope::Choice; break;
            case NodeOp::AddTerm: case NodeOp::SubTerm: case NodeOp::MulTerm:
                mergeChild(node_b[n]); mergeChild((int32_t)node_a[n]); break;
            case NodeOp::IteTerm:
                mergeChild((int32_t)node_a[n]); mergeChild(node_b[n]); mergeChild(node_c[n]); break;
            case NodeOp::TrueF: case NodeOp::FalseF: break;
            case NodeOp::NotF: mergeChild((int32_t)node_a[n]); break;
            case NodeOp::AndF: case NodeOp::OrF: case NodeOp::ImpliesF:
                mergeChild((int32_t)node_a[n]); mergeChild(node_b[n]); break;
            case NodeOp::IteF:
                mergeChild((int32_t)node_a[n]); mergeChild(node_b[n]); mergeChild(node_c[n]); break;
            case NodeOp::EqF: case NodeOp::NeF: case NodeOp::LtF: case NodeOp::LeF: case NodeOp::GtF: case NodeOp::GeF:
                mergeChild((int32_t)node_a[n]); mergeChild(node_b[n]); break;
            case NodeOp::InSetF: {
                int32_t term = (int32_t)node_a[n];
                STORM_LOG_THROW(
                    term >= 0 and (uint64_t)term < n and node_op[term] == NodeOp::ParamTerm, storm::exceptions::UnexpectedException,
                    "InSetF's operand must be a bare parameter reference"
                );
                STORM_LOG_THROW((uint64_t)node_b[n] < option_sets.size(), storm::exceptions::UnexpectedException, "InSetF references an invalid option set");
                mergeChild(term);
                break;
            }
            case NodeOp::InBitsF: {
                int32_t term = (int32_t)node_a[n];
                STORM_LOG_THROW(
                    term >= 0 and (uint64_t)term < n and node_op[term] == NodeOp::ParamTerm, storm::exceptions::UnexpectedException,
                    "InBitsF's first operand must be a bare parameter reference"
                );
                mergeChild(term); mergeChild(node_b[n]);
                break;
            }
        }
        node_support[n] = std::move(support);
        node_scope[n] = scope;
    }

    // static per-state supports (Coloring's getStateToHoles), independent of any family
    state_to_holes_.resize(numStates());
    for(uint64_t state = 0; state < numStates(); ++state) {
        BitVector support(num_parameters,false);
        for(uint64_t choice = row_groups[state]; choice < row_groups[state+1]; ++choice) {
            if(choice_root[choice] >= 0) {
                support |= node_support[choice_root[choice]];
            }
        }
        state_to_holes_[state] = std::move(support);
    }

    query_stamp.assign(numNodes(),0); state_stamp.assign(numNodes(),0); choice_stamp.assign(numNodes(),0);
    query_tri.assign(numNodes(),Tri::U); state_tri.assign(numNodes(),Tri::U); choice_tri.assign(numNodes(),Tri::U);
    query_iv.assign(numNodes(),Interval{0,0}); state_iv.assign(numNodes(),Interval{0,0}); choice_iv.assign(numNodes(),Interval{0,0});

    // Z3 constants for areChoicesConsistent/exactSat (see the header) -- built once, here
    param_vars.reserve(num_parameters);
    for(uint64_t p = 0; p < num_parameters; ++p) {
        param_vars.push_back(ctx.int_const(("p" + std::to_string(p)).c_str()));
    }
}

std::shared_ptr<ColoringGeneral> ColoringGeneral::fromColoring(
    Family const& family, std::vector<uint64_t> const& row_groups, Coloring const& coloring
) {
    auto const& choice_to_assignment = coloring.getChoiceToAssignment();
    uint64_t num_parameters = family.numHoles();
    uint64_t num_choices = choice_to_assignment.size();

    std::vector<uint8_t> node_op;
    std::vector<int64_t> node_a;
    std::vector<int32_t> node_b, node_c;
    auto addNode = [&](NodeOp op, int64_t a, int32_t b, int32_t c) -> int32_t {
        node_op.push_back(static_cast<uint8_t>(op));
        node_a.push_back(a); node_b.push_back(b); node_c.push_back(c);
        return (int32_t)(node_op.size()-1);
    };

    // one ParamTerm node per hole, built lazily and shared across choices
    std::vector<int32_t> param_node(num_parameters,-1);
    auto paramNode = [&](uint64_t hole) -> int32_t {
        if(param_node[hole] < 0) {
            param_node[hole] = addNode(NodeOp::ParamTerm,(int64_t)hole,0,0);
        }
        return param_node[hole];
    };
    // one EQ(hole,option) atom per distinct (hole,option) pair, shared across choices
    std::map<std::pair<uint64_t,uint64_t>,int32_t> eq_node;

    std::vector<int32_t> choice_root(num_choices,-1);
    for(uint64_t choice = 0; choice < num_choices; ++choice) {
        auto const& assignment = choice_to_assignment[choice];
        if(assignment.empty()) {
            continue; // uncolored: stays -1, matching Coloring's own always-included convention
        }
        int32_t conjunct = -1;
        for(auto const& [hole,option]: assignment) {
            auto key = std::make_pair(hole,option);
            int32_t atom;
            auto it = eq_node.find(key);
            if(it != eq_node.end()) {
                atom = it->second;
            } else {
                int32_t k = addNode(NodeOp::ConstTerm,(int64_t)option,0,0);
                atom = addNode(NodeOp::EqF,paramNode(hole),k,0);
                eq_node[key] = atom;
            }
            conjunct = conjunct < 0 ? atom : addNode(NodeOp::AndF,conjunct,atom,0);
        }
        choice_root[choice] = conjunct;
    }

    std::vector<std::vector<int64_t>> state_data(row_groups.size()-1);
    std::vector<std::vector<int64_t>> choice_data(num_choices);
    BitVector state_is_relevant; // empty -> every state relevant (default)
    return std::make_shared<ColoringGeneral>(
        row_groups, num_parameters, node_op, node_a, node_b, node_c,
        std::vector<std::vector<uint64_t>>{}, choice_root, state_data, choice_data, state_is_relevant
    );
}

uint64_t ColoringGeneral::numStates() const { return row_groups.size()-1; }
uint64_t ColoringGeneral::numChoices() const { return row_groups.back(); }
uint64_t ColoringGeneral::numNodes() const { return node_op.size(); }

void ColoringGeneral::enableStateExploration(uint64_t initial_state, std::vector<std::vector<uint64_t>> const& choice_destinations) {
    this->state_exploration_enabled = true;
    this->initial_state = initial_state;
    this->choice_destinations = choice_destinations;
}

void ColoringGeneral::setExactMode(bool exact) {
    this->exact_mode = exact;
}

void ColoringGeneral::visitChoice(uint64_t choice, BitVector& state_reached, std::queue<uint64_t>& unexplored) const {
    if(not state_exploration_enabled) {
        return;
    }
    for(uint64_t dst: choice_destinations[choice]) {
        if(not state_reached[dst]) {
            state_reached.set(dst,true);
            unexplored.push(dst);
        }
    }
}

std::vector<BitVector> const& ColoringGeneral::getStateToHoles() const {
    return state_to_holes_;
}

// ==================================================================================================================
// Kleene evaluation
// ==================================================================================================================

ColoringGeneral::Tri ColoringGeneral::evalFormula(int32_t node) {
    Scope scope = node_scope[node];
    uint64_t epoch = scope == Scope::Query ? query_epoch : (scope == Scope::State ? state_epoch : choice_epoch);
    auto& stamp = scope == Scope::Query ? query_stamp : (scope == Scope::State ? state_stamp : choice_stamp);
    auto& tri = scope == Scope::Query ? query_tri : (scope == Scope::State ? state_tri : choice_tri);
    if(stamp[node] == epoch) {
        return tri[node];
    }
    Tri result = computeFormula(node);
    stamp[node] = epoch;
    tri[node] = result;
    return result;
}

ColoringGeneral::Tri ColoringGeneral::cachedTri(int32_t node) const {
    Scope scope = node_scope[node];
    auto const& stamp = scope == Scope::Query ? query_stamp : (scope == Scope::State ? state_stamp : choice_stamp);
    auto const& tri = scope == Scope::Query ? query_tri : (scope == Scope::State ? state_tri : choice_tri);
    uint64_t epoch = scope == Scope::Query ? query_epoch : (scope == Scope::State ? state_epoch : choice_epoch);
    STORM_LOG_THROW(stamp[node] == epoch, storm::exceptions::UnexpectedException, "cachedTri read before evalFormula in this context");
    return tri[node];
}

ColoringGeneral::Interval ColoringGeneral::evalTerm(int32_t node) {
    Scope scope = node_scope[node];
    uint64_t epoch = scope == Scope::Query ? query_epoch : (scope == Scope::State ? state_epoch : choice_epoch);
    auto& stamp = scope == Scope::Query ? query_stamp : (scope == Scope::State ? state_stamp : choice_stamp);
    auto& iv = scope == Scope::Query ? query_iv : (scope == Scope::State ? state_iv : choice_iv);
    if(stamp[node] == epoch) {
        return iv[node];
    }
    Interval result = computeTerm(node);
    stamp[node] = epoch;
    iv[node] = result;
    return result;
}

ColoringGeneral::Tri ColoringGeneral::triNot(Tri t) {
    return t == Tri::F ? Tri::T : (t == Tri::T ? Tri::F : Tri::U);
}

ColoringGeneral::Tri ColoringGeneral::computeFormula(int32_t node) {
    NodeOp op = node_op[node];
    int32_t a = (int32_t)node_a[node], b = node_b[node], c = node_c[node];
    switch(op) {
        case NodeOp::TrueF: return Tri::T;
        case NodeOp::FalseF: return Tri::F;
        case NodeOp::NotF: return triNot(evalFormula(a));
        case NodeOp::AndF: {
            Tri x = evalFormula(a);
            if(x == Tri::F) return Tri::F;
            Tri y = evalFormula(b);
            if(y == Tri::F) return Tri::F;
            return (x == Tri::T and y == Tri::T) ? Tri::T : Tri::U;
        }
        case NodeOp::OrF: {
            Tri x = evalFormula(a);
            if(x == Tri::T) return Tri::T;
            Tri y = evalFormula(b);
            if(y == Tri::T) return Tri::T;
            return (x == Tri::F and y == Tri::F) ? Tri::F : Tri::U;
        }
        case NodeOp::ImpliesF: { // a -> b == (not a) or b
            Tri x = evalFormula(a);
            if(x == Tri::F) return Tri::T;
            Tri y = evalFormula(b);
            if(y == Tri::T) return Tri::T;
            return (x == Tri::T and y == Tri::F) ? Tri::F : Tri::U;
        }
        case NodeOp::IteF: {
            Tri cond = evalFormula(a);
            if(cond == Tri::T) return evalFormula(b);
            if(cond == Tri::F) return evalFormula(c);
            Tri t1 = evalFormula(b), t2 = evalFormula(c);
            return t1 == t2 ? t1 : Tri::U;
        }
        case NodeOp::EqF: case NodeOp::NeF: case NodeOp::LtF: case NodeOp::LeF: case NodeOp::GtF: case NodeOp::GeF:
            return evalComparison(op,a,b);
        case NodeOp::InSetF: return evalInSet(a,b);
        case NodeOp::InBitsF: return evalInBits(a,b);
        default:
            STORM_LOG_THROW(false, storm::exceptions::UnexpectedException, "not a formula node");
    }
    return Tri::U;
}

ColoringGeneral::Interval ColoringGeneral::computeTerm(int32_t node) {
    NodeOp op = node_op[node];
    int64_t a64 = node_a[node];
    int32_t a = (int32_t)a64, b = node_b[node], c = node_c[node];
    switch(op) {
        case NodeOp::ConstTerm: return {a64,a64};
        case NodeOp::ParamTerm: {
            auto const& opts = family->holeOptions((uint64_t)a64);
            return {(int64_t)opts.front(), (int64_t)opts.back()};
        }
        case NodeOp::StateColTerm: { int64_t v = state_data[current_state][(uint64_t)a64]; return {v,v}; }
        case NodeOp::ChoiceColTerm: { int64_t v = choice_data[current_choice][(uint64_t)a64]; return {v,v}; }
        case NodeOp::AddTerm: { auto ia = evalTerm(a), ib = evalTerm(b); return {ia.lo+ib.lo, ia.hi+ib.hi}; }
        case NodeOp::SubTerm: { auto ia = evalTerm(a), ib = evalTerm(b); return {ia.lo-ib.hi, ia.hi-ib.lo}; }
        case NodeOp::MulTerm: {
            auto ia = evalTerm(a), ib = evalTerm(b);
            int64_t p[4] = {ia.lo*ib.lo, ia.lo*ib.hi, ia.hi*ib.lo, ia.hi*ib.hi};
            int64_t lo = *std::min_element(p,p+4), hi = *std::max_element(p,p+4);
            return {lo,hi};
        }
        case NodeOp::IteTerm: {
            Tri cond = evalFormula(a);
            if(cond == Tri::T) return evalTerm(b);
            if(cond == Tri::F) return evalTerm(c);
            auto t1 = evalTerm(b), t2 = evalTerm(c);
            return {std::min(t1.lo,t2.lo), std::max(t1.hi,t2.hi)};
        }
        default:
            STORM_LOG_THROW(false, storm::exceptions::UnexpectedException, "not a term node");
    }
    return {0,0};
}

ColoringGeneral::Tri ColoringGeneral::evalComparison(NodeOp op, int32_t a, int32_t b) {
    // exactness boost: if one side is a bare parameter and the other side's interval is currently a single
    // concrete value, use Family's exact option mask instead of the (possibly loose) interval hull -- this is
    // what reproduces ColoringSmt's exact selection semantics for e.g. `leaf_param == choice_data("action")`
    if(op == NodeOp::EqF or op == NodeOp::NeF) {
        bool a_is_param = node_op[a] == NodeOp::ParamTerm;
        bool b_is_param = node_op[b] == NodeOp::ParamTerm;
        int32_t param_node_id = -1, other_node_id = -1;
        if(a_is_param and not b_is_param) { param_node_id = a; other_node_id = b; }
        else if(b_is_param and not a_is_param) { param_node_id = b; other_node_id = a; }
        if(param_node_id >= 0) {
            Interval other = evalTerm(other_node_id);
            if(other.lo == other.hi) {
                uint64_t p = (uint64_t)node_a[param_node_id];
                uint64_t v = (uint64_t)other.lo;
                bool possiblyIn = other.lo >= 0 and family->holeContains(p,v);
                bool certainlyIn = possiblyIn and family->holeNumOptions(p) == 1;
                Tri eq = certainlyIn ? Tri::T : (possiblyIn ? Tri::U : Tri::F);
                return op == NodeOp::EqF ? eq : triNot(eq);
            }
        }
    }
    Interval ia = evalTerm(a), ib = evalTerm(b);
    switch(op) {
        case NodeOp::EqF:
            if(ia.lo == ia.hi and ib.lo == ib.hi and ia.lo == ib.lo) return Tri::T;
            if(ia.hi < ib.lo or ib.hi < ia.lo) return Tri::F;
            return Tri::U;
        case NodeOp::NeF: {
            if(ia.lo == ia.hi and ib.lo == ib.hi and ia.lo == ib.lo) return Tri::F;
            if(ia.hi < ib.lo or ib.hi < ia.lo) return Tri::T;
            return Tri::U;
        }
        case NodeOp::LtF:
            if(ia.hi < ib.lo) return Tri::T;
            if(ia.lo >= ib.hi) return Tri::F;
            return Tri::U;
        case NodeOp::LeF:
            if(ia.hi <= ib.lo) return Tri::T;
            if(ia.lo > ib.hi) return Tri::F;
            return Tri::U;
        case NodeOp::GtF:
            if(ib.hi < ia.lo) return Tri::T;
            if(ib.lo >= ia.hi) return Tri::F;
            return Tri::U;
        case NodeOp::GeF:
            if(ib.hi <= ia.lo) return Tri::T;
            if(ib.lo > ia.hi) return Tri::F;
            return Tri::U;
        default:
            STORM_LOG_THROW(false, storm::exceptions::UnexpectedException, "not a comparison op");
    }
    return Tri::U;
}

ColoringGeneral::Tri ColoringGeneral::evalInSet(int32_t param_term, int32_t set_idx) {
    uint64_t p = (uint64_t)node_a[param_term];
    auto const& set_vals = option_sets[set_idx];
    auto const& domain = family->holeOptions(p);
    bool any_in = false, any_out = false;
    for(uint64_t v: domain) {
        bool in = std::binary_search(set_vals.begin(),set_vals.end(),v);
        (in ? any_in : any_out) = true;
        if(any_in and any_out) return Tri::U;
    }
    return any_in ? Tri::T : Tri::F;
}

ColoringGeneral::Tri ColoringGeneral::evalInBits(int32_t param_term, int32_t data_term) {
    uint64_t p = (uint64_t)node_a[param_term];
    Interval iv = evalTerm(data_term);
    STORM_LOG_THROW(iv.lo == iv.hi, storm::exceptions::UnexpectedException, "InBitsF's data operand must evaluate to a concrete value");
    uint64_t word = (uint64_t)iv.lo;
    auto const& domain = family->holeOptions(p);
    bool any_in = false, any_out = false;
    for(uint64_t v: domain) {
        bool bit = v < 64 and ((word >> v) & 1ULL);
        (bit ? any_in : any_out) = true;
        if(any_in and any_out) return Tri::U;
    }
    return any_in ? Tri::T : Tri::F;
}

// ==================================================================================================================
// selectCompatibleChoices
// ==================================================================================================================

BitVector ColoringGeneral::selectCompatibleChoices(Family const& family) {
    return selectCompatibleChoices(family, BitVector(numChoices(),true));
}

BitVector ColoringGeneral::selectCompatibleChoices(Family const& family, BitVector const& base_choices) {
    this->family = &family;
    ++query_epoch;

    BitVector selection(numChoices(),false);
    std::queue<uint64_t> unexplored;
    BitVector state_reached(numStates(),false);
    if(state_exploration_enabled) {
        unexplored.push(initial_state);
        state_reached.set(initial_state,true);
    } else {
        for(uint64_t state = 0; state < numStates(); ++state) {
            unexplored.push(state);
        }
    }

    while(not unexplored.empty()) {
        uint64_t state = unexplored.front(); unexplored.pop();
        current_state = state;
        ++state_epoch;

        if(not state_is_relevant[state]) {
            for(uint64_t choice = row_groups[state]; choice < row_groups[state+1]; ++choice) {
                if(not base_choices[choice]) continue;
                selection.set(choice,true);
                visitChoice(choice,state_reached,unexplored);
                break;
            }
            continue;
        }

        bool any_enabled = false;
        for(uint64_t choice = row_groups[state]; choice < row_groups[state+1]; ++choice) {
            if(choice_root[choice] < 0) {
                // uncolored: always included, regardless of base_choices -- matches Coloring's own invariant
                any_enabled = true;
                selection.set(choice,true);
                visitChoice(choice,state_reached,unexplored);
                continue;
            }
            if(not base_choices[choice]) continue;
            current_choice = choice;
            ++choice_epoch;
            Tri t = evalFormula(choice_root[choice]);
            bool keep = t != Tri::F;
            if(t == Tri::U and exact_mode) {
                // resolve the undecided color exactly (a color that is Kleene-FALSE is unsatisfiable, nothing to check)
                keep = exactSat(choice_root[choice], family, state_data[state], choice_data[choice]);
            }
            if(keep) {
                any_enabled = true;
                selection.set(choice,true);
                visitChoice(choice,state_reached,unexplored);
            }
        }
        STORM_LOG_THROW(any_enabled, storm::exceptions::UnexpectedException, "no choice is available in the sub-MDP");
    }
    return selection;
}

// ==================================================================================================================
// relevantParameters
// ==================================================================================================================

void ColoringGeneral::unionReasonTrue(int32_t node, BitVector& out) const {
    NodeOp op = node_op[node];
    int32_t a = (int32_t)node_a[node], b = node_b[node], c = node_c[node];
    switch(op) {
        case NodeOp::TrueF: return;
        case NodeOp::NotF: unionReasonFalse(a,out); return;
        case NodeOp::AndF: unionReasonTrue(a,out); unionReasonTrue(b,out); return; // both children needed true
        case NodeOp::OrF: { // either child suffices
            if(cachedTri(a) == Tri::T) unionReasonTrue(a,out); else unionReasonTrue(b,out);
            return;
        }
        case NodeOp::ImpliesF: { // a=F or b=T
            if(cachedTri(a) == Tri::F) unionReasonFalse(a,out); else unionReasonTrue(b,out);
            return;
        }
        case NodeOp::IteF: {
            Tri cond = cachedTri(a);
            if(cond == Tri::T) { unionReasonTrue(a,out); unionReasonTrue(b,out); return; }
            if(cond == Tri::F) { unionReasonFalse(a,out); unionReasonTrue(c,out); return; }
            unionReasonTrue(b,out); unionReasonTrue(c,out); return; // cond undecided, both branches agree T
        }
        default: out |= node_support[node]; return; // atoms: fall back to full syntactic support (sound, usually small)
    }
}

void ColoringGeneral::unionReasonFalse(int32_t node, BitVector& out) const {
    NodeOp op = node_op[node];
    int32_t a = (int32_t)node_a[node], b = node_b[node], c = node_c[node];
    switch(op) {
        case NodeOp::FalseF: return;
        case NodeOp::NotF: unionReasonTrue(a,out); return;
        case NodeOp::AndF: { // either child suffices
            if(cachedTri(a) == Tri::F) unionReasonFalse(a,out); else unionReasonFalse(b,out);
            return;
        }
        case NodeOp::OrF: unionReasonFalse(a,out); unionReasonFalse(b,out); return; // both children needed false
        case NodeOp::ImpliesF: unionReasonTrue(a,out); unionReasonFalse(b,out); return; // a=T and b=F
        case NodeOp::IteF: {
            Tri cond = cachedTri(a);
            if(cond == Tri::T) { unionReasonTrue(a,out); unionReasonFalse(b,out); return; }
            if(cond == Tri::F) { unionReasonFalse(a,out); unionReasonFalse(c,out); return; }
            unionReasonFalse(b,out); unionReasonFalse(c,out); return;
        }
        default: out |= node_support[node]; return;
    }
}

BitVector ColoringGeneral::relevantParameters(Family const& family, std::vector<uint64_t> const& states) {
    this->family = &family;
    ++query_epoch;
    BitVector result(num_parameters,false);
    for(uint64_t state: states) {
        if(not state_is_relevant[state]) {
            continue; // irrelevant states select their first choice whatever the family is
        }
        current_state = state;
        ++state_epoch;
        for(uint64_t choice = row_groups[state]; choice < row_groups[state+1]; ++choice) {
            if(choice_root[choice] < 0) continue;
            current_choice = choice;
            ++choice_epoch;
            int32_t root = choice_root[choice];
            Tri t = evalFormula(root);
            if(t == Tri::F) {
                unionReasonFalse(root,result);
            } else if(t == Tri::U and exact_mode) {
                // the exact check may have excluded this undecided color, and that has no Kleene reason: the whole support has to stay fixed
                result |= node_support[root];
            }
            // otherwise the choice is selected, and stays so under any widening (Kleene evaluation is monotone)
        }
    }
    return result;
}

// ==================================================================================================================
// exact (Z3) mode
// ==================================================================================================================

z3::expr ColoringGeneral::domainConstraint(uint64_t parameter, z3::expr const& var, Family const& family, z3::context& ctx) const {
    z3::expr_vector options(ctx);
    for(uint64_t v: family.holeOptions(parameter)) {
        options.push_back(var == ctx.int_val((int64_t)v));
    }
    return z3::mk_or(options);
}

z3::expr ColoringGeneral::buildTerm(int32_t node, Z3BuildContext const& bc) const {
    NodeOp op = node_op[node];
    int64_t a64 = node_a[node];
    int32_t a = (int32_t)a64, b = node_b[node], c = node_c[node];
    switch(op) {
        case NodeOp::ConstTerm: return bc.ctx.int_val((int64_t)a64);
        case NodeOp::ParamTerm: return bc.param_vars[(uint64_t)a64];
        case NodeOp::StateColTerm: return bc.ctx.int_val((int64_t)(*bc.state_row)[(uint64_t)a64]);
        case NodeOp::ChoiceColTerm: return bc.ctx.int_val((int64_t)(*bc.choice_row)[(uint64_t)a64]);
        case NodeOp::AddTerm: return buildTerm(a,bc) + buildTerm(b,bc);
        case NodeOp::SubTerm: return buildTerm(a,bc) - buildTerm(b,bc);
        case NodeOp::MulTerm: return buildTerm(a,bc) * buildTerm(b,bc);
        case NodeOp::IteTerm: return z3::ite(buildFormula(a,bc), buildTerm(b,bc), buildTerm(c,bc));
        default:
            STORM_LOG_THROW(false, storm::exceptions::UnexpectedException, "not a term node");
    }
    return bc.ctx.int_val(0);
}

z3::expr ColoringGeneral::buildFormula(int32_t node, Z3BuildContext const& bc) const {
    NodeOp op = node_op[node];
    int64_t a64 = node_a[node];
    int32_t a = (int32_t)a64, b = node_b[node], c = node_c[node];
    switch(op) {
        case NodeOp::TrueF: return bc.ctx.bool_val(true);
        case NodeOp::FalseF: return bc.ctx.bool_val(false);
        case NodeOp::NotF: return !buildFormula(a,bc);
        case NodeOp::AndF: return buildFormula(a,bc) and buildFormula(b,bc);
        case NodeOp::OrF: return buildFormula(a,bc) or buildFormula(b,bc);
        case NodeOp::ImpliesF: return z3::implies(buildFormula(a,bc), buildFormula(b,bc));
        case NodeOp::IteF: return z3::ite(buildFormula(a,bc), buildFormula(b,bc), buildFormula(c,bc));
        case NodeOp::EqF: return buildTerm(a,bc) == buildTerm(b,bc);
        case NodeOp::NeF: return buildTerm(a,bc) != buildTerm(b,bc);
        case NodeOp::LtF: return buildTerm(a,bc) < buildTerm(b,bc);
        case NodeOp::LeF: return buildTerm(a,bc) <= buildTerm(b,bc);
        case NodeOp::GtF: return buildTerm(a,bc) > buildTerm(b,bc);
        case NodeOp::GeF: return buildTerm(a,bc) >= buildTerm(b,bc);
        case NodeOp::InSetF: {
            z3::expr term = buildTerm(a,bc);
            z3::expr_vector options(bc.ctx);
            for(uint64_t v: option_sets[b]) {
                options.push_back(term == bc.ctx.int_val((int64_t)v));
            }
            return options.empty() ? bc.ctx.bool_val(false) : z3::mk_or(options);
        }
        case NodeOp::InBitsF: {
            // the parameter's value, restricted to its current family domain, must have a set bit in the (concrete)
            // data word -- built as a disjunction over the domain, exactly like domainConstraint
            uint64_t p = (uint64_t)node_a[a];
            z3::expr term = buildTerm(a,bc);
            uint64_t word = (uint64_t)constTermValue(b, *bc.state_row, *bc.choice_row);
            z3::expr_vector options(bc.ctx);
            for(uint64_t v: family->holeOptions(p)) {
                if(v < 64 and ((word >> v) & 1ULL)) {
                    options.push_back(term == bc.ctx.int_val((int64_t)v));
                }
            }
            return options.empty() ? bc.ctx.bool_val(false) : z3::mk_or(options);
        }
        default:
            STORM_LOG_THROW(false, storm::exceptions::UnexpectedException, "not a formula node");
    }
    return bc.ctx.bool_val(false);
}

int64_t ColoringGeneral::constTermValue(int32_t node, std::vector<int64_t> const& state_row, std::vector<int64_t> const& choice_row) const {
    NodeOp op = node_op[node];
    int64_t a64 = node_a[node];
    switch(op) {
        case NodeOp::ConstTerm: return a64;
        case NodeOp::StateColTerm: return state_row[(uint64_t)a64];
        case NodeOp::ChoiceColTerm: return choice_row[(uint64_t)a64];
        case NodeOp::AddTerm: return constTermValue((int32_t)a64,state_row,choice_row) + constTermValue(node_b[node],state_row,choice_row);
        case NodeOp::SubTerm: return constTermValue((int32_t)a64,state_row,choice_row) - constTermValue(node_b[node],state_row,choice_row);
        case NodeOp::MulTerm: return constTermValue((int32_t)a64,state_row,choice_row) * constTermValue(node_b[node],state_row,choice_row);
        default:
            STORM_LOG_THROW(false, storm::exceptions::UnexpectedException, "InBitsF's data operand must be a data-only term (no parameters, no ite)");
    }
    return 0;
}

bool ColoringGeneral::exactSat(
    int32_t node, Family const& family, std::vector<int64_t> const& state_row, std::vector<int64_t> const& choice_row
) {
    this->family = &family;
    z3::solver solver(ctx);
    for(uint64_t p: node_support[node]) {
        solver.add(domainConstraint(p,param_vars[p],family,ctx));
    }
    Z3BuildContext bc{ctx, param_vars, &state_row, &choice_row};
    solver.add(buildFormula(node,bc));
    return solver.check() == z3::sat;
}

// ==================================================================================================================
// areChoicesConsistent
// ==================================================================================================================

std::vector<std::vector<uint64_t>> ColoringGeneral::extractAssignment(z3::model const& model) const {
    std::vector<std::vector<uint64_t>> result(num_parameters);
    for(uint64_t p = 0; p < num_parameters; ++p) {
        int64_t v = model.eval(param_vars[p]).get_numeral_int64();
        result[p] = {(uint64_t)v};
    }
    return result;
}

std::pair<bool,std::vector<std::vector<uint64_t>>> ColoringGeneral::areChoicesConsistent(BitVector const& choices, Family const& family) {
    this->family = &family;
    z3::solver solver(ctx);
    for(uint64_t p = 0; p < num_parameters; ++p) {
        solver.add(domainConstraint(p,param_vars[p],family,ctx));
    }
    for(uint64_t state = 0; state < numStates(); ++state) {
        if(not state_is_relevant[state]) continue;
        for(uint64_t choice = row_groups[state]; choice < row_groups[state+1]; ++choice) {
            if(not choices[choice] or choice_root[choice] < 0) continue;
            Z3BuildContext bc{ctx, param_vars, &state_data[state], &choice_data[choice]};
            solver.add(buildFormula(choice_root[choice],bc));
        }
    }
    if(solver.check() == z3::sat) {
        return {true, extractAssignment(solver.get_model())};
    }
    return {false, std::vector<std::vector<uint64_t>>(num_parameters)};
}

}
