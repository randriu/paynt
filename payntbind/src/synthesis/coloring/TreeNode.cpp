#include "TreeNode.h"

#include <storm/exceptions/UnexpectedException.h>

#include <sstream>

namespace synthesis {


TreeNode::Hole::Hole(bool is_interval_type, z3::context& ctx)
: is_interval_type(is_interval_type), solver_variable(ctx), solver_variable_harm(ctx), restriction(ctx) {
    // left intentionally blank
}

TreeNode::Hole::Hole(bool is_interval_type, z3::context& ctx, uint64_t depth)
: is_interval_type(is_interval_type), solver_variable(ctx), solver_variable_harm(ctx), restriction(ctx), depth(depth) {
    // left intentionally blank
}

void TreeNode::Hole::createSolverVariable() {
    solver_variable = solver_variable.ctx().int_const(name.c_str());
    solver_string_domain = "h_" + std::to_string(hole);
    solver_string_domain_harm = "z_" + std::to_string(hole);
    solver_string_restriction = "r_" + std::to_string(hole);

    name_harm = name + "_h";
    solver_variable_harm = solver_variable.ctx().int_const(name_harm.c_str());
}

uint64_t TreeNode::Hole::getModelValue(z3::model const& model) const {
    return model.eval(solver_variable).get_numeral_int64();
}
uint64_t TreeNode::Hole::getModelValueHarmonizing(z3::model const& model) const {
    return model.eval(solver_variable_harm).get_numeral_int64();
}

z3::expr TreeNode::Hole::domainEncoding(Family const& subfamily, bool harmonizing) const {
    z3::expr const& variable = not harmonizing ? solver_variable : solver_variable_harm;
    std::vector<uint64_t> const& domain = subfamily.holeOptions(hole);
    if(is_interval_type) {
        int domain_min = domain.front();
        int domain_max = domain.back();
        if(domain_min < domain_max) {
            return (domain_min <= variable) and (variable <= domain_max);
        } else {
            return (variable == domain_min);
        }
    } else {
        z3::expr_vector options(variable.ctx());
        for(uint64_t option: domain) {
            options.push_back(variable == (int)option);
        }
        return z3::mk_or(options);
    }
}

void TreeNode::Hole::addDomainEncoding(Family const& subfamily, z3::solver& solver) const {
    solver.add(domainEncoding(subfamily, false), solver_string_domain.c_str());
    solver.add(domainEncoding(subfamily, true), solver_string_domain_harm.c_str());

    /*if(is_interval_type) {
        std::cout << solver_string_restriction.c_str() << " : " << restriction << std::endl;
        solver.add(restriction, solver_string_restriction.c_str());    
    }*/
}

bool TreeNode::verifyExpression(z3::solver & solver, z3::expr const& expr) {
    solver.push();
    solver.add(expr);
    z3::check_result result = solver.check();
    solver.pop();
    return result == z3::sat;
}

TreeNode::TreeNode(
    uint64_t identifier, z3::context& ctx,
    std::vector<std::string> const& variable_name,
    std::vector<std::vector<int64_t>> const& variable_domain
) : identifier(identifier), ctx(ctx), variable_name(variable_name), variable_domain(variable_domain) {
    parent = NULL;
    child_true = NULL;
    child_false = NULL;
    depth = 0;
}

const uint64_t TreeNode::numVariables() const {
    return variable_name.size();
}

void TreeNode::createTree(
    std::vector<std::tuple<uint64_t,uint64_t,uint64_t>> const& tree_list,
    std::vector<std::shared_ptr<TreeNode>> & tree
) {
    auto [parent_index,child_true_index,child_false_index] = tree_list[identifier];
    if(parent_index < tree.size()) {
        parent = tree[parent_index];
        depth = parent->depth+1;
    }
    if(child_true_index < tree.size()) {
        child_true = tree[child_true_index];
        child_true->createTree(tree_list,tree);
    }
    if(child_false_index < tree.size()) {
        child_false = tree[child_false_index];
        child_false->createTree(tree_list,tree);
    }
}

bool TreeNode::isRoot() const {
    return parent == NULL;
}

bool TreeNode::isTerminal() const {
    return child_true == NULL;
}

bool TreeNode::isTrueChild() const {
    return identifier == parent->child_true->identifier;
}

std::shared_ptr<TreeNode> TreeNode::getChild(bool condition) const {
    return condition ? child_true : child_false;
}

/*const TreeNode *TreeNode::getNodeOfPath(std::vector<bool> const& path, uint64_t step) const {
    if(step == depth) {
        return this;
    }
    bool step_to_true_child = path[depth];
    return getChild(step_to_true_child)->getNodeOfPath(path,step);
}*/




TerminalNode::TerminalNode(
    uint64_t identifier, z3::context & ctx,
    std::vector<std::string> const& variable_name,
    std::vector<std::vector<int64_t>> const& variable_domain,
    uint64_t num_actions
) : TreeNode(identifier,ctx,variable_name,variable_domain),
    num_actions(num_actions), action_hole(false,ctx),
    action_expression(ctx), action_expression_harmonizing(ctx) {}

const TerminalNode *TerminalNode::getTerminal(std::vector<bool> const& path) const {
    return this;
}

void TerminalNode::createHoles(Family& family) {
    action_hole.hole = family.addHole(num_actions);
    action_hole.name = "A_" + std::to_string(identifier);
    action_hole.depth = depth;
    action_hole.createSolverVariable();
}

void TerminalNode::loadHoleInfo(std::vector<std::tuple<uint64_t,std::string,std::string>> & hole_info) const {
    hole_info[action_hole.hole] = std::make_tuple(identifier,action_hole.name,"__action__");
}

void TerminalNode::createPaths(z3::expr const& harmonizing_variable) {
    paths.push_back({true});
    // paths_ptr.push_back({std::shared_ptr<TreeNode>(this)});
}

uint64_t TerminalNode::getPathActionHole(std::vector<bool> const& path) {
    return action_hole.hole;
}

void TerminalNode::substituteActionExpressions() {
    for(uint64_t action = 0; action < num_actions; ++action) {
        action_expression.push_back(action_hole.solver_variable == ctx.int_val(action));
    }
}

void TerminalNode::substituteActionExpressionsHarmonizing(z3::expr const& harmonizing_variable) {
    for(uint64_t action = 0; action < num_actions; ++action) {
        action_expression_harmonizing.push_back(action_expression[action] or (harmonizing_variable == ctx.int_val(action_hole.hole) and action_hole.solver_variable_harm == ctx.int_val(action)));
    }
}

void TerminalNode::clearCache() {
    action_expression.resize(0);
    action_expression_harmonizing.resize(0);
}

void TerminalNode::addFamilyEncoding(Family const& subfamily, z3::solver& solver) const {
    action_hole.addDomainEncoding(subfamily,solver);
}

void TerminalNode::loadHoleAssignmentFromModel(z3::model const& model, std::vector<std::vector<uint64_t>> & hole_options) const {
    hole_options[action_hole.hole].push_back(action_hole.getModelValue(model));
}
void TerminalNode::loadHoleAssignmentFromModelHarmonizing(
    z3::model const& model, std::vector<std::vector<uint64_t>> & hole_options, uint64_t harmonizing_hole
) const {
    if(harmonizing_hole == action_hole.hole) {
        hole_options[action_hole.hole].push_back(action_hole.getModelValueHarmonizing(model));
    }
}

void TerminalNode::unsatCoreAnalysis(
    Family const& subfamily,
    std::vector<bool> const& path,
    std::vector<uint64_t> const& state_valuation,
    uint64_t path_action, bool action_enabled,
    std::vector<std::set<uint64_t>> & hole_options
) const {
    if(not action_enabled) {
        return;
    }
    hole_options[action_hole.hole].insert(path_action);
}



InnerNode::InnerNode(
    uint64_t identifier, z3::context & ctx,
    std::vector<std::string> const& variable_name,
    std::vector<std::vector<int64_t>> const& variable_domain
) : TreeNode(identifier,ctx,variable_name,variable_domain),
    decision_hole(false,ctx),
    clauses(ctx),
    decision_is_variable(ctx), decision_harm_is_variable(ctx), harm_is_hole(ctx),
    harm_decision_true(ctx), harm_decision_false(ctx),
    substituted_true(ctx), substituted_false(ctx),
    array_true(0), array_false(0)
{
    clauses_true.resize(numVariables(), ctx.bool_val(false));
    array_true.resize(numVariables());
    clauses_false.resize(numVariables(), ctx.bool_val(false));
    array_false.resize(numVariables());
}

const TerminalNode *InnerNode::getTerminal(std::vector<bool> const& path) const {
    return getChild(path[depth])->getTerminal(path);
}

void InnerNode::createHoles(Family& family) {
    decision_hole.hole = family.addHole(numVariables());
    decision_hole.name = "V_" + std::to_string(identifier);
    decision_hole.depth = depth;
    decision_hole.createSolverVariable();
    for(uint64_t variable = 0; variable < numVariables(); ++variable) {
        Hole hole(true,ctx,depth);
        hole.hole = family.addHole(variable_domain[variable].size()-1);
        hole.name = variable_name[variable] + "_" + std::to_string(identifier);
        hole.createSolverVariable();
        variable_hole.push_back(hole);
    }

    InnerNode *node = this;
    InnerNode *node_parent = std::dynamic_pointer_cast<InnerNode>(node->parent).get();
    std::vector<InnerNode *> lower_bounds;
    std::vector<InnerNode *> upper_bounds;
    while(node_parent != NULL) {
        if(node->isTrueChild()) {
            upper_bounds.push_back(node_parent);
        } else {
            lower_bounds.push_back(node_parent);
        }
        node = node_parent;
        node_parent = std::dynamic_pointer_cast<InnerNode>(node->parent).get();
    }
    for(uint64_t variable = 0; variable < numVariables(); ++variable) {
        // Vi=v => (V_lb=v => Xlb <= Xi) and (V_ub=v => Xi <= Xub)
        z3::expr lhs = decision_hole.solver_variable != ctx.int_val(variable);
        z3::expr_vector rhs(ctx);
        z3::expr const& var_node = variable_hole[variable].solver_variable;
        for(InnerNode *bound: lower_bounds) {
            z3::expr const& var_bound = bound->variable_hole[variable].solver_variable;
            rhs.push_back(bound->decision_hole.solver_variable != ctx.int_val(variable) or var_bound <= var_node);
        }
        for(InnerNode *bound: upper_bounds) {
            z3::expr const& var_bound = bound->variable_hole[variable].solver_variable;
            rhs.push_back(bound->decision_hole.solver_variable != ctx.int_val(variable) or var_node <= var_bound);
        }
        variable_hole[variable].restriction = lhs or z3::mk_and(rhs);
    }
    child_true->createHoles(family);
    child_false->createHoles(family);
}

void InnerNode::loadHoleInfo(std::vector<std::tuple<uint64_t,std::string,std::string>> & hole_info) const {
    hole_info[decision_hole.hole] = std::make_tuple(identifier,decision_hole.name,"__decision__");
    for(uint64_t variable = 0; variable < numVariables(); ++variable) {
        Hole const& hole = variable_hole[variable];
        hole_info[hole.hole] = std::make_tuple(identifier,hole.name,variable_name[variable]);
    }
    child_true->loadHoleInfo(hole_info);
    child_false->loadHoleInfo(hole_info);
}

void InnerNode::createPaths(z3::expr const& harmonizing_variable) {

    child_true->createPaths(harmonizing_variable);
    child_false->createPaths(harmonizing_variable);

    // create paths
    for(bool condition: {true,false}) {
        std::shared_ptr<TreeNode> child = getChild(condition);
        for(std::vector<bool> const& suffix: child->paths) {
            std::vector<bool> path;
            path.push_back(condition);
            path.insert(path.end(),suffix.begin(),suffix.end());
            paths.push_back(path);
        }
        /*for(std::vector<std::shared_ptr<TreeNode>> const& suffix: child->paths_ptr) {
            std::vector<std::shared_ptr<TreeNode>> path;
            path.push_back(std::shared_ptr<TreeNode>(this));
            path.insert(path.end(),suffix.begin(),suffix.end());
            paths_ptr.push_back(path);
        }*/
    }

    // cache sub-expressions
    for(uint64_t var = 0; var < numVariables(); ++var) {
        decision_is_variable.push_back(decision_hole.solver_variable == ctx.int_val(var));
        decision_harm_is_variable.push_back(decision_hole.solver_variable_harm == ctx.int_val(var));
        harm_is_hole.push_back(harmonizing_variable == ctx.int_val(variable_hole[var].hole));
    }
}

uint64_t InnerNode::getPathActionHole(std::vector<bool> const& path) {
    return getChild(path[depth])->getPathActionHole(path);
}

void InnerNode::createPrefixSubstitutions(std::vector<uint64_t> const& state_valuation, z3::expr_vector const& state_valuation_int) {
    uint64_t num_clauses_true  = 0;
    uint64_t num_clauses_false  = 0;
    for(uint64_t var = 0; var < numVariables(); ++var) {
        z3::expr const& vv = variable_hole[var].solver_variable;
        uint64_t largest_value = variable_domain[var].size()-1;
        uint64_t state_value = state_valuation[var];
        // mind the negation: not (Vi = j => sj<=xj) == (Vi = vj) and not (sj<=xj))
        if(state_value == 0) {
            // if sj is 0, then 0 <= xj is always true, i.e. the clause above is false, so we skip
        } else if(state_value == largest_value) {
            // if sj is max, then max <= xj is always false since xj cannot attain the value of max
            clauses_true[num_clauses_true] = decision_is_variable[var];
            array_true[num_clauses_true] = clauses_true[num_clauses_true];
            num_clauses_true++;
        } else {
            clauses_true[num_clauses_true] = (decision_is_variable[var] and not(state_valuation_int[var] <= vv));
            array_true[num_clauses_true] = clauses_true[num_clauses_true];
            num_clauses_true++;
        }

        if(state_value == largest_value) {
            // if sj is max, then max > xj is always true, i.e. the clause above is false, so we skip
        } else if(state_value == 0) {
            // if sj is 0, then 0 > xj is always false
            clauses_false[num_clauses_false] = decision_is_variable[var];
            array_false[num_clauses_false] = clauses_false[num_clauses_false];
            num_clauses_false++;
        } else {
            clauses_false[num_clauses_false] = (decision_is_variable[var] and not(state_valuation_int[var]  > vv));
            array_false[num_clauses_false] = clauses_false[num_clauses_false];
            num_clauses_false++;
        }
    }

    substituted_true = z3::expr(ctx, Z3_mk_or(ctx, num_clauses_true, array_true.ptr()) );
    substituted_false = z3::expr(ctx, Z3_mk_or(ctx, num_clauses_false, array_false.ptr()) );

    child_true->createPrefixSubstitutions(state_valuation, state_valuation_int);
    child_false->createPrefixSubstitutions(state_valuation, state_valuation_int);
}

void InnerNode::substitutePrefixExpression(std::vector<bool> const& path, z3::array<Z3_ast> & substituted) const {
    bool step_to_true_child = path[depth];
    // substituted.push_back(step_to_true_child ? substituted_true : substituted_false);
    substituted[depth] = step_to_true_child ? substituted_true : substituted_false;
    getChild(step_to_true_child)->substitutePrefixExpression(path,substituted);
}

void InnerNode::substituteActionExpressions() {
    child_true->substituteActionExpressions();
    child_false->substituteActionExpressions();
}

void InnerNode::createPrefixSubstitutionsHarmonizing(std::vector<uint64_t> const& state_valuation, z3::expr_vector const& state_valuation_int, z3::expr const& harmonizing_variable) {

    for(uint64_t var = 0; var < numVariables(); ++var) {
        z3::expr const& vv = variable_hole[var].solver_variable;
        z3::expr const& state_var = state_valuation_int[var];
        // clauses_true[var] = ( not(decision_harm_is_variable[var] and state_var <= vv));
        clauses_true[var] = ( decision_harm_is_variable[var] and not(state_var <= vv) );
        array_true[var] = clauses_true[var];
        // clauses_false[var] = (not(decision_harm_is_variable[var] and state_var  > vv));
        clauses_false[var] = ( decision_harm_is_variable[var] and not(state_var  > vv) );
        array_false[var] = clauses_false[var];
    }
    harm_decision_true =  harmonizing_variable == ctx.int_val(decision_hole.hole) and z3::expr(ctx, Z3_mk_or(ctx, numVariables(), array_true.ptr()) );
    harm_decision_false = harmonizing_variable == ctx.int_val(decision_hole.hole) and z3::expr(ctx, Z3_mk_or(ctx, numVariables(), array_false.ptr()) );

    for(uint64_t var = 0; var < numVariables(); ++var) {
        Hole const& hole = variable_hole[var];
        z3::expr const& vv = variable_hole[var].solver_variable;
        z3::expr const& vvh = variable_hole[var].solver_variable_harm;
        z3::expr const& state_var = state_valuation_int[var];
        // clauses_true[var] = (not(decision_is_variable[var] and state_var <= vv) or (harm_is_hole[var] and not(state_var <= vvh) ) );
        clauses_true[var] = decision_is_variable[var] and (not (state_var <= vv) or (harm_is_hole[var] and not(state_var <= vvh) ));
        array_true[var] = clauses_true[var];
        // clauses_false[var] = (not(decision_is_variable[var] and state_var  > vv) or (harm_is_hole[var] and not(state_var >  vvh) ) );
        clauses_false[var] = decision_is_variable[var] and (not (state_var > vv) or (harm_is_hole[var] and not(state_var > vvh) ));
        array_false[var] = clauses_false[var];
    }
    substituted_true = harm_decision_true or z3::expr(ctx, Z3_mk_or(ctx, numVariables(), array_true.ptr()) );
    substituted_false = harm_decision_false or z3::expr(ctx, Z3_mk_or(ctx, numVariables(), array_false.ptr()) );

    child_true->createPrefixSubstitutionsHarmonizing(state_valuation, state_valuation_int, harmonizing_variable);
    child_false->createPrefixSubstitutionsHarmonizing(state_valuation, state_valuation_int, harmonizing_variable);
}

void InnerNode::substitutePrefixExpressionHarmonizing(std::vector<bool> const& path, z3::array<Z3_ast> & substituted) const {
    bool step_to_true_child = path[depth];
    substituted[depth] = (step_to_true_child ? substituted_true : substituted_false);
    getChild(step_to_true_child)->substitutePrefixExpressionHarmonizing(path,substituted);
}

void InnerNode::substituteActionExpressionsHarmonizing(z3::expr const& harmonizing_variable) {
    child_true->substituteActionExpressionsHarmonizing(harmonizing_variable);
    child_false->substituteActionExpressionsHarmonizing(harmonizing_variable);
}

void InnerNode::clearCache() {
    decision_is_variable.resize(0);
    decision_harm_is_variable.resize(0);
    harm_is_hole.resize(0);
    clauses_true.clear();
    array_true.resize(0);
    clauses_false.clear();
    array_false.resize(0);

    child_true->clearCache();
    child_false->clearCache();
}

void InnerNode::addFamilyEncoding(Family const& subfamily, z3::solver& solver) const {
    decision_hole.addDomainEncoding(subfamily,solver);
    for(Hole const& hole: variable_hole) {
        hole.addDomainEncoding(subfamily,solver);
    }
    child_true->addFamilyEncoding(subfamily,solver);
    child_false->addFamilyEncoding(subfamily,solver);
}

void InnerNode::arePathsEnabledInState(Family const& subfamily, std::vector<uint64_t> const& state_valuation) {
   true_branch_enabled = false;
   false_branch_enabled = false;
    for(uint64_t variable: subfamily.holeOptions(decision_hole.hole)) {
        uint64_t value = state_valuation[variable];
        std::vector<uint64_t> const& domain = subfamily.holeOptions(variable_hole[variable].hole);
        if(value <= domain.back()) {
            true_branch_enabled = true;
        }
        if(value > domain.front()) {
            false_branch_enabled = true;
        }
    }
    child_true->arePathsEnabledInState(subfamily,state_valuation);
    child_false->arePathsEnabledInState(subfamily,state_valuation);
}

bool InnerNode::isPathEnabled(std::vector<bool> const& path) const {
    bool step_to_true_child = path[depth];
    if( (step_to_true_child and not true_branch_enabled) or (not step_to_true_child and not false_branch_enabled) ) {
        return false;
    }
    return getChild(step_to_true_child)->isPathEnabled(path);
}

void InnerNode::loadHoleAssignmentFromModel(z3::model const& model, std::vector<std::vector<uint64_t>> & hole_options) const {
    hole_options[decision_hole.hole].push_back(decision_hole.getModelValue(model));
    for(Hole const& hole: variable_hole) {
        hole_options[hole.hole].push_back(hole.getModelValue(model));
    }
    child_true->loadHoleAssignmentFromModel(model,hole_options);
    child_false->loadHoleAssignmentFromModel(model,hole_options);
}
void InnerNode::loadHoleAssignmentFromModelHarmonizing(
    z3::model const& model, std::vector<std::vector<uint64_t>> & hole_options, uint64_t harmonizing_hole
) const {
    const Hole *harmonizing_hole_ptr = NULL;
    if(harmonizing_hole == decision_hole.hole) {
        harmonizing_hole_ptr = &decision_hole;
    } else {
        for(Hole const& hole: variable_hole) {
            if(harmonizing_hole == hole.hole) {
                harmonizing_hole_ptr = &hole;
            }
        }
    }
    if(harmonizing_hole_ptr != NULL) {
        hole_options[harmonizing_hole_ptr->hole].push_back(harmonizing_hole_ptr->getModelValueHarmonizing(model));
        return;
    }
    child_true->loadHoleAssignmentFromModelHarmonizing(model,hole_options,harmonizing_hole);
    child_false->loadHoleAssignmentFromModelHarmonizing(model,hole_options,harmonizing_hole);
}

void InnerNode::unsatCoreAnalysis(
    Family const& subfamily,
    std::vector<bool> const& path,
    std::vector<uint64_t> const& state_valuation,
    uint64_t path_action, bool action_enabled,
    std::vector<std::set<uint64_t>> & hole_options
) const {
    bool step_to_true_child = path[depth];

    if(action_enabled) {
        bool sat_variable_found = false;
        uint64_t sat_variable;
        uint64_t sat_variable_option;
        for(uint64_t variable: subfamily.holeOptions(decision_hole.hole)) {
            std::vector<uint64_t> const& domain = subfamily.holeOptions(variable_hole[variable].hole);
            if(step_to_true_child and state_valuation[variable] <= domain.back()) {
                sat_variable_found = true;
                sat_variable = variable;
                sat_variable_option = domain.back();
                break;
            }
            if(not step_to_true_child and state_valuation[variable] > domain.front()) {
                sat_variable_found = true;
                sat_variable = variable;
                sat_variable_option = domain.front();
                break;
            }
        }
        STORM_LOG_THROW(
            sat_variable_found, storm::exceptions::UnexpectedException,
            "Sat variable assignment not found."
        );
        hole_options[decision_hole.hole].insert(sat_variable);
        hole_options[variable_hole[sat_variable].hole].insert(sat_variable_option);
    } else {
        bool sat_variable_found = false;
        uint64_t sat_variable;
        uint64_t sat_variable_option;
        for(uint64_t variable: subfamily.holeOptions(decision_hole.hole)) {
            std::vector<uint64_t> const& domain = subfamily.holeOptions(variable_hole[variable].hole);
            if(step_to_true_child and not (state_valuation[variable] <= domain.front()) ) {
                sat_variable_found = true;
                sat_variable = variable;
                sat_variable_option = domain.front();
                break;
            }
            if(not step_to_true_child and not (state_valuation[variable] > domain.back()) ) {
                sat_variable_found = true;
                sat_variable = variable;
                sat_variable_option = domain.back();
                break;
            }
        }
        if(sat_variable_found) {
            hole_options[decision_hole.hole].insert(sat_variable);
            hole_options[variable_hole[sat_variable].hole].insert(sat_variable_option);
            return;
        }
    }
    getChild(step_to_true_child)->unsatCoreAnalysis(subfamily,path,state_valuation,path_action,action_enabled,hole_options);
}

}
