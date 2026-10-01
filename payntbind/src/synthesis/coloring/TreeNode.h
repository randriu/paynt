#pragma once

#include "src/synthesis/coloring/Family.h"

#include <storm/storage/BitVector.h>
#include <storm/storage/valuations/Valuations.h>
#include <storm/utility/Stopwatch.h>

#include <cstdint>
#include <vector>
#include <memory>

#include <z3++.h>

namespace synthesis {

using BitVector = storm::storage::BitVector;

class TerminalNode;

class TreeNode {
public:

    class Hole {
    public:
        /** Unique hole identifier. */
        uint64_t hole;
        /** Hole name. */
        std::string name;
        /** True if the variable is of interval type, else if it is of enumeration type. */
        bool is_interval_type;

        z3::expr solver_variable;
        std::string solver_string_domain;

        // harmonizing variable
        std::string name_harm;
        z3::expr solver_variable_harm;
        std::string solver_string_domain_harm;

        z3::expr restriction;
        std::string solver_string_restriction;

        uint64_t depth;

        Hole(bool is_interval_type, z3::context& ctx);
        Hole(bool is_interval_type, z3::context& ctx, uint64_t depth);
        void createSolverVariable();
        uint64_t getModelValue(z3::model const& model) const;
        uint64_t getModelValueHarmonizing(z3::model const& model) const;

        z3::expr domainEncoding(Family const& subfamily, bool harmonizing) const;
        void addDomainEncoding(Family const& subfamily, z3::solver& solver) const;

    protected:
        /** Express option domain as an interval. */
        z3::expr domainInterval(Family const& subfamily) const;
        /** Express option domain as enumeration. */
        z3::expr domainEnumeration(Family const& subfamily) const;
    };

    /** Verify expression in the using provided solver. The solver may already contain some assertions. */
    static bool verifyExpression(z3::solver & solver, z3::expr const& expr);

    /** Unique node identifier. */
    uint64_t identifier;
    /** Solver context. */
    z3::context& ctx;
    /** List of variable names. */
    std::vector<std::string> const& variable_name;
    /** List of variable domains, used to derive number of hole options. */
    std::vector<std::vector<int64_t>> const& variable_domain;

    /** Parent node, NULL for the root node. */
    std::shared_ptr<TreeNode> parent;
    /** Child node if the condition holds, NULL for the terminal node. */
    std::shared_ptr<TreeNode> child_true;
    /** Child node if the condition does not hold, NULL for the terminal node. */
    std::shared_ptr<TreeNode> child_false;

    /** Depth of this node in the tree. */
    uint64_t depth;
    /** Every possible path (a sequence of condition steps) that can be taken from this node. */
    std::vector<std::vector<bool>> paths;
    /** Every possible path (a sequence of nodes) that can be taken from this node. */
    // std::vector<std::vector<std::shared_ptr<TreeNode>>> paths_ptr;

    /** Create a tree node with the unique identifier. */
    TreeNode(
        uint64_t identifier, z3::context& ctx,
        std::vector<std::string> const& variable_name,
        std::vector<std::vector<int64_t>> const& variable_domain
    );
    /** Number of program variables. */
    const uint64_t numVariables() const;

    /** Store pointers to parent and children, if exist. */
    void createTree(
        std::vector<std::tuple<uint64_t,uint64_t,uint64_t>> const& tree_list,
        std::vector<std::shared_ptr<TreeNode>> & tree
    );

    /** Return true if this node has no parent. */
    bool isRoot() const;
    /** Return true if this node has no children. */
    bool isTerminal() const;
    /** Return true if this node if a true child of its parent. */
    bool isTrueChild() const;
    /** Retrieve true child of this node if the condition holds, get false child otherwise. */
    std::shared_ptr<TreeNode> getChild(bool condition) const;

    /** Retreive the terminal node of the given path. */
    virtual const TerminalNode *getTerminal(std::vector<bool> const& path) const {return NULL;};

    /** Create all holes and solver variables associated with this node. */
    virtual void createHoles(Family& family) {}
    /** Collect name and type (action, decision, or variable) of each hole. */
    virtual void loadHoleInfo(std::vector<std::tuple<uint64_t,std::string,std::string>> & hole_info) const {}

    /** Create a list of paths from this node. */
    virtual void createPaths(z3::expr const& harmonizing_variable) {}
    /** Retrieve action hole associated with the path. */
    virtual uint64_t getPathActionHole(std::vector<bool> const& path) {return 0;}

    /** Add a step expression evaluated for a given state valuation. */
    virtual void createPrefixSubstitutions(std::vector<uint64_t> const& state_valuation, z3::expr_vector const& state_valuation_int) {};
    virtual void substitutePrefixExpression(std::vector<bool> const& path, z3::array<Z3_ast> & substituted) const {};
    /** Add an action expression evaluated for a given state valuation. */
    virtual void substituteActionExpressions() {};

    /** Add a step expression evaluated for a given state valuation (harmonizing). */
    virtual void createPrefixSubstitutionsHarmonizing(std::vector<uint64_t> const& state_valuation, z3::expr_vector const& state_valuation_int, z3::expr const& harmonizing_variable) {};
    virtual void substitutePrefixExpressionHarmonizing(std::vector<bool> const& path, z3::array<Z3_ast> & substituted) const {};
    /** Add an action expression evaluated for a given state valuation (harmonizing). */
    virtual void substituteActionExpressionsHarmonizing(z3::expr const& harmonizing_variable) {};

    /** Clear cache. */
    virtual void clearCache() {};

    /** Add encoding of hole option in the given family. */
    virtual void addFamilyEncoding(Family const& subfamily, z3::solver & solver) const {}
    /** Set enabledness of true or false branch for a given state. */
    virtual void arePathsEnabledInState(Family const& subfamily, std::vector<uint64_t> const& state_valuation) {};
    /** Check whether the path is enabled in the given state for the given subfamily. */
    virtual bool isPathEnabled(std::vector<bool> const& path) const {return true;};
    
    /** Extract hole assignments used in the SMT model. */
    virtual void loadHoleAssignmentFromModel(
        z3::model const& model, std::vector<std::vector<uint64_t>> & hole_options
    ) const {}
    virtual void loadHoleAssignmentFromModelHarmonizing(
        z3::model const& model, std::vector<std::vector<uint64_t>> & hole_options, uint64_t harmonizing_hole
    ) const {}
    
    virtual void unsatCoreAnalysis(
        Family const& subfamily,
        std::vector<bool> const& path,
        std::vector<uint64_t> const& state_valuation,
        uint64_t path_action, bool action_enabled,
        std::vector<std::set<uint64_t>> & hole_options
    ) const {}

};


class TerminalNode: public TreeNode {
public:
    const uint64_t num_actions;
    Hole action_hole;

    // cache
    z3::expr_vector action_expression;
    z3::expr_vector action_expression_harmonizing;

    TerminalNode(
        uint64_t identifier, z3::context & ctx,
        std::vector<std::string> const& variable_name,
        std::vector<std::vector<int64_t>> const& variable_domain,
        uint64_t num_actions
    );

    const TerminalNode *getTerminal(std::vector<bool> const& path) const override;

    void createHoles(Family& family) override;
    void loadHoleInfo(std::vector<std::tuple<uint64_t,std::string,std::string>> & hole_info) const override;
    void createPaths(z3::expr const& harmonizing_variable) override;
    uint64_t getPathActionHole(std::vector<bool> const& path);

    void substituteActionExpressions() override;
    void substituteActionExpressionsHarmonizing(z3::expr const& harmonizing_variable) override;

    void clearCache() override;

    void addFamilyEncoding(Family const& subfamily, z3::solver & solver) const override;

    void loadHoleAssignmentFromModel(
        z3::model const& model, std::vector<std::vector<uint64_t>> & hole_options
    ) const override;
    void loadHoleAssignmentFromModelHarmonizing(
        z3::model const& model, std::vector<std::vector<uint64_t>> & hole_options, uint64_t harmonizing_hole
    ) const override;

    void unsatCoreAnalysis(
        Family const& subfamily,
        std::vector<bool> const& path,
        std::vector<uint64_t> const& state_valuation,
        uint64_t path_action, bool action_enabled,
        std::vector<std::set<uint64_t>> & hole_options
    ) const override;
};


class InnerNode: public TreeNode {
public:

    Hole decision_hole;
    std::vector<Hole> variable_hole;

    z3::expr_vector clauses;

    bool true_branch_enabled;
    bool false_branch_enabled;

    // Auxiliary variables to be reused during initalization for memory efficiency.
    // Calling \p clearCache clears these variables.
    z3::expr_vector decision_is_variable;
    z3::expr_vector decision_harm_is_variable;
    z3::expr_vector harm_is_hole;
    z3::expr harm_decision_true;
    z3::expr harm_decision_false;
    z3::expr substituted_true;
    z3::expr substituted_false;
    std::vector<z3::expr> clauses_true;
    z3::array<Z3_ast> array_true;
    std::vector<z3::expr> clauses_false;
    z3::array<Z3_ast> array_false;

    InnerNode(
        uint64_t identifier, z3::context & ctx,
        std::vector<std::string> const& variable_name,
        std::vector<std::vector<int64_t>> const& variable_domain
    );

    const TerminalNode *getTerminal(std::vector<bool> const& path) const override;

    void createHoles(Family& family) override;
    void loadHoleInfo(std::vector<std::tuple<uint64_t,std::string,std::string>> & hole_info) const override;
    void createPaths(z3::expr const& harmonizing_variable) override;
    uint64_t getPathActionHole(std::vector<bool> const& path);

    void createPrefixSubstitutions(std::vector<uint64_t> const& state_valuation, z3::expr_vector const& state_valuation_int) override;
    void substitutePrefixExpression(std::vector<bool> const& path, z3::array<Z3_ast> & substituted) const override;
    void substituteActionExpressions() override;

    void createPrefixSubstitutionsHarmonizing(std::vector<uint64_t> const& state_valuation, z3::expr_vector const& state_valuation_int, z3::expr const& harmonizing_variable) override;
    void substitutePrefixExpressionHarmonizing(std::vector<bool> const& path, z3::array<Z3_ast> & substituted) const override;
    void substituteActionExpressionsHarmonizing(z3::expr const& harmonizing_variable) override;

    void clearCache() override;

    void addFamilyEncoding(Family const& subfamily, z3::solver & solver) const override;
    void arePathsEnabledInState(Family const& subfamily, std::vector<uint64_t> const& state_valuation) override;
    bool isPathEnabled(std::vector<bool> const& path) const override;

    void loadHoleAssignmentFromModel(
        z3::model const& model, std::vector<std::vector<uint64_t>> & hole_options
    ) const override;
    void loadHoleAssignmentFromModelHarmonizing(
        z3::model const& model, std::vector<std::vector<uint64_t>> & hole_options, uint64_t harmonizing_hole
    ) const override;

    void unsatCoreAnalysis(
        Family const& subfamily,
        std::vector<bool> const& path,
        std::vector<uint64_t> const& state_valuation,
        uint64_t path_action, bool action_enabled,
        std::vector<std::set<uint64_t>> & hole_options
    ) const override;
};

}
