#include "synthesis.h"

#include <storm/adapters/RationalNumberAdapter.h>
#include <storm/logic/Formula.h>
#include <storm/logic/UntilFormula.h>
#include <storm/logic/OperatorFormula.h>
#include <storm/logic/EventuallyFormula.h>
#include <storm/logic/RewardOperatorFormula.h>
#include <storm/logic/ProbabilityOperatorFormula.h>
#include <storm/logic/FormulaContext.h>

#include <storm/utility/initialize.h>
#include <storm/environment/solver/NativeSolverEnvironment.h>
#include <storm/environment/solver/MinMaxSolverEnvironment.h>
#include <storm/environment/solver/GmmxxSolverEnvironment.h>
#include <storm/environment/solver/SolverEnvironment.h>
#include <storm/storage/SparseMatrix.h>
#include <storm/models/sparse/Model.h>

#include <storm/storage/jani/TemplateEdge.h>

namespace synthesis {

template<typename ValueType>
std::shared_ptr<storm::logic::Formula> transformUntilToEventually(
    storm::logic::Formula const& formula
) {
    auto const& of = formula.asOperatorFormula();
    bool is_reward = of.isRewardOperatorFormula();

    auto ef = std::make_shared<storm::logic::EventuallyFormula>(
        of.getSubformula().asUntilFormula().getRightSubformula().asSharedPointer(),
        !is_reward ? storm::logic::FormulaContext::Probability : storm::logic::FormulaContext::Reward);

    std::shared_ptr<storm::logic::Formula> modified_formula;
    if(!is_reward) {
        modified_formula = std::make_shared<storm::logic::ProbabilityOperatorFormula>(ef, of.getOperatorInformation());
    } else {
        modified_formula = std::make_shared<storm::logic::RewardOperatorFormula>(ef, of.asRewardOperatorFormula().getRewardModelName(), of.getOperatorInformation());
    }

    return modified_formula;
}

template<typename ValueType>
void removeRewardModel(storm::models::sparse::Model<ValueType> & model, std::string const& reward_name) {
    model.removeRewardModel(reward_name);
}

bool janiTemplateEdgeAddTransientAssignment(storm::jani::TemplateEdge & template_edge, storm::jani::Assignment const& assignment, bool add_to_existing = false) {
    return template_edge.addTransientAssignment(assignment,add_to_existing);
}

void janiTemplateEdgeAddAssignments(storm::jani::TemplateEdge & template_edge, storm::jani::OrderedAssignments const& assignments) {
    for(auto const& assignment: assignments) {
        template_edge.addTransientAssignment(assignment);
    }
}

}


void define_helpers(py::module& m) {

    m.def("set_loglevel_off", []() { storm::utility::setLogLevel(l3pp::LogLevel::OFF); }, "set loglevel for storm to off");

    m.def("set_precision_native", [](storm::NativeSolverEnvironment& nsenv, double value) {
        nsenv.setPrecision(storm::utility::convertNumber<storm::RationalNumber>(value));
    });
    m.def("set_precision_minmax", [](storm::MinMaxSolverEnvironment& nsenv, double value) {
        nsenv.setPrecision(storm::utility::convertNumber<storm::RationalNumber>(value));
    });
    m.def("set_max_iterations_minmax", [](storm::MinMaxSolverEnvironment& nsenv, uint64_t value) {
        nsenv.setMaximalNumberOfIterations(value);
    });
    // MinMaxSolverEnvironment's own iteration cap (above) bounds the *outer* policy-improvement loop of
    // policy iteration, but each outer iteration evaluates the current policy via an inner linear
    // equation solve -- gmmxx by default -- which has its own, separate iteration cap unaffected by the
    // one above. On a large model that inner solve can dominate wall-clock time by itself (confirmed:
    // "GmmxxLinearEquationSolver did not converge" observed on a 1.65M-state model even with the outer
    // minmax cap set to 1), so it needs capping too for the same reason the outer one does.
    m.def("set_max_iterations_gmmxx", [](storm::SolverEnvironment& senv, uint64_t value) {
        senv.gmmxx().setMaximalNumberOfIterations(value);
    });
    // Confirmed the above alone is not sufficient: gmmxx's default method (GMRES) restarts its Krylov
    // subspace every getRestartThreshold() iterations (see GmmxxLinearEquationSolver.cpp's gmm::gmres
    // call), a *separate* limit from setMaximalNumberOfIterations -- observed directly: raising the
    // iteration cap from 1 to 10000 did nothing, but the solver still reported "did not converge within
    // 22 iteration(s)" every time, exactly matching the (small, matrix-size-derived) default restart
    // threshold rather than the cap.
    m.def("set_restart_threshold_gmmxx", [](storm::SolverEnvironment& senv, uint64_t value) {
        senv.gmmxx().setRestartThreshold(value);
    });

    m.def("transform_until_to_eventually", &synthesis::transformUntilToEventually<double>, py::arg("formula"));
    m.def("remove_reward_model", &synthesis::removeRewardModel<double>, py::arg("model"), py::arg("reward_name"));
    m.def("remove_reward_model_exact", &synthesis::removeRewardModel<storm::RationalNumber>, py::arg("model"), py::arg("reward_name"));

    m.def("multiply_with_vector", [] (storm::storage::SparseMatrix<double> matrix,std::vector<double> vector) {
        std::vector<double> result(matrix.getRowCount());
        matrix.multiplyWithVector(vector, result);
        return result;
    }, py::arg("matrix"), py::arg("vector"));

    m.def("multiply_with_vector_exact", [] (storm::storage::SparseMatrix<storm::RationalNumber> matrix,std::vector<storm::RationalNumber> vector) {
        std::vector<storm::RationalNumber> result(matrix.getRowCount());
        matrix.multiplyWithVector(vector, result);
        return result;
    }, py::arg("matrix"), py::arg("vector"));

    m.def("janiTemplateEdgeAddAssignments", &synthesis::janiTemplateEdgeAddAssignments, py::arg("template_edge"), py::arg("assignments"));
}

