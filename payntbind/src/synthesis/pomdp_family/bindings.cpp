#include "src/synthesis/synthesis.h"

#include "ObservationEvaluator.h"
#include "FscUnfolder.h"
#include "GameAbstractionSolver.h"
#include "SmgAbstraction.h"

void bindings_pomdp_family(py::module& m) {

    py::classh<synthesis::ObservationEvaluator<double>>(m, "ObservationEvaluator")
        .def(py::init<storm::prism::Program &,storm::models::sparse::Model<double> const& >(), py::arg("prism"), py::arg("model"))
        .def_readonly("num_obs_expressions", &synthesis::ObservationEvaluator<double>::num_obs_expressions)
        .def_readonly("obs_expr_label", &synthesis::ObservationEvaluator<double>::obs_expr_label)
        .def_readonly("obs_expr_is_boolean", &synthesis::ObservationEvaluator<double>::obs_expr_is_boolean)
        .def_readonly("num_obs_classes", &synthesis::ObservationEvaluator<double>::num_obs_classes)
        .def_property("state_to_obs_class", &synthesis::ObservationEvaluator<double>::get_state_to_obs_class, &synthesis::ObservationEvaluator<double>::set_state_to_obs_class, "number of observation classes")
        .def("obs_class_value", &synthesis::ObservationEvaluator<double>::observationClassValue, py::arg("obs_class"), py::arg("obs_expr"))
        .def("obs_valuation", &synthesis::ObservationEvaluator<double>::observationValuation, py::arg("obs_class"))
        .def("add_observations_to_submdp", &synthesis::ObservationEvaluator<double>::addObservationsToSubMdp, py::arg("mdp"), py::arg("state_sub_to_full"))
        ;

    py::classh<synthesis::FscUnfolder<double>>(m, "FscUnfolder")
        .def(
            py::init<storm::models::sparse::Model<double> const&,
            std::vector<uint32_t> const&,
            uint64_t,
            std::vector<uint64_t> const&>()
        )
        .def("applyFsc", &synthesis::FscUnfolder<double>::applyFsc, py::arg("transitions"))
        .def("applyFscFactored", &synthesis::FscUnfolder<double>::applyFscFactored, py::arg("action_function"), py::arg("udate_function"))
        .def_property_readonly("product", [](synthesis::FscUnfolder<double>& m) {return m.product;} )
        .def_property_readonly("product_choice_to_choice", [](synthesis::FscUnfolder<double>& m) {return m.product_choice_to_choice;} )
        // .def_property_readonly("product_state_to_state", [](synthesis::FscUnfolder<double>& m) {return m.product_state_to_state;} )
        // .def_property_readonly("product_state_to_state_memory_action", [](synthesis::FscUnfolder<double>& m) {return m.product_state_to_state_memory_action;} )
        ;

    // m.def("randomize_action_variant", &synthesis::randomizeActionVariant<double>);
    py::classh<synthesis::GameAbstractionSolver<double>>(m, "GameAbstractionSolver")
        .def(
            py::init<
                storm::models::sparse::Model<double> const&,
                uint64_t,
                std::vector<uint64_t> const&,
                std::shared_ptr<storm::logic::Formula const>,
                bool,
                std::string const&,
                double
            >(),
            py::arg("quotient"), py::arg("num_actions"), py::arg("choice_to_action"), py::arg("formula"), py::arg("player1_maximizing"), py::arg("target_label"), py::arg("precision")
        )
        .def("solve_sg", &synthesis::GameAbstractionSolver<double>::solveSg)
        .def("solve_smg", &synthesis::GameAbstractionSolver<double>::solveSmg)
        .def_property_readonly("solution_state_values", [](synthesis::GameAbstractionSolver<double>& solver) {return solver.solution_state_values;})
        .def_property_readonly("solution_value", [](synthesis::GameAbstractionSolver<double>& solver) {return solver.solution_value;})
        .def_property_readonly("solution_state_to_player1_action", [](synthesis::GameAbstractionSolver<double>& solver) {return solver.solution_state_to_player1_action;})
        .def_property_readonly("solution_state_to_quotient_choice", [](synthesis::GameAbstractionSolver<double>& solver) {return solver.solution_state_to_quotient_choice;})
        .def("enable_profiling", &synthesis::GameAbstractionSolver<double>::enableProfiling)
        .def("print_profiling", &synthesis::GameAbstractionSolver<double>::printProfiling)
        ;

        py::classh<synthesis::SmgAbstraction<double>>(m, "SmgAbstraction")
        .def(py::init<
            storm::models::sparse::Model<double> const&,
            uint64_t,
            std::vector<uint64_t> const&,
            storm::storage::BitVector const&
        >(), py::arg("model"), py::arg("num_actions"), py::arg("choice_to_action"), py::arg("choice_mask"))
        .def_readonly("smg", &synthesis::SmgAbstraction<double>::smg)
        .def_readonly("state_to_quotient_state_action", &synthesis::SmgAbstraction<double>::state_to_quotient_state_action)
        .def_readonly("choice_to_quotient_choice", &synthesis::SmgAbstraction<double>::choice_to_quotient_choice)
        ;
}
