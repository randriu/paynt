#include "src/synthesis/synthesis.h"

#include "Counterexample.h"
#include "CounterexampleMdp.h"

void bindings_counterexamples(py::module& m) {

    py::classh<synthesis::CounterexampleGenerator<>>(m, "CounterexampleGenerator", "Counterexample generation")
        .def(
            py::init<
                storm::models::sparse::Mdp<double> const&, uint64_t,
                std::vector<std::set<uint64_t>> const&,
                std::vector<std::shared_ptr<storm::logic::Formula const>> const&
            >(),
            py::arg("quotient_mdp"), py::arg("hole_count"), py::arg("mdp_holes"), py::arg("formulae")
        )
        .def("prepare_dtmc", &synthesis::CounterexampleGenerator<>::prepareDtmc, py::arg("dtmc"), py::arg("quotient_state_map"))
        .def(
            "construct_conflict", &synthesis::CounterexampleGenerator<>::constructConflict,
            py::arg("formula_index"), py::arg("formula_bound"), py::arg("mdp_bounds"), py::arg("mdp_quotient_state_map")
        )
        .def("print_profiling", &synthesis::CounterexampleGenerator<>::printProfiling)
        ;


    // MDP Counterexample generation
    py::classh<synthesis::CounterexampleGeneratorMdp<>>(m, "CounterexampleGeneratorMdp", "Counterexample generation")
        .def(
            py::init<
                storm::models::sparse::Mdp<double> const&, uint64_t,
                std::vector<std::set<uint64_t>> const&,
                std::vector<std::shared_ptr<storm::logic::Formula const>> const&
            >(),
            py::arg("quotient_mdp"), py::arg("hole_count"), py::arg("mdp_holes"), py::arg("formulae")
        )
        .def("prepare_mdp", &synthesis::CounterexampleGeneratorMdp<>::prepareMdp)
        .def("construct_conflict", &synthesis::CounterexampleGeneratorMdp<>::constructConflict)
        ;
}
