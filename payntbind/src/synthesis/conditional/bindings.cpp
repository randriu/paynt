#include "src/synthesis/synthesis.h"

#include "Conditional.h"

#include <storm/adapters/RationalNumberAdapter.h>

template<typename ValueType>
void bindings_conditional_vt(py::module& m) {
    m.def("some_scheduler_reaches", &synthesis::someSchedulerReaches<ValueType>, py::arg("model"), py::arg("states"));
    m.def("some_scheduler_avoids", &synthesis::someSchedulerAvoids<ValueType>, py::arg("model"), py::arg("states"));
    m.def("conditional_scheduler_choices", &synthesis::conditionalSchedulerChoices<ValueType>,
        py::arg("mdp"), py::arg("scheduler"), py::arg("target"), py::arg("condition"), py::arg("choice_map"));
    m.def("dtmc_as_mdp", &synthesis::dtmcAsMdp<ValueType>, py::arg("dtmc"));
    m.def("unfold_conditional_phases", &synthesis::unfoldConditionalPhases<ValueType>, py::arg("mdp"), py::arg("target"), py::arg("condition"));
}

void bindings_conditional(py::module& m) {
    bindings_conditional_vt<double>(m);
    bindings_conditional_vt<storm::RationalNumber>(m);
}
