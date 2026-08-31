#include "src/synthesis/synthesis.h"

#include "MemoryUnfolder.h"
#include <storm/adapters/RationalNumberAdapter.h>

template<typename ValueType>
void bindings_mdp_family_vt(py::module& m, std::string const& vtSuffix) {

    py::classh<synthesis::MemoryUnfolder<ValueType>>(m, (vtSuffix + "MemoryUnfolder").c_str(), "Memory Unfolder class")
        .def(py::init<storm::models::sparse::Mdp<ValueType> const&>(), "Constructor.", py::arg("mdp"))
        .def("construct_unfolded_model", &synthesis::MemoryUnfolder<ValueType>::constructUnfoldedModel,
            "Construct an unfolded MDP from a given MDP and memory size.",
            py::arg("memory_size")
        )
        .def_property_readonly("state_prototype", [](synthesis::MemoryUnfolder<ValueType>& unfolder) {return unfolder.statePrototype;})
        .def_property_readonly("state_memory", [](synthesis::MemoryUnfolder<ValueType>& unfolder) {return unfolder.stateMemory;})
        .def_property_readonly("choice_map", [](synthesis::MemoryUnfolder<ValueType>& unfolder) {return unfolder.choiceMap;})
        ;

}


void bindings_mdp_family(py::module& m) {
    bindings_mdp_family_vt<double>(m, "");
    bindings_mdp_family_vt<storm::RationalNumber>(m, "Exact");
}
