#include "src/synthesis/synthesis.h"

#include "Posmg.h"
#include "PosmgManager.h"

#include <storm/models/sparse/Smg.h>

template <typename ValueType>
void bindings_posmg_vt(py::module &m, std::string const& vtSuffix) {
    py::classh<synthesis::Posmg<ValueType>, storm::models::sparse::Smg<ValueType>>(m, ("Posmg" + vtSuffix).c_str())
    //    .def(py::init<storm::storage::sparse::ModelComponents<double> const&>(), py::arg("components"))
    //    .def(py::init<storm::storage::sparse::ModelComponents<double> &&>(), py::arg("components"));
        .def(("get_observations" + vtSuffix).c_str(), &synthesis::Posmg<ValueType>::getObservations)
        .def(("get_p0_observation_count" + vtSuffix).c_str(), &synthesis::Posmg<ValueType>::getP0ObservationCount)
        .def(("get_mdp" + vtSuffix).c_str(), &synthesis::Posmg<ValueType>::getMdp)
        .def_property_readonly(("nondeterministic_choice_indices" + vtSuffix).c_str(), [](synthesis::Posmg<ValueType> const& m) { return m.getNondeterministicChoiceIndices(); })
        .def(("get_pomdp" + vtSuffix).c_str(), &synthesis::Posmg<ValueType>::getPomdp)
        // this binding (calculation) is done in stormpy for mdp, but posmg doesn't inherit from mdp, so it is also copied here
        .def(("get_nr_available_actions" + vtSuffix).c_str(), [](synthesis::Posmg<ValueType> const& posmg, uint64_t stateIndex)
            { return posmg.getNondeterministicChoiceIndices()[stateIndex+1] - posmg.getNondeterministicChoiceIndices()[stateIndex] ; },
             py::arg("state"))
        // same as ^
        .def(("get_choice_index" + vtSuffix).c_str(), [](synthesis::Posmg<ValueType> const& posmg, uint64_t state, uint64_t actOff)
            { return posmg.getNondeterministicChoiceIndices()[state]+actOff; },
             py::arg("state"), py::arg("action_offset"), "gets the choice index for the offset action from the given state.");


    m.def(("posmg_from_pomdp" + vtSuffix).c_str(), &synthesis::posmgFromPomdp<ValueType>, py::arg("pomdp"), py::arg("state_player_indications"));
    m.def(("posmg_from_smg" + vtSuffix).c_str(), &synthesis::posmgFromSmg<ValueType>, py::arg("smg"), py::arg("observability_classes"));

    py::classh<synthesis::PosmgManager<ValueType>>(m, ("PosmgManager" + vtSuffix).c_str())
        .def(py::init<synthesis::Posmg<ValueType> const&, uint64_t>(), py::arg("posmg"), py::arg("optimizing_player"))
        .def(("construct_mdp" + vtSuffix).c_str(), &synthesis::PosmgManager<ValueType>::constructMdp)
        .def(("get_observation_mapping" + vtSuffix).c_str(), &synthesis::PosmgManager<ValueType>::getObservationMapping)
        .def(("set_observation_memory_size" + vtSuffix).c_str(), &synthesis::PosmgManager<ValueType>::setObservationMemorySize,
            py::arg("observation"), py::arg("memory_size"))
        .def(("get_state_player_indications" + vtSuffix).c_str(), &synthesis::PosmgManager<ValueType>::getStatePlayerIndications)
        .def(("get_action_count" + vtSuffix).c_str(), &synthesis::PosmgManager<ValueType>::getActionCount, py::arg("state"))
        .def_property_readonly(("state_prototype" + vtSuffix).c_str(), [](synthesis::PosmgManager<ValueType>& manager) {return manager.statePrototype;})
        .def_property_readonly(("state_memory" + vtSuffix).c_str(), [](synthesis::PosmgManager<ValueType>& manager) {return manager.stateMemory;})
        .def_property_readonly(("observation_memory_size" + vtSuffix).c_str(), [](synthesis::PosmgManager<ValueType>& manager) {return manager.optPlayerObservationMemorySize;})
        .def_property_readonly(("observation_actions" + vtSuffix).c_str(), [](synthesis::PosmgManager<ValueType>& manager) {return manager.optPlayerObservationActions;})
        .def_property_readonly(("observation_successors" + vtSuffix).c_str(), [](synthesis::PosmgManager<ValueType>& manager) {return manager.succesors;})
        .def_property_readonly(("max_successor_memory_size" + vtSuffix).c_str(), [](synthesis::PosmgManager<ValueType>& manager) {return manager.maxSuccesorDuplicateCount;})
        .def_property_readonly(("num_holes" + vtSuffix).c_str(), [](synthesis::PosmgManager<ValueType>& manager) {return manager.holeCount;})
        .def_property_readonly(("action_holes" + vtSuffix).c_str(), [](synthesis::PosmgManager<ValueType>& manager) {return manager.actionHoles;})
        .def_property_readonly(("memory_holes" + vtSuffix).c_str(), [](synthesis::PosmgManager<ValueType>& manager) {return manager.memoryHoles;})
        .def_property_readonly(("hole_options" + vtSuffix).c_str(), [](synthesis::PosmgManager<ValueType>& manager) {return manager.holeOptionCount;})
        .def_property_readonly(("row_action_hole" + vtSuffix).c_str(), [](synthesis::PosmgManager<ValueType>& manager) {return manager.rowActionHole;})
        .def_property_readonly(("row_action_option" + vtSuffix).c_str(), [](synthesis::PosmgManager<ValueType>& manager) {return manager.rowActionOption;})
        .def_property_readonly(("row_memory_hole" + vtSuffix).c_str(), [](synthesis::PosmgManager<ValueType>& manager) {return manager.rowMemoryHole;})
        .def_property_readonly(("row_memory_option" + vtSuffix).c_str(), [](synthesis::PosmgManager<ValueType>& manager) {return manager.rowMemoryOption;})
        ;
}

void bindings_posmg(py::module& m) {
    bindings_posmg_vt<double>(m, "");
    bindings_posmg_vt<storm::RationalNumber>(m, "Exact");
}