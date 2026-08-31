#include "src/synthesis/synthesis.h"

#include "DecPomdp.h"

void bindings_decpomdp(py::module& m) {

    py::classh<synthesis::DecPomdp>(m, "DecPomdp", "dec-POMDP")
        // .def(py::init<std::string const&>(), "constructor.", py::arg("filename"));
        .def("construct_quotient_mdp", &synthesis::DecPomdp::constructQuotientMdp)
        .def("construct_mdp", &synthesis::DecPomdp::constructMdp)
        .def("construct_pomdp", &synthesis::DecPomdp::constructPomdp)
        .def("set_agent_observation_memory_size", &synthesis::DecPomdp::setAgentObservationMemorySize, "Set memory size to a selected agent and its observation.", py::arg("agent"), py::arg("observation"), py::arg("memory_size"))
        .def("set_global_memory_size", &synthesis::DecPomdp::setGlobalMemorySize, py::arg("memory_size"))
        .def_property_readonly("num_agents", [](synthesis::DecPomdp& decpomdp) {return decpomdp.num_agents;})
        .def_property_readonly("num_quotient_rows", [](synthesis::DecPomdp& decpomdp) {return decpomdp.num_quotient_rows;})
        .def_property_readonly("joint_observations", [](synthesis::DecPomdp& decpomdp) {return decpomdp.joint_observations;})
        .def_property_readonly("agent_observation_labels", [](synthesis::DecPomdp& decpomdp) {return decpomdp.agent_observation_labels;})
        .def_property_readonly("num_agent_actions_at_observation", [](synthesis::DecPomdp& decpomdp) {return decpomdp.num_agent_actions_at_observation;})
        .def("num_decpomdp_states", &synthesis::DecPomdp::num_states)
        
        .def_property_readonly("reward_model_name", [](synthesis::DecPomdp& decpomdp) {return decpomdp.reward_model_name;})
        .def_property_readonly("reward_minimizing", [](synthesis::DecPomdp& decpomdp) {return decpomdp.reward_minimizing;})
        .def_property_readonly("discount_factor", [](synthesis::DecPomdp& decpomdp) {return decpomdp.discount_factor;})

        .def("set_constraint", &synthesis::DecPomdp::set_constraint_bound, py::arg("bound"))
        .def_property_readonly("discounted", [](synthesis::DecPomdp& decpomdp) {return decpomdp.discounted;})
        
        .def("apply_discount_factor_transformation", &synthesis::DecPomdp::applyDiscountFactorTransformation)
        .def_property_readonly("discount_sink_label", [](synthesis::DecPomdp& decpomdp) {return decpomdp.discount_sink_label;})

        .def_property_readonly("row_joint_action", [](synthesis::DecPomdp& decpomdp) {return decpomdp.row_joint_action;}, "row_joint_action")
        .def_property_readonly("state_joint_observation", [](synthesis::DecPomdp& decpomdp) {return decpomdp.state_joint_observation;},"state_joint_observation")
        .def_property_readonly("joint_actions", [](synthesis::DecPomdp& decpomdp) {return decpomdp.joint_actions;})
        .def_property_readonly("agent_action_labels", [](synthesis::DecPomdp& decpomdp) {return decpomdp.agent_action_labels;})
        .def_property_readonly("max_successor_memory_size", [](synthesis::DecPomdp& decpomdp) {return decpomdp.max_successor_memory_size;})
        .def_property_readonly("agent_max_successor_memory_size", [](synthesis::DecPomdp& decpomdp) {return decpomdp.agent_max_successor_memory_size;})

        .def_property_readonly("agent_row_action_hole", [](synthesis::DecPomdp& decpomdp) {return decpomdp.agent_row_action_hole;})
        .def_property_readonly("agent_row_action_option", [](synthesis::DecPomdp& decpomdp) {return decpomdp.agent_row_action_option;})
        .def_property_readonly("agent_row_memory_hole", [](synthesis::DecPomdp& decpomdp) {return decpomdp.agent_row_memory_hole;})
        .def_property_readonly("agent_row_memory_option", [](synthesis::DecPomdp& decpomdp) {return decpomdp.agent_row_memory_option;})
        .def_property_readonly("num_holes", [](synthesis::DecPomdp& decpomdp) {return decpomdp.num_holes;})
        .def_property_readonly("transition_matrix", [](synthesis::DecPomdp& decpomdp) {return decpomdp.transition_matrix;}, "transition matrix")

        .def_property_readonly("memory_joint_observation", [](synthesis::DecPomdp& decpomdp) {return decpomdp.memory_joint_observation;})
        .def_property_readonly("action_to_memory_joint_observation", [](synthesis::DecPomdp& decpomdp) {return decpomdp.action_to_memory_joint_observation;})
        .def_property_readonly("state_to_memory_joint_observation", [](synthesis::DecPomdp& decpomdp) {return decpomdp.state_to_memory_joint_observation;})
        ;

    m.def("parse_decpomdp", &synthesis::parseDecPomdp,  py::arg("filename"));

}
