#include "../synthesis.h"

#include "storm/storage/Scheduler.h"

namespace synthesis
{
    storm::storage::Scheduler<double> createScheduler(uint_fast64_t numberOfModelStates) {
        return storm::storage::Scheduler<double>(numberOfModelStates);
    }

    void setDontCareStateForScheduler(storm::storage::Scheduler<double>& scheduler, uint_fast64_t modelState, uint_fast64_t memoryState, bool setArbitraryChoice) {
        scheduler.setDontCare(modelState, memoryState, setArbitraryChoice);
    }


}

void bindings_storage(py::module& m) {
    
    m.def("create_scheduler", &synthesis::createScheduler, py::arg("number_of_model_states"));
    m.def("set_dont_care_state_for_scheduler", &synthesis::setDontCareStateForScheduler, py::arg("scheduler"), py::arg("model_state"), py::arg("memory_state"), py::arg("set_arbitrary_choice"));
}