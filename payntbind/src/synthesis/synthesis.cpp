#include "synthesis.h"

#include <storm/utility/initialize.h>
#include <storm/settings/SettingsManager.h>

void define_synthesis(py::module& m) {

    // Storm's settings singleton must be initialized before any bindings below are registered:
    // some of them construct storm::Environment (e.g. as a pybind default argument), which is
    // evaluated eagerly at import time and transitively queries the "general" settings module.
    if (!storm::settings::manager().hasModule("general")) {
        storm::settings::initializeAll("payntbind", "payntbind");
    }

    m.def("_set_up", [](std::string const& args) {
            storm::settings::mutableManager().setFromString(args);
        }, "Initialize Storm", py::arg("arguments"));

    define_helpers(m);

    bindings_translation(m);
    bindings_pomdp(m);
    bindings_decpomdp(m);
    bindings_counterexamples(m);
    bindings_pomdp_family(m);
    bindings_storage(m);
    bindings_mdp_family(m);

    bindings_coloring(m);
    bindings_settrie(m);

    #ifndef DISABLE_SMG
    bindings_smg(m);
    bindings_posmg(m);
    #endif // DISABLE_SMG
}