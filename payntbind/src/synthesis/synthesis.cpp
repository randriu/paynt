#include "synthesis.h"
#include "storm/utility/initialize.h"
#include "storm/settings/SettingsManager.h"

void define_synthesis(py::module& m) {

    m.def("_set_up", [](std::string const& args) {
            if (!storm::settings::manager().hasModule("general")) {
                storm::settings::initializeAll("payntbind", "payntbind");
                storm::settings::mutableManager().setFromString(args);
            }
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

    #ifndef DISABLE_SMG
    bindings_smg(m);
    bindings_posmg(m);
    #endif // DISABLE_SMG
}