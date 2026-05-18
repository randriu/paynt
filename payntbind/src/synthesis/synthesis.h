#pragma once

#include "src/common.h"

void define_synthesis(py::module& m);
void define_helpers(py::module &m);

void bindings_translation(py::module &m);
void bindings_pomdp(py::module &m);
void bindings_decpomdp(py::module &m);
void bindings_counterexamples(py::module &m);
void bindings_pomdp_family(py::module &m);
void bindings_storage(py::module &m);
void bindings_mdp_family(py::module &m);

void bindings_coloring(py::module &m);

void bindings_smg(py::module &m);
void bindings_posmg(py::module &m);
