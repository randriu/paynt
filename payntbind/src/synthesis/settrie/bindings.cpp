#include "src/synthesis/synthesis.h"

#include "SetTrie.h"

void bindings_settrie(py::module& m) {
    py::classh<synthesis::SetTrie>(m, "SetTrie", R"doc(
Set trie of sets of non-negative integers, each stored under an integer payload.

Answers which of the stored sets are subsets, and which are supersets, of a query set. A set is a list of integers: the order and the repetitions do not
matter, and the empty set is a set like any other. The payloads of a query come back in the order of the stored sets compared lexicographically.
)doc")
        .def(py::init<>())
        .def("insert", &synthesis::SetTrie::insert, py::arg("elements"), py::arg("payload"),
             "Store a set under a payload. Raises ValueError if the payload is stored already.")
        .def("remove", &synthesis::SetTrie::remove, py::arg("payload"),
             "Remove the set stored under a payload. Returns False if there is none.")
        .def("contains", &synthesis::SetTrie::contains, py::arg("elements"), "Whether exactly this set is stored.")
        .def("subsets", &synthesis::SetTrie::subsets, py::arg("elements"),
             "The payloads of the stored sets that are subsets of this one (this one included).")
        .def("supersets", &synthesis::SetTrie::supersets, py::arg("elements"),
             "The payloads of the stored sets that are supersets of this one (this one included).")
        .def("has_subset", &synthesis::SetTrie::hasSubset, py::arg("elements"),
             "Whether some stored set is a subset of this one.")
        .def("has_superset", &synthesis::SetTrie::hasSuperset, py::arg("elements"),
             "Whether some stored set is a superset of this one.")
        .def("__len__", &synthesis::SetTrie::size);
}
