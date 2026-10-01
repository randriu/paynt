#pragma once

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

namespace synthesis {

/**
 * Set trie (Savnik, "Index data structure for fast subset and superset queries", 2013) of sets of non-negative
 * integers, each stored under an integer payload.
 *
 * It answers the two questions that the cache of the SMPMC engine (paynt/synthesizer/smpmc/cache.py) asks about the
 * partial parameter assignments it has seen -- which of the stored sets are contained in a query set, and which contain
 * it -- without looking at every stored set. A set is given as a list of elements; the order of the list and the
 * repetitions in it do not matter, and the empty set is a set like any other.
 *
 * The elements of a set are kept in ascending order along a path of the trie from the root, so sets with a common
 * prefix share it. Every query is a depth-first search that prunes the branches which cannot lead to a match, and
 * returns the payloads in the order of the stored sets compared lexicographically in ascending order of their elements,
 * a set before its extensions. The queries use no recursion, so the size of a set is not limited by the stack.
 */
class SetTrie {
   public:
    using Element = uint64_t;
    using Payload = int64_t;
    using Elements = std::vector<Element>;

    SetTrie();

    /** Stores a set under a payload. Throws std::invalid_argument if the payload is stored already. */
    void insert(Elements elements, Payload payload);

    /** Removes the set stored under a payload and prunes the branch that only led to it. Returns false if there is
     * none. */
    bool remove(Payload payload);

    /** Whether exactly this set is stored. */
    bool contains(Elements elements) const;

    /** The payloads of the stored sets that are subsets of the given set (this one included). */
    std::vector<Payload> subsets(Elements elements) const;

    /** The payloads of the stored sets that are supersets of the given set (this one included). */
    std::vector<Payload> supersets(Elements elements) const;

    /** Whether some stored set is a subset of the given set. Stops at the first one. */
    bool hasSubset(Elements elements) const;

    /** Whether some stored set is a superset of the given set. Stops at the first one. */
    bool hasSuperset(Elements elements) const;

    /** The number of sets stored. */
    size_t size() const;

   private:
    static constexpr uint32_t ROOT = 0;

    struct Node {
        /** The children, in ascending order of the element that leads to them (the index of the node in nodes). */
        std::vector<std::pair<Element, uint32_t>> children;
        /** The payloads of the sets that end in this node. */
        std::vector<Payload> payloads;
    };

    /** Sorts the elements and drops the repetitions. */
    static void normalize(Elements& elements);

    /** The position in the children of a node of the first child whose element is not below the given one. */
    static size_t lowerBound(Node const& node, Element element);

    uint32_t newNode();
    void releaseNode(uint32_t node);

    /** Walks the stored sets that are subsets of the (normalized) set. Collects their payloads in out, or just looks
     * for the first match if out is null. */
    bool visitSubsets(Elements const& elements, std::vector<Payload>* out) const;
    /** The same for the supersets. */
    bool visitSupersets(Elements const& elements, std::vector<Payload>* out) const;

    /** The nodes. Pruned nodes are reused, which is why they are in a vector and refer to each other by index. */
    std::vector<Node> nodes;
    std::vector<uint32_t> free_nodes;
    /** The set of every payload stored, to find its node again when the payload is removed. */
    std::unordered_map<Payload, Elements> payload_to_set;
};

}  // namespace synthesis
