#include "SetTrie.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace synthesis {

SetTrie::SetTrie() {
    nodes.emplace_back();  // the root
}

void SetTrie::normalize(Elements& elements) {
    std::sort(elements.begin(), elements.end());
    elements.erase(std::unique(elements.begin(), elements.end()), elements.end());
}

size_t SetTrie::lowerBound(Node const& node, Element element) {
    auto const& children = node.children;
    auto it = std::lower_bound(children.begin(), children.end(), element,
                               [](std::pair<Element, uint32_t> const& child, Element e) { return child.first < e; });
    return it - children.begin();
}

uint32_t SetTrie::newNode() {
    if (!free_nodes.empty()) {
        uint32_t node = free_nodes.back();
        free_nodes.pop_back();
        return node;
    }
    if (nodes.size() >= std::numeric_limits<uint32_t>::max()) {
        throw std::length_error("the set trie has too many nodes");
    }
    nodes.emplace_back();
    return static_cast<uint32_t>(nodes.size() - 1);
}

void SetTrie::releaseNode(uint32_t node) {
    nodes[node].children.clear();
    nodes[node].payloads.clear();
    free_nodes.push_back(node);
}

void SetTrie::insert(Elements elements, Payload payload) {
    if (payload_to_set.count(payload) > 0) {
        throw std::invalid_argument("the payload " + std::to_string(payload) + " is stored already");
    }
    normalize(elements);
    uint32_t node = ROOT;
    for (Element element : elements) {
        size_t position = lowerBound(nodes[node], element);
        auto const& children = nodes[node].children;
        if (position < children.size() && children[position].first == element) {
            node = children[position].second;
            continue;
        }
        uint32_t child = newNode();  // may move the nodes, so the children are looked up again
        auto& siblings = nodes[node].children;
        siblings.insert(siblings.begin() + position, {element, child});
        node = child;
    }
    nodes[node].payloads.push_back(payload);
    payload_to_set.emplace(payload, std::move(elements));
}

bool SetTrie::remove(Payload payload) {
    auto stored = payload_to_set.find(payload);
    if (stored == payload_to_set.end()) {
        return false;
    }

    // the path from the root to the node of the set: for every step, the node that it leaves and the position of the
    // child in it
    std::vector<std::pair<uint32_t, size_t>> path;
    path.reserve(stored->second.size());
    uint32_t node = ROOT;
    for (Element element : stored->second) {
        size_t position = lowerBound(nodes[node], element);
        path.emplace_back(node, position);
        node = nodes[node].children[position].second;
    }
    auto& payloads = nodes[node].payloads;
    payloads.erase(std::remove(payloads.begin(), payloads.end(), payload), payloads.end());

    // prune the nodes that no set goes through any more
    while (!path.empty() && nodes[node].payloads.empty() && nodes[node].children.empty()) {
        auto [parent, position] = path.back();
        path.pop_back();
        auto& siblings = nodes[parent].children;
        siblings.erase(siblings.begin() + position);
        releaseNode(node);
        node = parent;
    }
    payload_to_set.erase(stored);
    return true;
}

bool SetTrie::contains(Elements elements) const {
    normalize(elements);
    uint32_t node = ROOT;
    for (Element element : elements) {
        auto const& children = nodes[node].children;
        size_t position = lowerBound(nodes[node], element);
        if (position == children.size() || children[position].first != element) {
            return false;
        }
        node = children[position].second;
    }
    return !nodes[node].payloads.empty();
}

bool SetTrie::visitSubsets(Elements const& elements, std::vector<Payload>* out) const {
    // the nodes still to visit, each with the position in the elements of the first one that the sets below it may
    // still contain
    std::vector<std::pair<uint32_t, size_t>> stack;
    stack.emplace_back(ROOT, 0);
    while (!stack.empty()) {
        auto [index, from] = stack.back();
        stack.pop_back();
        Node const& node = nodes[index];
        if (!node.payloads.empty()) {
            if (out == nullptr) {
                return true;
            }
            out->insert(out->end(), node.payloads.begin(), node.payloads.end());
        }
        // The children that lead on with an element of the query, pushed in descending order of the element so that
        // they are visited in ascending order. The cheaper of two ways to find them: look up every child among the
        // elements, or every element among the children.
        size_t remaining = elements.size() - from;
        if (node.children.size() < remaining) {
            auto const first = elements.begin() + from;
            for (size_t k = node.children.size(); k-- > 0;) {
                auto const& [element, child] = node.children[k];
                auto it = std::lower_bound(first, elements.end(), element);
                if (it != elements.end() && *it == element) {
                    stack.emplace_back(child, static_cast<size_t>(it - elements.begin()) + 1);
                }
            }
        } else {
            for (size_t j = elements.size(); j-- > from;) {
                size_t position = lowerBound(node, elements[j]);
                if (position < node.children.size() && node.children[position].first == elements[j]) {
                    stack.emplace_back(node.children[position].second, j + 1);
                }
            }
        }
    }
    return false;
}

bool SetTrie::visitSupersets(Elements const& elements, std::vector<Payload>* out) const {
    // the nodes still to visit, each with the number of the elements that the sets below it have taken so far
    std::vector<std::pair<uint32_t, size_t>> stack;
    stack.emplace_back(ROOT, 0);
    while (!stack.empty()) {
        auto [index, taken] = stack.back();
        stack.pop_back();
        Node const& node = nodes[index];
        auto const& children = node.children;

        if (taken == elements.size()) {
            // the sets below this node contain all the elements, so they all match
            if (!node.payloads.empty()) {
                if (out == nullptr) {
                    return true;
                }
                out->insert(out->end(), node.payloads.begin(), node.payloads.end());
            }
            for (size_t k = children.size(); k-- > 0;) {
                stack.emplace_back(children[k].second, taken);
            }
            continue;
        }

        // A set that ends here lacks the next element, and so does one that goes on with a larger element: only the
        // children up to the next element lead to supersets, and the one for the next element itself takes it.
        Element next = elements[taken];
        size_t last =
            std::upper_bound(children.begin(), children.end(), next,
                             [](Element e, std::pair<Element, uint32_t> const& child) { return e < child.first; }) -
            children.begin();
        for (size_t k = last; k-- > 0;) {
            stack.emplace_back(children[k].second, children[k].first == next ? taken + 1 : taken);
        }
    }
    return false;
}

std::vector<SetTrie::Payload> SetTrie::subsets(Elements elements) const {
    normalize(elements);
    std::vector<Payload> payloads;
    visitSubsets(elements, &payloads);
    return payloads;
}

std::vector<SetTrie::Payload> SetTrie::supersets(Elements elements) const {
    normalize(elements);
    std::vector<Payload> payloads;
    visitSupersets(elements, &payloads);
    return payloads;
}

bool SetTrie::hasSubset(Elements elements) const {
    normalize(elements);
    return visitSubsets(elements, nullptr);
}

bool SetTrie::hasSuperset(Elements elements) const {
    normalize(elements);
    return visitSupersets(elements, nullptr);
}

size_t SetTrie::size() const {
    return payload_to_set.size();
}

}  // namespace synthesis
