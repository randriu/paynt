"""Tests of payntbind.synthesis.SetTrie, the set trie behind the cache of the SMPMC engine (paynt/synthesizer/smpmc/cache.py).

Its answers are checked by hand on small examples, and against a brute-force reference (Python sets, and a loop over everything stored) on random operations.
"""

from __future__ import annotations

import random

import pytest
from payntbind.synthesis import SetTrie


def expected_payloads(stored, query, related):
    """The payloads of the stored sets that are related to the query, in the order that SetTrie promises: the stored sets compared lexicographically in
    ascending order of their elements (a set before its extensions), and the payloads of one set in the order they were stored in.

    :param stored: payload -> (the set, the number of its insertion)
    :param related: whether a stored set is a subset (or a superset) of the query
    """
    matching = [(sorted(elements), number, payload) for payload, (elements, number) in stored.items() if related(elements, query)]
    return [payload for _, _, payload in sorted(matching)]


class TestBasics:
    def test_an_empty_trie_matches_nothing(self):
        trie = SetTrie()
        assert len(trie) == 0
        assert trie.subsets([1, 2]) == [] and trie.supersets([1, 2]) == []
        assert trie.subsets([]) == [] and trie.supersets([]) == []
        assert not trie.has_subset([1]) and not trie.has_superset([1]) and not trie.has_superset([])
        assert not trie.contains([1]) and not trie.contains([])

    def test_a_set_is_a_subset_and_a_superset_of_itself(self):
        trie = SetTrie()
        trie.insert([3, 1, 2], 7)
        assert trie.subsets([1, 2, 3]) == [7]
        assert trie.supersets([1, 2, 3]) == [7]
        assert trie.has_subset([1, 2, 3]) and trie.has_superset([1, 2, 3])
        assert trie.contains([1, 2, 3])

    def test_the_stored_sets_contained_in_a_set_and_those_that_contain_it(self):
        trie = SetTrie()
        for elements, payload in [([1], 10), ([1, 2], 20), ([1, 2, 3], 30), ([2, 3], 40)]:
            trie.insert(elements, payload)
        assert len(trie) == 4
        assert trie.subsets([1, 2]) == [10, 20]
        assert trie.subsets([1, 2, 3]) == [10, 20, 30, 40]
        assert trie.subsets([3]) == []
        assert trie.supersets([2]) == [20, 30, 40]
        assert trie.supersets([1, 3]) == [30]
        assert trie.supersets([4]) == []

    def test_the_order_and_the_repetitions_of_the_elements_do_not_matter(self):
        trie = SetTrie()
        trie.insert([3, 1, 3, 2, 1], 1)
        assert trie.contains([2, 3, 1])
        assert trie.subsets([2, 2, 1, 3, 9]) == [1]
        assert trie.supersets([3, 3]) == [1]
        assert len(trie) == 1

    def test_the_empty_set_is_a_set_like_any_other(self):
        trie = SetTrie()
        trie.insert([], 0)
        assert trie.contains([]) and not trie.contains([1])
        assert trie.subsets([]) == [0] and trie.subsets([1, 2]) == [0]
        assert trie.supersets([]) == [0] and trie.supersets([1]) == []
        assert trie.has_subset([5]) and trie.has_superset([])
        trie.insert([1], 1)
        assert trie.subsets([1]) == [0, 1]
        assert trie.supersets([]) == [0, 1]
        assert trie.remove(0)
        assert trie.subsets([1]) == [1] and not trie.contains([])

    def test_a_payload_may_be_zero(self):
        trie = SetTrie()
        trie.insert([1], 0)
        assert trie.subsets([1, 2]) == [0] and trie.supersets([1]) == [0]
        assert trie.has_subset([1, 2]) and trie.has_superset([1])
        assert trie.remove(0)
        assert len(trie) == 0 and not trie.has_subset([1, 2])

    def test_a_payload_may_be_negative_or_large(self):
        trie = SetTrie()
        trie.insert([1], -7)
        trie.insert([1, 2], 2**62)
        trie.insert([1, 2, 3], -(2**62))
        assert trie.subsets([1, 2, 3]) == [-7, 2**62, -(2**62)]
        assert trie.remove(2**62)
        assert trie.subsets([1, 2, 3]) == [-7, -(2**62)]

    def test_an_element_may_use_all_sixty_four_bits(self):
        trie = SetTrie()
        trie.insert([2**64 - 1], 1)
        trie.insert([(7 << 32) | 3], 2)
        assert trie.subsets([2**64 - 1, (7 << 32) | 3]) == [2, 1]
        assert trie.supersets([2**64 - 1]) == [1]
        assert trie.subsets([7 << 32, 3]) == []

    def test_several_payloads_may_share_a_set(self):
        trie = SetTrie()
        trie.insert([1, 2], 10)
        trie.insert([2, 1], 11)
        trie.insert([1, 2], 12)
        assert len(trie) == 3
        assert trie.subsets([1, 2]) == [10, 11, 12]
        assert trie.remove(11)
        assert trie.subsets([1, 2]) == [10, 12] and trie.contains([1, 2])
        assert trie.remove(10) and trie.remove(12)
        assert not trie.contains([1, 2]) and trie.subsets([1, 2]) == []


class TestErrors:
    def test_a_payload_cannot_be_stored_twice(self):
        trie = SetTrie()
        trie.insert([1], 5)
        with pytest.raises(ValueError, match="stored already"):
            trie.insert([2], 5)
        assert len(trie) == 1
        assert trie.subsets([1, 2]) == [5]

    def test_an_element_cannot_be_negative(self):
        trie = SetTrie()
        with pytest.raises(TypeError):
            trie.insert([-1], 1)
        with pytest.raises(TypeError):
            trie.subsets([1, -1])
        assert len(trie) == 0

    def test_removing_what_is_not_stored(self):
        trie = SetTrie()
        assert not trie.remove(5)
        trie.insert([1], 5)
        assert trie.remove(5)
        assert not trie.remove(5)


class TestRemove:
    def test_removing_a_set_keeps_the_others(self):
        trie = SetTrie()
        trie.insert([1, 2], 1)
        trie.insert([1, 3], 2)
        trie.insert([1], 3)
        assert trie.remove(1)
        assert trie.subsets([1, 2, 3]) == [3, 2]
        assert not trie.contains([1, 2]) and trie.contains([1, 3]) and trie.contains([1])
        assert len(trie) == 2

    def test_a_branch_that_was_pruned_can_be_stored_again(self):
        trie = SetTrie()
        for number in range(3):
            trie.insert([1 + number, 5, 9], number)
            trie.insert([1 + number, 5], 10 + number)
            assert trie.subsets([1 + number, 5, 9]) == [10 + number, number]
            assert trie.remove(number) and trie.remove(10 + number)
            assert len(trie) == 0
            assert trie.subsets([1, 2, 3, 5, 9]) == [] and not trie.has_superset([])

    def test_removing_a_set_leaves_its_prefixes_and_its_extensions(self):
        trie = SetTrie()
        trie.insert([1, 2], 1)
        trie.insert([1, 2, 3], 2)
        trie.insert([1], 3)
        assert trie.remove(1)
        assert trie.subsets([1, 2, 3]) == [3, 2]
        assert trie.supersets([2, 3]) == [2]
        assert trie.remove(2)
        assert trie.subsets([1, 2, 3]) == [3]
        assert trie.remove(3)
        assert len(trie) == 0 and not trie.has_subset([1, 2, 3])


class TestOrder:
    SETS = {1: [3, 4], 2: [1, 9], 3: [1], 4: [1, 2, 3], 5: [], 6: [2]}

    def test_the_payloads_come_back_in_the_order_of_the_sets_not_of_the_insertions(self):
        rng = random.Random(0)
        for _ in range(30):
            items = list(self.SETS.items())
            rng.shuffle(items)
            trie = SetTrie()
            for payload, elements in items:
                trie.insert(rng.sample(elements, len(elements)), payload)
            assert trie.subsets([1, 2, 3, 4, 9]) == [5, 3, 4, 2, 6, 1]
            assert trie.supersets([]) == [5, 3, 4, 2, 6, 1]


class TestAgainstPythonSets:
    @pytest.mark.parametrize("seed", range(8))
    def test_random_operations_agree_with_a_loop_over_python_sets(self, seed):
        rng = random.Random(seed)
        universe = [4, 8, 14][seed % 3]
        trie = SetTrie()
        stored = {}
        inserted = 0

        def random_elements():
            elements = rng.sample(range(universe), rng.randint(0, universe))
            return elements + elements[: rng.randint(0, 2)]

        for _step in range(400):
            action = rng.random()
            if action < 0.5:
                elements = random_elements()
                trie.insert(elements, inserted)
                stored[inserted] = (frozenset(elements), inserted)
                inserted += 1
            elif action < 0.7 and stored:
                payload = rng.choice(sorted(stored))
                assert trie.remove(payload)
                del stored[payload]

            query = random_elements()
            query_set = frozenset(query)
            subsets = expected_payloads(stored, query_set, frozenset.issubset)
            supersets = expected_payloads(stored, query_set, frozenset.issuperset)
            assert trie.subsets(query) == subsets
            assert trie.supersets(query) == supersets
            assert trie.has_subset(query) == bool(subsets)
            assert trie.has_superset(query) == bool(supersets)
            assert trie.contains(query) == any(elements == query_set for elements, _ in stored.values())
            assert len(trie) == len(stored)


class TestLargeSets:
    def test_a_set_of_a_hundred_thousand_elements(self):
        """No recursion is deep, so the size of a set is not limited by the stack, when a query, a removal or the destruction of the trie walks along it."""
        big = list(range(100_000))
        trie = SetTrie()
        trie.insert(big, 1)
        trie.insert(big[:50_000], 2)
        assert trie.subsets(big) == [2, 1]
        assert trie.supersets(big[:10]) == [2, 1]
        assert trie.has_subset(big) and trie.has_superset(big) and trie.contains(big)
        assert trie.remove(1)
        assert trie.subsets(big) == [2]
        assert trie.remove(2)
        assert len(trie) == 0
