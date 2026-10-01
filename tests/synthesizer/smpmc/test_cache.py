"""Pure unit tests for PartialModelCache's set-trie-backed subset/superset subsumption -- no Storm, no ColoredMdp, just the cache's own {parameter: option} dict
contract.

See cache.py's module docstring for the subsumption rules these tests exercise.
"""

from __future__ import annotations

import subprocess
import sys

import payntbind.synthesis

from paynt.synthesizer.smpmc.cache import MISS, PartialModelCache, Refutation


def _conflicts(result):
    """The conflicts of the refutations a lookup returned."""
    assert isinstance(result, list), f"expected refutations, got {result}"
    return sorted(refutation.conflict_parameters for refutation in result)


class TestExactMatch:
    """Baseline behavior a subset/superset-aware cache must still get right: an unrelated query misses, and an identical query hits."""

    def test_lookup_on_empty_cache_misses(self):
        cache = PartialModelCache()
        assert cache.lookup({0: 1, 1: 2}, True, epoch=0) is MISS

    def test_exact_match_refuted_lookup_hits(self):
        cache = PartialModelCache()
        cache.insert_refuted({0: 1, 1: 2}, True, conflict_parameters=[0, 1], epoch=0)
        assert _conflicts(cache.lookup({0: 1, 1: 2}, True, epoch=0)) == [[0, 1]]

    def test_exact_match_inconclusive_lookup_hits(self):
        cache = PartialModelCache()
        cache.insert_inconclusive({0: 1, 1: 2}, True, epoch=0)
        assert cache.lookup({0: 1, 1: 2}, True, epoch=0) is None

    def test_different_option_on_the_same_parameter_misses(self):
        cache = PartialModelCache()
        cache.insert_refuted({0: 1}, True, conflict_parameters=[0], epoch=0)
        assert cache.lookup({0: 2}, True, epoch=0) is MISS


class TestRefutedSubsetSubsumption:
    """A refutation keyed by a *subset* of the query refutes the query too -- fixing more parameters can only narrow the induced sub-MDP further, so an already-
    infeasible region stays infeasible."""

    def test_a_superset_query_reuses_a_smaller_cached_refutation(self):
        cache = PartialModelCache()
        cache.insert_refuted({0: 1}, True, conflict_parameters=[0], epoch=0)
        assert _conflicts(cache.lookup({0: 1, 1: 2, 2: 3}, True, epoch=0)) == [[0]]

    def test_a_subset_query_does_not_reuse_a_larger_cached_refutation(self):
        """The opposite direction is not a valid subsumption: a larger fixed set being refuted says nothing about a smaller (less constrained) one."""
        cache = PartialModelCache()
        cache.insert_refuted({0: 1, 1: 2}, True, conflict_parameters=[0, 1], epoch=0)
        assert cache.lookup({0: 1}, True, epoch=0) is MISS

    def test_every_matching_refutation_is_returned(self):
        cache = PartialModelCache()
        cache.insert_refuted({0: 1}, True, conflict_parameters=[0], epoch=0)
        cache.insert_refuted({1: 2, 2: 3}, True, conflict_parameters=[1, 2], epoch=0)
        assert _conflicts(cache.lookup({0: 1, 1: 2, 2: 3, 3: 4}, True, epoch=0)) == [[0], [1, 2]]

    def test_a_non_matching_option_on_a_shared_parameter_does_not_subsume(self):
        cache = PartialModelCache()
        cache.insert_refuted({0: 1}, True, conflict_parameters=[0], epoch=0)
        assert cache.lookup({0: 2, 1: 2}, True, epoch=0) is MISS

    def test_a_refuted_viable_literal_survives_an_epoch_change(self):
        """No completion meets the threshold: neither can one meet a tighter one."""
        cache = PartialModelCache()
        cache.insert_refuted({0: 1}, True, conflict_parameters=[0], epoch=0)
        assert _conflicts(cache.lookup({0: 1}, True, epoch=5)) == [[0]]

    def test_a_refuted_not_viable_literal_is_dropped_by_an_epoch_change(self):
        """Every completion meets the threshold: that need not hold for a tighter one."""
        cache = PartialModelCache()
        cache.insert_refuted({0: 1}, False, conflict_parameters=[0], epoch=0)
        assert _conflicts(cache.lookup({0: 1}, False, epoch=0)) == [[0]]
        assert cache.lookup({0: 1}, False, epoch=1) is MISS

    def test_a_refutation_keeps_its_value(self):
        cache = PartialModelCache()
        cache.insert_refuted({0: 1, 1: 0}, False, conflict_parameters=[0], epoch=0, value=0.25, exact=True)
        assert cache.lookup({0: 1, 1: 1}, False, epoch=0) == [Refutation([0], 0.25, True)]


class TestRefutedCompaction:
    """Inserting a smaller (more general) refutation should retire any existing larger, now-redundant cached entry it subsumes -- mirrors molehill's own insert-
    time trie compaction."""

    def test_inserting_a_subset_conflict_retires_a_previously_cached_superset(self):
        cache = PartialModelCache()
        cache.insert_refuted({0: 1, 1: 2}, True, conflict_parameters=[0, 1], epoch=0)
        assert len(cache._refuted[1]) == 1
        cache.insert_refuted({0: 1}, True, conflict_parameters=[0], epoch=0)
        assert len(cache._refuted[1]) == 1, "the now-redundant {0:1,1:2} entry should have been removed"
        assert _conflicts(cache.lookup({0: 1, 1: 2}, True, epoch=0)) == [[0]], "the surviving entry must still answer the old query"

    def test_conflicts_over_the_same_parameters_are_kept_apart(self):
        """Retiring one of them must not retire the other (as it could when their payload doubled as their trie id)."""
        cache = PartialModelCache()
        cache.insert_refuted({0: 1, 1: 1}, True, conflict_parameters=[0, 1], epoch=0)
        cache.insert_refuted({0: 2, 1: 2}, True, conflict_parameters=[0, 1], epoch=0)
        cache.insert_refuted({0: 1}, True, conflict_parameters=[0], epoch=0)
        assert _conflicts(cache.lookup({0: 2, 1: 2}, True, epoch=0)) == [[0, 1]]
        assert _conflicts(cache.lookup({0: 1, 1: 1}, True, epoch=0)) == [[0]]

    def test_inserting_a_superset_conflict_does_not_disturb_an_existing_subset(self):
        cache = PartialModelCache()
        cache.insert_refuted({0: 1}, True, conflict_parameters=[0], epoch=0)
        cache.insert_refuted({0: 1, 1: 2}, True, conflict_parameters=[0, 1], epoch=0)
        assert len(cache._refuted[1]) == 2, "the new, strictly-larger entry is not redundant and should be kept"


class TestInconclusiveSupersetSubsumption:
    """An inconclusive verdict keyed by a *superset* of the query means the query is inconclusive too: if a more-constrained region wasn't refuted, a less-
    constrained one (whose completions are a superset) can't be either."""

    def test_a_subset_query_reuses_a_larger_cached_inconclusive_verdict(self):
        cache = PartialModelCache()
        cache.insert_inconclusive({0: 1, 1: 2}, True, epoch=0)
        assert cache.lookup({0: 1}, True, epoch=0) is None

    def test_a_superset_query_does_not_reuse_a_smaller_cached_inconclusive_verdict(self):
        cache = PartialModelCache()
        cache.insert_inconclusive({0: 1}, True, epoch=0)
        assert cache.lookup({0: 1, 1: 2}, True, epoch=0) is MISS

    def test_inconclusive_entries_are_invalidated_by_an_epoch_change(self):
        cache = PartialModelCache()
        cache.insert_inconclusive({0: 1, 1: 2}, True, epoch=0)
        assert cache.lookup({0: 1}, True, epoch=0) is None
        assert cache.lookup({0: 1}, True, epoch=1) is MISS, "a tightened threshold must invalidate a stale inconclusive verdict"

    def test_a_fresh_insert_after_an_epoch_change_is_usable_again(self):
        cache = PartialModelCache()
        cache.insert_inconclusive({0: 1}, True, epoch=0)
        cache.insert_inconclusive({0: 2}, True, epoch=1)
        assert cache.lookup({0: 2}, True, epoch=1) is None
        assert cache.lookup({0: 1}, True, epoch=1) is MISS


class TestCrossPolaritySubsumption:
    """A refutation for the *opposite* polarity, whether a subset or superset of the query, can only ever make the current polarity's query inconclusive --
    never a false refutation.

    See cache.py's module docstring for the derivation.
    """

    def test_an_opposite_polarity_subset_refutation_makes_the_query_inconclusive(self):
        cache = PartialModelCache()
        cache.insert_refuted({0: 1}, False, conflict_parameters=[0], epoch=0)
        assert cache.lookup({0: 1, 1: 2}, True, epoch=0) is None

    def test_an_opposite_polarity_superset_refutation_makes_the_query_inconclusive(self):
        cache = PartialModelCache()
        cache.insert_refuted({0: 1, 1: 2}, False, conflict_parameters=[0, 1], epoch=0)
        assert cache.lookup({0: 1}, True, epoch=0) is None

    def test_an_exact_opposite_polarity_refutation_makes_the_query_inconclusive(self):
        """The most common real case: a full assignment's viable(fixed) is refuted, so the subsequent not_viable(fixed) query on the exact same fixed set must
        come back inconclusive, never refuted -- the two literals are logical negations of each other on a fully-fixed assignment."""
        cache = PartialModelCache()
        cache.insert_refuted({0: 1, 1: 2}, True, conflict_parameters=[0, 1], epoch=0)
        assert cache.lookup({0: 1, 1: 2}, False, epoch=0) is None

    def test_cross_polarity_reuse_never_produces_a_false_refutation(self):
        cache = PartialModelCache()
        cache.insert_refuted({0: 1}, False, conflict_parameters=[0], epoch=0)
        result = cache.lookup({0: 1, 1: 2}, True, epoch=0)
        assert result is None, "cross-polarity information must never manufacture a positive refutation"

    def test_same_polarity_refutation_still_takes_priority_and_returns_a_conflict(self):
        """If the query is refuted for its *own* polarity too, that (more useful) answer must win over the weaker cross-polarity "just inconclusive"
        shortcut."""
        cache = PartialModelCache()
        cache.insert_refuted({0: 1}, True, conflict_parameters=[0], epoch=0)
        cache.insert_refuted({1: 2}, False, conflict_parameters=[1], epoch=0)
        assert _conflicts(cache.lookup({0: 1, 1: 2}, True, epoch=0)) == [[0]]


class TestSetTrieBackedStructure:
    """The cache keeps its sets in payntbind's own set tries, not in the mercury-settrie of the reference implementation."""

    def test_the_tries_are_payntbind_set_tries(self):
        cache = PartialModelCache()
        assert all(isinstance(trie, payntbind.synthesis.SetTrie) for trie in (*cache._refuted, *cache._inconclusive))

    def test_the_smpmc_package_does_not_need_mercury_settrie(self):
        """That package only comes as a source distribution that has to be compiled with setuptools, which PAYNT's build does without: nothing imports it."""
        script = "import sys; sys.modules['settrie'] = None; import paynt.api, paynt.synthesizer.smpmc, paynt.synthesizer.smpmc.cache"
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stderr

    def test_a_parameter_and_an_option_are_not_confused_with_each_other(self):
        """A pair is one integer of the trie: (parameter << 32) | option."""
        cache = PartialModelCache()
        cache.insert_refuted({1: 0}, True, conflict_parameters=[1], epoch=0)
        assert cache.lookup({0: 1}, True, epoch=0) is MISS
        assert _conflicts(cache.lookup({1: 0, 0: 1}, True, epoch=0)) == [[1]]

    def test_parameters_and_options_beyond_sixteen_bits_are_kept_apart(self):
        cache = PartialModelCache()
        cache.insert_refuted({70_000: 70_001, 3: 5}, True, conflict_parameters=[70_000, 3], epoch=0)
        assert cache.lookup({70_000: 70_002, 3: 5}, True, epoch=0) is MISS
        assert cache.lookup({70_001: 70_001, 3: 5}, True, epoch=0) is MISS
        assert _conflicts(cache.lookup({70_000: 70_001, 3: 5, 4: 9}, True, epoch=0)) == [[3, 70_000]]


class TestTheFirstEntryOfATrie:
    """The first set stored in a trie has the payload 0, which is falsy: it has to be found like any other, which a lookup that tested the payloads it gets back
    for truth would not do."""

    def test_the_first_inconclusive_verdict_is_used(self):
        cache = PartialModelCache()
        cache.insert_inconclusive({0: 1, 1: 2}, True, epoch=0)
        assert cache.lookup({0: 1}, True, epoch=0) is None

    def test_the_first_refutation_of_the_opposite_polarity_is_used(self):
        cache = PartialModelCache()
        cache.insert_refuted({0: 1}, False, conflict_parameters=[0], epoch=0)
        assert cache.lookup({0: 1, 1: 2}, True, epoch=0) is None
        assert cache.lookup({}, True, epoch=0) is None

    def test_the_first_refutation_is_used(self):
        cache = PartialModelCache()
        cache.insert_refuted({0: 1}, True, conflict_parameters=[0], epoch=0)
        assert _conflicts(cache.lookup({0: 1, 1: 2}, True, epoch=0)) == [[0]]

    def test_a_refutation_that_depends_on_no_parameter_refutes_everything(self):
        """Its key is the empty set, a subset of every query."""
        cache = PartialModelCache()
        cache.insert_refuted({0: 1, 1: 2}, True, conflict_parameters=[], epoch=0)
        assert _conflicts(cache.lookup({5: 5}, True, epoch=0)) == [[]]
        assert _conflicts(cache.lookup({}, True, epoch=0)) == [[]]

    def test_the_empty_conflict_retires_every_other_refutation_of_its_polarity(self):
        cache = PartialModelCache()
        cache.insert_refuted({0: 1}, True, conflict_parameters=[0], epoch=0)
        cache.insert_refuted({1: 2}, True, conflict_parameters=[1], epoch=0)
        cache.insert_refuted({1: 2}, False, conflict_parameters=[1], epoch=0)
        cache.insert_refuted({0: 1, 1: 2}, True, conflict_parameters=[], epoch=0)
        assert len(cache._refuted[1]) == 1 and len(cache._refuted[0]) == 1
        assert _conflicts(cache.lookup({0: 1, 1: 2}, True, epoch=0)) == [[]]
