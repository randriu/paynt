"""Synthesizer.time_remaining: the time left before time_limit_reached() turns true, whichever of the synthesis timer and the global timer is tighter; and the
engines stopping there."""

from __future__ import annotations

import time

import pytest

import paynt.synthesizer.synthesizer
import paynt.utils.timer

from tests.helpers.models import load_colored_mdp


@pytest.fixture
def synthesizer(monkeypatch):
    monkeypatch.setattr(paynt.utils.timer.GlobalTimer, "global_timer", None)
    # the base class only stores what it is given, so a colored MDP and a task are not needed to ask about time
    return paynt.synthesizer.synthesizer.Synthesizer(None, None)  # type: ignore[arg-type]


def start_synthesis_timer(synthesizer, limit):
    synthesizer.synthesis_timer = paynt.utils.timer.Timer(limit)
    synthesizer.synthesis_timer.start()


class TestTimeRemaining:
    def test_is_none_without_any_limit(self, synthesizer):
        assert synthesizer.time_remaining() is None
        start_synthesis_timer(synthesizer, None)
        paynt.utils.timer.GlobalTimer.start()
        assert synthesizer.time_remaining() is None

    def test_follows_the_synthesis_timer(self, synthesizer):
        start_synthesis_timer(synthesizer, 100)
        assert synthesizer.time_remaining() == pytest.approx(100, abs=5)

    def test_follows_the_global_timer(self, synthesizer):
        paynt.utils.timer.GlobalTimer.start(40)
        assert synthesizer.time_remaining() == pytest.approx(40, abs=5)

    def test_is_the_tighter_of_the_two_timers(self, synthesizer):
        start_synthesis_timer(synthesizer, 100)
        paynt.utils.timer.GlobalTimer.start(40)
        assert synthesizer.time_remaining() == pytest.approx(40, abs=5)
        start_synthesis_timer(synthesizer, 10)
        assert synthesizer.time_remaining() == pytest.approx(10, abs=5)

    @pytest.mark.parametrize("limit", [-1, 100])
    def test_is_negative_exactly_when_the_time_limit_is_reached(self, synthesizer, limit):
        start_synthesis_timer(synthesizer, limit)
        assert synthesizer.time_limit_reached() is (synthesizer.time_remaining() < 0)


@pytest.mark.parametrize("method", ["onebyone", "ar", "cegis", "hybrid"])
def test_an_engine_stops_at_its_time_limit(monkeypatch, method):
    """The dpm-demo of models/performance-test takes every engine far longer than a second; SMPMC, which does not take its two properties, has tests of its own
    in smpmc/test_synthesizer.py."""
    monkeypatch.setattr(paynt.utils.timer.GlobalTimer, "global_timer", None)
    colored_mdp, task = load_colored_mdp("performance-test/dpm-demo")
    synthesizer = paynt.synthesizer.synthesizer.Synthesizer.for_method(colored_mdp, task, method)
    start = time.monotonic()
    synthesizer.synthesize(timeout=1, print_stats=False)
    assert synthesizer.time_limit_reached()
    assert time.monotonic() - start < 5
