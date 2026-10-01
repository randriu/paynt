"""Synthesizer.time_remaining: the time left before time_limit_reached() turns true, whichever of the synthesis timer and the global timer is tighter."""

from __future__ import annotations

import pytest

import paynt.synthesizer.synthesizer
import paynt.utils.timer


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

    def test_is_negative_exactly_when_the_time_limit_is_reached(self, synthesizer):
        start_synthesis_timer(synthesizer, -1)
        assert synthesizer.time_limit_reached()
        assert synthesizer.time_remaining() < 0
