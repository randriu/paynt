"""paynt.utils.timer: how much time is left before a limit."""

from __future__ import annotations

import pytest

import paynt.utils.timer


class Clock:
    """A settable stand-in for Timer.timestamp, so that no test has to sleep."""

    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


@pytest.fixture
def clock(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(paynt.utils.timer.Timer, "timestamp", staticmethod(clock))
    return clock


class TestTimerTimeRemaining:
    def test_is_none_without_a_limit(self, clock):
        timer = paynt.utils.timer.Timer()
        timer.start()
        clock.now += 30
        assert timer.time_remaining() is None

    def test_counts_down_from_the_limit(self, clock):
        timer = paynt.utils.timer.Timer(100)
        assert timer.time_remaining() == pytest.approx(100)  # not started: nothing has elapsed
        timer.start()
        clock.now += 30
        assert timer.time_remaining() == pytest.approx(70)

    def test_stops_counting_down_with_the_timer(self, clock):
        timer = paynt.utils.timer.Timer(100)
        timer.start()
        clock.now += 30
        timer.stop()
        clock.now += 50
        assert timer.time_remaining() == pytest.approx(70)

    def test_is_negative_exactly_when_the_limit_is_reached(self, clock):
        timer = paynt.utils.timer.Timer(100)
        timer.start()
        clock.now += 100
        assert not timer.time_limit_reached()
        assert timer.time_remaining() == pytest.approx(0)
        clock.now += 1
        assert timer.time_limit_reached()
        assert timer.time_remaining() == pytest.approx(-1)


class TestGlobalTimerTimeRemaining:
    def test_is_none_before_the_timer_is_started(self, monkeypatch):
        monkeypatch.setattr(paynt.utils.timer.GlobalTimer, "global_timer", None)
        assert paynt.utils.timer.GlobalTimer.time_remaining() is None

    def test_is_none_without_a_limit(self, monkeypatch):
        monkeypatch.setattr(paynt.utils.timer.GlobalTimer, "global_timer", None)
        paynt.utils.timer.GlobalTimer.start()
        assert paynt.utils.timer.GlobalTimer.time_remaining() is None

    def test_counts_down_from_the_limit(self, monkeypatch, clock):
        monkeypatch.setattr(paynt.utils.timer.GlobalTimer, "global_timer", None)
        paynt.utils.timer.GlobalTimer.start(60)
        clock.now += 15
        assert paynt.utils.timer.GlobalTimer.time_remaining() == pytest.approx(45)
