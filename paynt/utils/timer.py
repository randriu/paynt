from __future__ import annotations

import time
import psutil
import os

import logging

logger = logging.getLogger(__name__)


class Timer:
    def __init__(self, time_limit_seconds: float | None = None):
        self.running = False
        self.last_timestamp: float | None = None
        self.time: float = 0  # total time measured
        self.time_limit_seconds = time_limit_seconds

    @staticmethod
    def timestamp() -> float:
        return time.perf_counter()

    def reset(self) -> None:
        self.__init__()  # type: ignore[misc]

    def start(self) -> None:
        if self.running:
            return
        self.last_timestamp = self.timestamp()
        self.running = True

    def stop(self) -> None:
        if not self.running:
            return
        assert self.last_timestamp is not None
        self.time += self.timestamp() - self.last_timestamp
        self.last_timestamp = None
        self.running = False

    def read(self) -> float:
        if not self.running:
            return self.time
        assert self.last_timestamp is not None
        return self.time + (self.timestamp() - self.last_timestamp)

    def time_limit_reached(self) -> bool:
        return self.time_limit_seconds is not None and self.read() > self.time_limit_seconds

    def time_remaining(self) -> float | None:
        """Seconds left until time_limit_reached() turns true (negative once it has), or None if there is no time limit."""
        return None if self.time_limit_seconds is None else self.time_limit_seconds - self.read()


class GlobalTimer:
    global_timer: Timer | None = None

    @classmethod
    def start(cls, time_limit_seconds: float | None = None) -> None:
        cls.global_timer = Timer(time_limit_seconds)
        cls.global_timer.start()

    @classmethod
    def read(cls) -> float:
        if cls.global_timer is None:
            logger.warning("attempting to read uninitialized GlobalTimer")
            return -1
        return cls.global_timer.read()

    @classmethod
    def time_limit_reached(cls) -> bool:
        return cls.global_timer is not None and cls.global_timer.time_limit_reached()

    @classmethod
    def time_remaining(cls) -> float | None:
        return None if cls.global_timer is None else cls.global_timer.time_remaining()


class GlobalMemoryLimit:
    memory_limit_mb: float | None = None

    @classmethod
    def limit_reached(cls) -> bool:
        process = psutil.Process(os.getpid())
        allocated_mb = process.memory_info().rss / (1024 * 1024)
        return cls.memory_limit_mb is not None and allocated_mb > cls.memory_limit_mb
