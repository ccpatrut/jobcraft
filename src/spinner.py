"""Animated spinner for long-running operations."""

from __future__ import annotations

import itertools
import sys
import threading
import time


class Spinner:
    """Context manager that shows an animated spinner with elapsed time.

    Usage:
        with Spinner("Loading model"):
            slow_operation()
        # prints: Loading model... done (3.2s)
    """

    _FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, message: str = "Working", stream=None):
        self.message = message
        self.stream = stream or sys.stdout
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._start_time = 0.0

    def _spin(self) -> None:
        frames = itertools.cycle(self._FRAMES)
        while not self._stop.is_set():
            elapsed = time.time() - self._start_time
            frame = next(frames)
            self.stream.write(f"\r  {frame} {self.message}... {elapsed:.0f}s")
            self.stream.flush()
            self._stop.wait(0.1)

    def __enter__(self) -> Spinner:
        self._start_time = time.time()
        self._stop.clear()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join()
        elapsed = time.time() - self._start_time
        self.stream.write(f"\r  ✓ {self.message} — done ({elapsed:.1f}s)" + " " * 10 + "\n")
        self.stream.flush()
