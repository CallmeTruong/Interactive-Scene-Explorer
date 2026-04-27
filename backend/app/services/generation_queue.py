from collections.abc import Callable
from threading import Lock
from typing import TypeVar

T = TypeVar("T")


class GenerationSkipped(RuntimeError):
    """Raised when optional generation work is skipped because the GPU is busy."""


class GenerationQueue:
    """Single-process generation lock for local GPU-bound diffusion work."""

    def __init__(self) -> None:
        self._lock = Lock()

    @property
    def busy(self) -> bool:
        return self._lock.locked()

    def run_exclusive(self, task: Callable[[], T]) -> T:
        with self._lock:
            return task()

    def try_run_exclusive(self, task: Callable[[], T]) -> T:
        acquired = self._lock.acquire(blocking=False)
        if not acquired:
            raise GenerationSkipped()

        try:
            return task()
        finally:
            self._lock.release()


generation_queue = GenerationQueue()
