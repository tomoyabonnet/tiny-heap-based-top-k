"""Bounded min-heap that tracks the largest N observations.

The heap holds at most ``capacity`` items. The smallest item in the heap is
at the root, so when a new observation arrives we can compare it against the
root in O(1) and only push/pop when it belongs in the top K. This keeps
insertion O(log N) and extraction O(N log N) while using O(N) memory.
"""

from __future__ import annotations

import heapq
from typing import Iterable, List


class TopKTracker:
    """Track the largest ``capacity`` numeric observations seen so far.

    The tracker is a bounded min-heap. The root is the smallest of the
    currently retained values, which is exactly the value most likely to be
    evicted by the next observation.

    Args:
        capacity: Maximum number of observations to retain. Must be positive.

    Raises:
        ValueError: If ``capacity`` is not a positive integer.
    """

    __slots__ = ("_capacity", "_heap")

    def __init__(self, capacity: int) -> None:
        if not isinstance(capacity, int) or isinstance(capacity, bool):
            raise TypeError("capacity must be an int")
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._capacity = capacity
        self._heap: List[float] = []

    @property
    def capacity(self) -> int:
        """The maximum number of observations retained."""
        return self._capacity

    def __len__(self) -> int:
        """Number of observations currently retained (at most ``capacity``)."""
        return len(self._heap)

    def __contains__(self, value: object) -> bool:
        """Return True if ``value`` is currently retained.

        This is O(N) because the heap is not a set; it exists for tests and
        debugging, not for hot paths.
        """
        return value in self._heap

    def add(self, value: float) -> None:
        """Offer ``value`` to the tracker.

        If the heap is not yet full, the value is always retained. Once full,
        the value is retained only if it is strictly greater than the current
        minimum; otherwise it is discarded. Ties are discarded because the
        existing observation is already in the top K.
        """
        if len(self._heap) < self._capacity:
            heapq.heappush(self._heap, value)
            return
        if value > self._heap[0]:
            heapq.heapreplace(self._heap, value)

    def extend(self, values: Iterable[float]) -> None:
        """Add every value from ``values`` in order."""
        for value in values:
            self.add(value)

    def top(self) -> List[float]:
        """Return the retained observations in descending order.

        The result is a new list; mutating it does not affect the tracker.
        """
        return sorted(self._heap, reverse=True)

    def min_retained(self) -> float:
        """Return the smallest retained observation.

        Raises:
            IndexError: If no observations have been added yet.
        """
        if not self._heap:
            raise IndexError("no observations retained")
        return self._heap[0]
