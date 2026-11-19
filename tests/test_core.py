"""Tests for tiny_heap_based_top_k.core."""

import unittest

from tiny_heap_based_top_k import TopKTracker


class TestTopKTracker(unittest.TestCase):
    def test_capacity_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            TopKTracker(0)
        with self.assertRaises(ValueError):
            TopKTracker(-1)

    def test_capacity_must_be_int(self) -> None:
        with self.assertRaises(TypeError):
            TopKTracker(2.5)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            TopKTracker(True)  # type: ignore[arg-type]

    def test_empty_top_is_empty_list(self) -> None:
        tracker = TopKTracker(3)
        self.assertEqual(tracker.top(), [])
        self.assertEqual(len(tracker), 0)

    def test_min_retained_raises_when_empty(self) -> None:
        tracker = TopKTracker(3)
        with self.assertRaises(IndexError):
            tracker.min_retained()

    def test_add_below_capacity_keeps_everything(self) -> None:
        tracker = TopKTracker(5)
        for value in (3.0, 1.0, 4.0):
            tracker.add(value)
        self.assertEqual(len(tracker), 3)
        self.assertEqual(tracker.top(), [4.0, 3.0, 1.0])
        self.assertEqual(tracker.min_retained(), 1.0)

    def test_add_evicts_smallest_once_full(self) -> None:
        tracker = TopKTracker(3)
        for value in (1.0, 2.0, 3.0, 4.0):
            tracker.add(value)
        self.assertEqual(len(tracker), 3)
        self.assertEqual(tracker.top(), [4.0, 3.0, 2.0])
        self.assertEqual(tracker.min_retained(), 2.0)

    def test_ties_do_not_displace_existing(self) -> None:
        tracker = TopKTracker(2)
        tracker.add(5.0)
        tracker.add(5.0)
        tracker.add(5.0)
        self.assertEqual(tracker.top(), [5.0, 5.0])
        self.assertEqual(len(tracker), 2)

    def test_negative_and_zero_values(self) -> None:
        tracker = TopKTracker(3)
        for value in (-5.0, 0.0, -1.0, -10.0):
            tracker.add(value)
        self.assertEqual(tracker.top(), [0.0, -1.0, -5.0])

    def test_extend_matches_repeated_add(self) -> None:
        values = [10.0, 1.0, 7.0, 3.0, 9.0, 2.0]
        a = TopKTracker(4)
        b = TopKTracker(4)
        for value in values:
            a.add(value)
        b.extend(values)
        self.assertEqual(a.top(), b.top())

    def test_top_returns_copy(self) -> None:
        tracker = TopKTracker(2)
        tracker.add(1.0)
        tracker.add(2.0)
        result = tracker.top()
        result.append(99.0)
        self.assertEqual(tracker.top(), [2.0, 1.0])

    def test_contains_reflects_retained_values(self) -> None:
        tracker = TopKTracker(2)
        tracker.add(1.0)
        tracker.add(2.0)
        tracker.add(3.0)
        self.assertIn(2.0, tracker)
        self.assertIn(3.0, tracker)
        self.assertNotIn(1.0, tracker)


if __name__ == "__main__":
    unittest.main()
