# tiny-heap-based-top-k

Maintains a bounded min-heap of the largest N numeric observations seen so far, with O(log N) insertion and O(N log N) extraction.

```python
from tiny_heap_based_top_k import TopKTracker

tracker = TopKTracker(3)
tracker.extend([5.0, 1.0, 9.0, 3.0, 7.0])
tracker.top()          # [9.0, 7.0, 5.0]
tracker.min_retained() # 5.0
len(tracker)           # 3
```

## Why this exists

Streaming observations often arrive faster than they can be stored. Keeping only the largest K values is enough for many dashboards and alerts, and a min-heap gives a cheap admission test: a new value only enters the heap if it beats the current minimum. The trade-off is that extraction is destructive in the sense that `top()` sorts a copy, so repeated extraction costs O(N log N) each time; the tracker is optimised for the streaming insert path, not for repeated full scans.

## Edge case

Ties are not displaced. If the heap is full and a new observation equals the current minimum, the new observation is discarded. That keeps the first-seen values in the top K and avoids unnecessary heap churn. The tracker also rejects a non-positive capacity and a non-integer capacity at construction time, so an empty tracker cannot be created by accident.
