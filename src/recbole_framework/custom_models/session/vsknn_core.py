"""Framework-independent building blocks for Vector Multiplication Session-kNN.

Keeping these operations independent from RecBole and PyTorch makes the algorithm
easy to verify against the published/reference implementation.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import math
from typing import Optional


WEIGHTING_FUNCTIONS = {"same", "linear", "div", "log", "quadratic"}
SIMILARITIES = {"vec", "cosine"}


def position_weights(items: Sequence[int], weighting: str) -> dict[int, float]:
    """Return VSKNN weights for the current session (last duplicate wins)."""
    if weighting not in WEIGHTING_FUNCTIONS:
        raise ValueError(f"Unsupported session weighting: {weighting}")

    length = len(items)
    weights: dict[int, float] = {}
    for position, item_id in enumerate(items, start=1):
        if weighting == "same":
            weight = 1.0
        elif weighting == "linear":
            weight = 1.0 - 0.1 * (length - position) if position <= 10 else 0.0
        elif weighting == "div":
            weight = position / length
        elif weighting == "log":
            weight = 1.0 / math.log10((length - position) + 1.7)
        else:
            weight = (position / length) ** 2
        weights[item_id] = weight
    return weights


def session_similarity(
    current_items: Sequence[int],
    neighbor_items: set[int],
    weighting: str = "div",
    similarity: str = "vec",
) -> float:
    """Compute the weighted VSKNN similarity used by the reference code."""
    if similarity not in SIMILARITIES:
        raise ValueError(f"Unsupported similarity: {similarity}")
    if not current_items or not neighbor_items:
        return 0.0

    weights = position_weights(current_items, weighting)
    return weighted_session_similarity(weights, neighbor_items, similarity)


def weighted_session_similarity(
    weights: dict[int, float],
    neighbor_items: set[int],
    similarity: str = "vec",
) -> float:
    """Compute similarity from weights prepared once for the current session."""
    if similarity not in SIMILARITIES:
        raise ValueError(f"Unsupported similarity: {similarity}")
    if not weights or not neighbor_items:
        return 0.0

    overlap = set(weights).intersection(neighbor_items)
    numerator = sum(weights[item_id] for item_id in overlap)

    if similarity == "vec":
        return numerator / len(weights)

    current_norm = math.sqrt(sum(weight * weight for weight in weights.values()))
    neighbor_norm = math.sqrt(len(neighbor_items))
    return numerator / (current_norm * neighbor_norm)


def score_decay(
    current_items: Sequence[int], neighbor_items: set[int], weighting: str
) -> float:
    """Weight a neighbor by the recency of its last shared current-session item."""
    if weighting not in WEIGHTING_FUNCTIONS:
        raise ValueError(f"Unsupported score weighting: {weighting}")

    step: Optional[int] = next(
        (
            distance
            for distance, item in enumerate(reversed(current_items), start=1)
            if item in neighbor_items
        ),
        None,
    )
    if step is None:
        return 0.0
    if weighting == "same":
        return 1.0
    if weighting == "linear":
        return 1.0 - 0.1 * step if step <= 100 else 0.0
    if weighting == "div":
        return 1.0 / step
    if weighting == "log":
        return 1.0 / math.log10(step + 1.7)
    return 1.0 / (step * step)


def recent_item_steps(current_items: Sequence[int]) -> dict[int, int]:
    """Map each item to its distance from the end (last occurrence wins)."""
    steps: dict[int, int] = {}
    for step, item_id in enumerate(reversed(current_items), start=1):
        steps.setdefault(item_id, step)
    return steps


def score_decay_from_steps(
    item_steps: dict[int, int], neighbor_items: set[int], weighting: str
) -> float:
    """Compute score decay from precomputed current-session recency steps."""
    if weighting not in WEIGHTING_FUNCTIONS:
        raise ValueError(f"Unsupported score weighting: {weighting}")
    shared_steps = (item_steps[item] for item in neighbor_items if item in item_steps)
    step = min(shared_steps, default=None)
    if step is None:
        return 0.0
    if weighting == "same":
        return 1.0
    if weighting == "linear":
        return 1.0 - 0.1 * step if step <= 100 else 0.0
    if weighting == "div":
        return 1.0 / step
    if weighting == "log":
        return 1.0 / math.log10(step + 1.7)
    return 1.0 / (step * step)


def score_neighbors(
    current_items: Sequence[int],
    neighbors: Iterable[tuple[set[int], float]],
    weighting: str = "div",
) -> dict[int, float]:
    """Aggregate item scores from ``(neighbor_items, similarity)`` pairs."""
    scores: dict[int, float] = {}
    for neighbor_items, similarity in neighbors:
        contribution = float(similarity) * score_decay(
            current_items, neighbor_items, weighting
        )
        for item_id in neighbor_items:
            scores[item_id] = scores.get(item_id, 0.0) + contribution
    return scores


def score_neighbors_from_steps(
    item_steps: dict[int, int],
    neighbors: Iterable[tuple[set[int], float]],
    weighting: str = "div",
) -> dict[int, float]:
    """Aggregate scores using current-session recency prepared once."""
    scores: dict[int, float] = {}
    for neighbor_items, similarity in neighbors:
        contribution = float(similarity) * score_decay_from_steps(
            item_steps, neighbor_items, weighting
        )
        for item_id in neighbor_items:
            scores[item_id] = scores.get(item_id, 0.0) + contribution
    return scores


def collapse_augmented_sessions(
    rows: Iterable[tuple[int, Sequence[int], int, int, float]],
) -> list[tuple[int, list[int], float]]:
    """Collapse RecBole prefix-target rows into one longest row per train session.

    Each input row is ``(session_id, prefix, prefix_length, target, timestamp)``.
    Only rows already present in the training split may be passed to this function.
    Consequently, the returned sessions cannot contain validation/test targets.
    """
    longest: dict[int, tuple[list[int], float]] = {}
    for session_id, prefix, prefix_length, target, timestamp in rows:
        items = [int(item) for item in prefix[:prefix_length] if int(item) > 0]
        if int(target) > 0:
            items.append(int(target))
        if not items:
            continue

        previous = longest.get(int(session_id))
        if previous is None or len(items) > len(previous[0]) or (
            len(items) == len(previous[0]) and float(timestamp) > previous[1]
        ):
            longest[int(session_id)] = (items, float(timestamp))

    return [
        (session_id, items, timestamp)
        for session_id, (items, timestamp) in sorted(longest.items())
    ]
