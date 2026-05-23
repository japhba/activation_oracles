from __future__ import annotations

import math
import random


def sample_cot_oracle_stochastic_positions(
    base_positions: list[int],
    rng: random.Random,
    *,
    max_k: int = 100,
    include_boundaries: bool = True,
) -> list[int]:
    """Match third_party/cot-oracle's stochastic activation-position sampler."""
    assert base_positions
    assert max_k >= 2

    K = len(base_positions)
    if K <= 1:
        return base_positions[-1:]

    r = rng.random()
    if r < 0.20:
        if include_boundaries:
            return sorted({base_positions[0], base_positions[-1]})
        return base_positions[-1:]
    if r < 0.35:
        picked = set(base_positions[-2:])
        if include_boundaries:
            picked.add(base_positions[0])
        return sorted(picked)
    if r < 0.50:
        picked = set(base_positions[-3:])
        if include_boundaries:
            picked.add(base_positions[0])
        return sorted(picked)

    lo, hi = 2, min(max_k, K)
    k = int(round(math.exp(rng.uniform(math.log(lo), math.log(hi)))))
    k = max(lo, min(k, K))

    picked = set(rng.sample(base_positions, k))
    if include_boundaries:
        picked.add(base_positions[0])
    picked.add(base_positions[-1])
    return sorted(picked)


def sample_cot_oracle_token_positions(
    num_tokens: int,
    rng: random.Random,
    *,
    max_k: int = 100,
) -> list[int]:
    assert num_tokens >= 1
    return sample_cot_oracle_stochastic_positions(list(range(num_tokens)), rng, max_k=max_k)
