from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Iterable


@dataclass(frozen=True)
class CreditConfig:
    """Two-stage credit assignment.

    Let S0 be the Stage-0 rendered RSim (0 on failure) and S1 the final RSim.
    We view S0 as an intermediate potential and S1-S0 as the incremental reward.

    Return-to-go used by each policy call:
      Stage-0: G0 = scale * S1
      Stage-1: G1 = scale * (S1 - S0)

    This keeps the objective simple while avoiding the pathological incentive of
    applying a shared +Delta reward to Stage-0 (which would reward making S0 low).
    """
    scale: float = 10.0


def stage_returns(s0: float, s1: float, cfg: CreditConfig | None = None) -> tuple[float, float]:
    cfg = cfg or CreditConfig()
    return cfg.scale * float(s1), cfg.scale * (float(s1) - float(s0))


def centered_advantages(values: Iterable[float]) -> list[float]:
    values = [float(x) for x in values]
    if not values:
        return []
    m = mean(values)
    return [x - m for x in values]


def grouped_stage_advantages(
    s0: list[float],
    s1: list[float],
    group_size: int,
    cfg: CreditConfig | None = None,
) -> tuple[list[float], list[float], list[float], list[float]]:
    if len(s0) != len(s1):
        raise ValueError("s0 and s1 must have the same length")
    if group_size <= 0 or len(s0) % group_size != 0:
        raise ValueError("trajectory count must be divisible by group_size")
    r0, r1, a0, a1 = [], [], [], []
    for start in range(0, len(s0), group_size):
        g0, g1 = [], []
        for x0, x1 in zip(s0[start:start+group_size], s1[start:start+group_size], strict=True):
            rr0, rr1 = stage_returns(x0, x1, cfg)
            g0.append(rr0); g1.append(rr1)
        r0.extend(g0); r1.extend(g1)
        a0.extend(centered_advantages(g0)); a1.extend(centered_advantages(g1))
    return r0, r1, a0, a1


def failure_status(error: str | None, hit_max_tokens: bool, ends_with_document: bool) -> str:
    tags = []
    if hit_max_tokens:
        tags.append("hit_max_tokens")
    if not ends_with_document:
        tags.append("document_incomplete")
    if error:
        tags.append(str(error).split(":", 1)[0])
    if not tags:
        tags.append("render_failed")
    return ", ".join(dict.fromkeys(tags))
