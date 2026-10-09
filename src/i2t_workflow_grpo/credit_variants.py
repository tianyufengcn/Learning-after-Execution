"""Credit-return variants used by the training objective.

The main method (Variant A) is implemented directly in
``i2t_workflow_grpo.workflow.stage_returns``:

    Stage-0 return G0 = lambda * S1
    Stage-1 return G1 = lambda * (S1 - S0)

Advantages are then the per-root-group mean-centred returns, computed
separately for the two stages, and the two stages are optimised with fixed
weights 0.5 / 0.5.

The factorised credit ablations in the paper replace only the two raw return
definitions above; grouping, centring, masking, optimisation schedule and all
other hyper-parameters are unchanged. Each ablation was trained from a copy of
the main implementation with ``stage_returns`` swapped, so the functions below
are provided as a convenience reproduction interface - they are not a switch
that existed in the original training entry point.

    A (main):  G0 = scale * S1          G1 = scale * (S1 - S0)
    B:         G0 = scale * S0          G1 = scale * (S1 - S0)
    C:         G0 = scale * S1          G1 = scale * S1
    D:         G0 = scale * S0          G1 = scale * S1

The training path reaches credit assignment through
``i2t_workflow_grpo.workflow.stage_returns`` (called by
``workflow.grouped_stage_advantages`` and by ``transport.py``), and that function
always implements Variant A. To reproduce an ablation, point those call sites at
the variant you want, for example by replacing the import in
``i2t_workflow_grpo/transport.py`` and the module-level call in
``workflow.grouped_stage_advantages`` with
``from .credit_variants import stage_returns`` and setting ``CREDIT_VARIANT=B``
(or ``C`` / ``D``) in the environment. ``CREDIT_VARIANT`` only affects the
``stage_returns`` defined in *this* module; it does not change Variant A.
"""

from __future__ import annotations

import os

from .workflow import CreditConfig, stage_returns as _main_stage_returns

__all__ = ["CREDIT_VARIANT", "stage_returns", "stage_returns_for_variant", "VARIANT_NAMES"]

VARIANT_NAMES = ("A", "B", "C", "D")


def _stage_returns_b(s0: float, s1: float, cfg: CreditConfig | None = None) -> tuple[float, float]:
    cfg = cfg or CreditConfig()
    return cfg.scale * float(s0), cfg.scale * (float(s1) - float(s0))


def _stage_returns_c(s0: float, s1: float, cfg: CreditConfig | None = None) -> tuple[float, float]:
    cfg = cfg or CreditConfig()
    return cfg.scale * float(s1), cfg.scale * float(s1)


def _stage_returns_d(s0: float, s1: float, cfg: CreditConfig | None = None) -> tuple[float, float]:
    cfg = cfg or CreditConfig()
    return cfg.scale * float(s0), cfg.scale * float(s1)


_VARIANTS = {
    "A": _main_stage_returns,
    "B": _stage_returns_b,
    "C": _stage_returns_c,
    "D": _stage_returns_d,
}

def active_variant() -> str:
    """The variant selected by ``CREDIT_VARIANT`` (defaults to the main method)."""
    return os.environ.get("CREDIT_VARIANT", "A").upper()


# Convenience snapshot of the default selection; the ablations are opt-in.
CREDIT_VARIANT = active_variant()


def stage_returns_for_variant(
    variant: str, s0: float, s1: float, cfg: CreditConfig | None = None
) -> tuple[float, float]:
    key = str(variant).upper()
    if key not in _VARIANTS:
        raise ValueError(f"Unknown credit variant {variant!r}; expected one of {VARIANT_NAMES}")
    return _VARIANTS[key](s0, s1, cfg)


def stage_returns(s0: float, s1: float, cfg: CreditConfig | None = None) -> tuple[float, float]:
    """Return the (G0, G1) pair of the selected credit variant."""
    return stage_returns_for_variant(active_variant(), s0, s1, cfg)
