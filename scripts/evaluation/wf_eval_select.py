#!/usr/bin/env python3
"""Execution-aware acceptance ("Selected" output) for two-stage workflow records.

The rule is the one used for the reported held-out results (see the paper's
workflow evaluation protocol): it is deterministic, uses only execution success
and the raw RSim score of the two stages, and needs no extra model inference.

```text
success0 = Stage0 render success        score0 = Stage0 RSim (raw)
success1 = Stage1 render success        score1 = Stage1 RSim (raw)

!success0 &  success1   -> Stage1  (FS_RESCUE)
 success0 & !success1   -> Stage0  (SF_PRESERVE)
 success0 &  success1   -> Stage1 if score1 > score0
                           Stage0 otherwise (exact tie -> Stage0)
!success0 & !success1   -> failure (no output)
```

Aggregates follow the fixed-zero convention used everywhere else in the
evaluation: a failed render contributes 0 to every visual metric (it is never
dropped from the denominator).

Usage (aggregation only; no generation, rendering or metric computation):

    python wf_eval_select.py --eval-root <root> --policy <name> [--json-out file.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

METRICS = ("rsim", "dino", "dreamsim", "clipimg")


def choose_stage(record: dict[str, Any]) -> tuple[str | None, str]:
    """Return (chosen stage, reason). `None` means unsuccessful sample."""
    s0, s1 = record.get("stage0") or {}, record.get("stage1") or {}
    # Same success test as the evaluator that produced these records: a stage is
    # usable only when it rendered and the rendered image was written.
    ok0 = bool(s0.get("render_success") and s0.get("render_path"))
    ok1 = bool(s1.get("render_success") and s1.get("render_path"))
    score0 = s0.get("rsim") if ok0 else None
    score1 = s1.get("rsim") if ok1 else None

    if not ok0 and not ok1:
        return None, "FF_FAILURE"
    if not ok0 and ok1:
        return "stage1", "FS_RESCUE"
    if ok0 and not ok1:
        return "stage0", "SF_PRESERVE"
    if float(score1) > float(score0):
        return "stage1", "SS_RSIM_IMPROVED"
    if float(score1) == float(score0):
        return "stage0", "SS_RSIM_TIE_PRESERVE_STAGE0"
    return "stage0", "SS_RSIM_KEPT_STAGE0"


def selected_metrics(record: dict[str, Any]) -> dict[str, float]:
    """Fixed-zero metrics of the accepted output for one sample."""
    stage, _ = choose_stage(record)
    if stage is None:
        return {"render": 0.0, **{m: 0.0 for m in METRICS}}
    payload = record[stage] or {}
    out = {"render": 1.0 if payload.get("render_success") else 0.0}
    for metric in METRICS:
        value = payload.get(metric)
        out[metric] = float(value) if isinstance(value, (int, float)) else 0.0
    return out


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(records)
    totals = {"render": 0.0, **{m: 0.0 for m in METRICS}}
    reasons: dict[str, int] = {}
    chosen = {"stage0": 0, "stage1": 0, "failure": 0}
    transitions: dict[str, dict[str, int]] = {}
    for record in records:
        stage, reason = choose_stage(record)
        reasons[reason] = reasons.get(reason, 0) + 1
        chosen[stage or "failure"] += 1
        transition = str(record.get("transition") or "?")
        bucket = transitions.setdefault(transition, {"n": 0, "stage0": 0, "stage1": 0, "failure": 0, "tie": 0})
        bucket["n"] += 1
        bucket[stage or "failure"] += 1
        if reason == "SS_RSIM_TIE_PRESERVE_STAGE0":
            bucket["tie"] += 1
        for key, value in selected_metrics(record).items():
            totals[key] += value
    return {
        "n": n,
        "selected": {k: (v / n if n else 0.0) for k, v in totals.items()},
        "selection_counts": chosen,
        "selection_reasons": reasons,
        "transitions": transitions,
        "convention": "fixed-zero: failed render contributes 0, never dropped",
    }


def load_records(eval_root: Path, policy: str) -> list[dict[str, Any]]:
    results = eval_root / policy / "results"
    if not results.is_dir():
        raise FileNotFoundError(f"no per-sample results under {results}")
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(results.glob("*.json"))]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--eval-root", required=True, help="evaluation root directory")
    parser.add_argument("--policy", required=True, help="policy sub-directory holding results/")
    parser.add_argument("--json-out", default=None, help="optional path for the JSON summary")
    args = parser.parse_args(argv)

    records = load_records(Path(args.eval_root), args.policy)
    summary = {"policy": args.policy, **aggregate(records)}
    order = ["n", "selected", "selection_counts", "selection_reasons", "convention"]
    ordered = {k: summary[k] for k in order if k in summary}
    ordered.update({k: v for k, v in summary.items() if k not in ordered})
    print(json.dumps(ordered, indent=2, ensure_ascii=False))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(ordered, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
