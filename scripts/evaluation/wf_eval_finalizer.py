"""Finalizer for the Workflow-GRPO two-stage evaluation.

Builds per-policy summary.json / predictions.jsonl / DONE, the initializer vs
trained-checkpoint comparison (JSON + CSV), and a simple sortable HTML gallery
for human inspection of visual self-correction. The sample list, its size and
every coverage/denominator quantity come from the manifest copy the controller
wrote for the evaluation root.

Metric semantics (sections 30-37 of the eval contract):
  - S = Stage-0 compile+render success, F = failure
  - transitions FF / FS / SF / SS
  - repair subset = Stage-0 fail; rescue = FS / (FF+FS)
  - revision subset = Stage-0 success; break = SF / (SF+SS)
  - revision_ss_* = strict visual metrics on SS only (E[V1-V0 | S0 & S1])
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import statistics
from pathlib import Path
from typing import Any

import numpy as np

from wf_eval_common import EVAL_ROOT, discover_policies, manifest_copy_path

EPSILON = 0.005
BUCKETS = [(0.0, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 1.0001)]
BUCKET_LABELS = ["[0,.4)", "[.4,.5)", "[.5,.6)", "[.6,.7)", "[.7,1.0]"]


def read_results(policy_root: Path, manifest_order: list[str]) -> list[dict[str, Any]]:
    results_dir = policy_root / "results"
    by_id: dict[str, dict[str, Any]] = {}
    if results_dir.exists():
        for p in results_dir.glob("*.json"):
            try:
                row = json.loads(p.read_text(encoding="utf-8"))
                by_id[str(row["sample_id"])] = row
            except Exception:
                continue
    out = []
    for sid in manifest_order:
        if sid in by_id:
            out.append(by_id[sid])
    return out


def _mean(vals: list[float]) -> float | None:
    return float(np.mean(vals)) if vals else None


def _median(vals: list[float]) -> float | None:
    return float(np.median(vals)) if vals else None


def _percentiles(vals: list[float]) -> dict[str, float | None]:
    if not vals:
        return {"p10": None, "p25": None, "p50": None, "p75": None, "p90": None}
    arr = np.asarray(vals, dtype=float)
    return {
        "p10": round(float(np.percentile(arr, 10)), 4),
        "p25": round(float(np.percentile(arr, 25)), 4),
        "p50": round(float(np.percentile(arr, 50)), 4),
        "p75": round(float(np.percentile(arr, 75)), 4),
        "p90": round(float(np.percentile(arr, 90)), 4),
    }


def stage_ok(row: dict[str, Any], stage: str) -> bool:
    s = row.get(stage) or {}
    return bool(s.get("render_success") and s.get("render_path"))


def stage_rsim(row: dict[str, Any], stage: str) -> float | None:
    s = row.get(stage) or {}
    v = s.get("rsim")
    return float(v) if isinstance(v, (int, float)) else None


def stage_metric(row: dict[str, Any], stage: str, metric: str) -> float | None:
    s = row.get(stage) or {}
    v = s.get(metric)
    return float(v) if isinstance(v, (int, float)) else None


def policy_metrics(rows: list[dict[str, Any]], n: int) -> dict[str, Any]:
    s0_compile = sum(1 for r in rows if (r.get("stage0") or {}).get("compile_success"))
    s1_compile = sum(1 for r in rows if (r.get("stage1") or {}).get("compile_success"))
    s0_render = sum(1 for r in rows if stage_ok(r, "stage0"))
    s1_render = sum(1 for r in rows if stage_ok(r, "stage1"))

    s0_rsim_render_only = [stage_rsim(r, "stage0") for r in rows if stage_ok(r, "stage0")]
    s1_rsim_render_only = [stage_rsim(r, "stage1") for r in rows if stage_ok(r, "stage1")]
    s0_rsim_render_only = [v for v in s0_rsim_render_only if v is not None]
    s1_rsim_render_only = [v for v in s1_rsim_render_only if v is not None]

    s0_fz = [float(stage_rsim(r, "stage0") or 0.0) for r in rows]
    s1_fz = [float(stage_rsim(r, "stage1") or 0.0) for r in rows]
    deltas_fz = [b - a for a, b in zip(s0_fz, s1_fz, strict=True)]

    trans = {"FF": 0, "FS": 0, "SF": 0, "SS": 0}
    for r in rows:
        t = r.get("transition")
        if t in trans:
            trans[t] += 1

    # repair subset (Stage-0 fail)
    repair = [r for r in rows if not stage_ok(r, "stage0")]
    repair_n = len(repair)
    ff_n = sum(1 for r in repair if r.get("transition") == "FF")
    fs_n = sum(1 for r in repair if r.get("transition") == "FS")
    repair_rescue_rate = fs_n / (ff_n + fs_n) if (ff_n + fs_n) else None
    rescued_rsim = [stage_rsim(r, "stage1") for r in repair if r.get("transition") == "FS"]
    rescued_rsim = [v for v in rescued_rsim if v is not None]

    # revision subset (Stage-0 success)
    revision = [r for r in rows if stage_ok(r, "stage0")]
    revision_n = len(revision)
    sf_n = sum(1 for r in revision if r.get("transition") == "SF")
    ss_n = sum(1 for r in revision if r.get("transition") == "SS")
    revision_break_compile_rate = sf_n / (sf_n + ss_n) if (sf_n + ss_n) else None

    # strict SS visual self-correction
    ss = [r for r in rows if r.get("transition") == "SS"]
    ss_deltas = []
    for r in ss:
        v0 = stage_rsim(r, "stage0")
        v1 = stage_rsim(r, "stage1")
        if v0 is not None and v1 is not None:
            ss_deltas.append(v1 - v0)
    ss_s0 = [stage_rsim(r, "stage0") for r in ss]
    ss_s1 = [stage_rsim(r, "stage1") for r in ss]
    ss_s0 = [v for v in ss_s0 if v is not None]
    ss_s1 = [v for v in ss_s1 if v is not None]
    improve = sum(1 for d in ss_deltas if d > EPSILON)
    regress = sum(1 for d in ss_deltas if d < -EPSILON)
    neutral = sum(1 for d in ss_deltas if abs(d) <= EPSILON)
    pos = [d for d in ss_deltas if d > EPSILON]
    neg = [d for d in ss_deltas if d < -EPSILON]

    # Stage-0 quality buckets (only Stage-0 success)
    buckets: list[dict[str, Any]] = []
    for (lo, hi), label in zip(BUCKETS, BUCKET_LABELS, strict=True):
        group = [r for r in revision if lo <= (stage_rsim(r, "stage0") or -1.0) < hi]
        g_ss = [r for r in group if r.get("transition") == "SS"]
        g_sf = [r for r in group if r.get("transition") == "SF"]
        g_deltas = []
        for r in g_ss:
            v0 = stage_rsim(r, "stage0")
            v1 = stage_rsim(r, "stage1")
            if v0 is not None and v1 is not None:
                g_deltas.append(v1 - v0)
        buckets.append(
            {
                "bucket": label,
                "n": len(group),
                "SS_n": len(g_ss),
                "SF_n": len(g_sf),
                "break_compile_rate": (
                    round(len(g_sf) / (len(g_sf) + len(g_ss)), 4)
                    if (len(g_sf) + len(g_ss))
                    else None
                ),
                "SS_delta_mean": round(float(np.mean(g_deltas)), 4) if g_deltas else None,
                "SS_delta_median": round(float(np.median(g_deltas)), 4) if g_deltas else None,
                "SS_improve_rate": (
                    round(sum(1 for d in g_deltas if d > EPSILON) / len(g_deltas), 4)
                    if g_deltas
                    else None
                ),
                "SS_regress_rate": (
                    round(sum(1 for d in g_deltas if d < -EPSILON) / len(g_deltas), 4)
                    if g_deltas
                    else None
                ),
            }
        )

    # Supplemental DINO / DreamSim / CLIPImg (scored from saved renders).
    supplemental: dict[str, Any] = {}
    for m in ("dino", "dreamsim", "clipimg"):
        s0ro = [
            v for v in [stage_metric(r, "stage0", m) for r in rows if stage_ok(r, "stage0")]
            if v is not None
        ]
        s1ro = [
            v for v in [stage_metric(r, "stage1", m) for r in rows if stage_ok(r, "stage1")]
            if v is not None
        ]
        s0fz = [float(stage_metric(r, "stage0", m) or 0.0) for r in rows]
        s1fz = [float(stage_metric(r, "stage1", m) or 0.0) for r in rows]
        ss_d = []
        for r in ss:
            v0 = stage_metric(r, "stage0", m)
            v1 = stage_metric(r, "stage1", m)
            if v0 is not None and v1 is not None:
                ss_d.append(v1 - v0)
        sup_improve = sum(1 for d in ss_d if d > EPSILON)
        sup_regress = sum(1 for d in ss_d if d < -EPSILON)
        supplemental[m] = {
            "stage0_render_only_mean": _mean(s0ro),
            "stage1_render_only_mean": _mean(s1ro),
            "stage0_fixed_zero_mean": _mean(s0fz),
            "stage1_fixed_zero_mean": _mean(s1fz),
            "ss_n": len(ss_d),
            "ss_delta_mean": _mean(ss_d),
            "ss_delta_median": _median(ss_d),
            "ss_improve_rate": round(sup_improve / len(ss_d), 4) if ss_d else None,
            "ss_regress_rate": round(sup_regress / len(ss_d), 4) if ss_d else None,
            "ss_neutral_rate": round((len(ss_d) - sup_improve - sup_regress) / len(ss_d), 4) if ss_d else None,
            "ss_percentiles": _percentiles(ss_d),
        }

    return {
        "n": len(rows),
        "expected": n,
        "stage0_compile_rate": round(s0_compile / n, 4) if n else None,
        "stage1_compile_rate": round(s1_compile / n, 4) if n else None,
        "stage0_render_rate": round(s0_render / n, 4) if n else None,
        "stage1_render_rate": round(s1_render / n, 4) if n else None,
        "stage0_rsim_render_only": _mean(s0_rsim_render_only),
        "stage1_rsim_render_only": _mean(s1_rsim_render_only),
        "stage0_rsim_fixed_zero": _mean(s0_fz),
        "stage1_rsim_fixed_zero": _mean(s1_fz),
        "overall_delta_fixed_zero": _mean(deltas_fz),
        "transitions": trans,
        "repair": {
            "repair_n": repair_n,
            "FF_n": ff_n,
            "FS_n": fs_n,
            "repair_rescue_rate": round(repair_rescue_rate, 4) if repair_rescue_rate is not None else None,
            "rescued_rsim_mean": _mean(rescued_rsim),
        },
        "revision": {
            "revision_n": revision_n,
            "SF_n": sf_n,
            "SS_n": ss_n,
            "revision_break_compile_rate": (
                round(revision_break_compile_rate, 4)
                if revision_break_compile_rate is not None
                else None
            ),
        },
        "revision_ss": {
            "revision_ss_n": len(ss_deltas),
            "revision_ss_s0_rsim_mean": _mean(ss_s0),
            "revision_ss_s1_rsim_mean": _mean(ss_s1),
            "revision_ss_delta_mean": _mean(ss_deltas),
            "revision_ss_delta_median": _median(ss_deltas),
            "revision_ss_improve_rate": round(improve / len(ss_deltas), 4) if ss_deltas else None,
            "revision_ss_regress_rate": round(regress / len(ss_deltas), 4) if ss_deltas else None,
            "revision_ss_neutral_rate": round(neutral / len(ss_deltas), 4) if ss_deltas else None,
            "revision_ss_positive_delta_mean": _mean(pos),
            "revision_ss_negative_delta_mean": _mean(neg),
            "percentiles": _percentiles(ss_deltas),
        },
        "buckets": buckets,
        "supplemental": supplemental,
    }


def write_predictions(policy_root: Path, rows: list[dict[str, Any]]) -> Path:
    out = policy_root / "predictions.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    return out


def build_comparison(
    metrics: dict[str, dict[str, Any]], p0: str, p1: str
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    keys = [
        "stage0_compile_rate",
        "stage1_compile_rate",
        "stage0_render_rate",
        "stage1_render_rate",
        "overall_delta_fixed_zero",
    ]
    for k in keys:
        rows.append(
            {
                "metric": k,
                p0: metrics[p0].get(k),
                p1: metrics[p1].get(k),
                "delta": _delta(metrics[p0].get(k), metrics[p1].get(k)),
            }
        )
    nested = [
        ("repair", "repair_rescue_rate"),
        ("repair", "rescued_rsim_mean"),
        ("revision", "revision_break_compile_rate"),
        ("revision_ss", "revision_ss_delta_mean"),
        ("revision_ss", "revision_ss_delta_median"),
        ("revision_ss", "revision_ss_improve_rate"),
        ("revision_ss", "revision_ss_regress_rate"),
    ]
    for section, key in nested:
        a = metrics[p0].get(section, {}).get(key)
        b = metrics[p1].get(section, {}).get(key)
        rows.append({"metric": f"{section}.{key}", p0: a, p1: b, "delta": _delta(a, b)})
    for m in ("dino", "dreamsim", "clipimg"):
        for key in (
            "stage0_fixed_zero_mean",
            "stage1_fixed_zero_mean",
            "ss_delta_mean",
            "ss_improve_rate",
            "ss_regress_rate",
        ):
            a = metrics[p0].get("supplemental", {}).get(m, {}).get(key)
            b = metrics[p1].get("supplemental", {}).get(m, {}).get(key)
            rows.append(
                {
                    "metric": f"supplemental.{m}.{key}",
                    p0: a,
                    p1: b,
                    "delta": _delta(a, b),
                }
            )
    rows.append(
        {
            "metric": "revision_ss_n",
            p0: metrics[p0].get("revision_ss", {}).get("revision_ss_n"),
            p1: metrics[p1].get("revision_ss", {}).get("revision_ss_n"),
            "delta": None,
        }
    )
    return {"rows": rows}


def _delta(a: Any, b: Any) -> float | None:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return round(float(b) - float(a), 4)
    return None


def write_comparison_files(
    eval_root: Path, comparison: dict[str, Any], policies: tuple[str, str]
) -> None:
    summary_dir = eval_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    p0, p1 = policies
    json_path = summary_dir / f"comparison_{p0}_vs_{p1}.json"
    json_path.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    csv_path = summary_dir / f"comparison_{p0}_vs_{p1}.csv"
    with csv_path.open("w", encoding="utf-8") as fh:
        fh.write(f"metric,{p0},{p1},delta\n")
        for row in comparison["rows"]:
            fh.write(
                f"{row['metric']},{row[p0]},{row[p1]},{row['delta']}\n"
            )
    print(f"[finalizer] comparison files: {json_path} / {csv_path}", flush=True)


def _img_rel(eval_root: Path, path: str | None) -> str:
    if not path:
        return ""
    try:
        return str(Path(path).resolve().relative_to(eval_root.resolve()))
    except Exception:
        return ""


def build_gallery(
    eval_root: Path,
    results: dict[str, list[dict[str, Any]]],
    manifest_order: list[str],
    policies: tuple[str, str],
) -> Path:
    per_sample: dict[str, dict[str, Any]] = {}
    for policy in policies:
        for r in results.get(policy, []):
            per_sample.setdefault(str(r["sample_id"]), {})[policy] = r
    rows = []
    for sid in manifest_order:
        if sid not in per_sample:
            continue
        entry: dict[str, Any] = {"sample_id": sid}
        for policy in policies:
            r = per_sample[sid].get(policy)
            if r is None:
                entry[policy] = None
                continue
            s0 = r.get("stage0") or {}
            s1 = r.get("stage1") or {}
            entry[policy] = {
                "route": r.get("route"),
                "transition": r.get("transition"),
                "s0_render": _img_rel(eval_root, s0.get("render_path")),
                "s1_render": _img_rel(eval_root, s1.get("render_path")),
                "s0_rsim": s0.get("rsim"),
                "s1_rsim": s1.get("rsim"),
                "delta": (s1.get("rsim") - s0.get("rsim")) if isinstance(s1.get("rsim"), (int, float)) and isinstance(s0.get("rsim"), (int, float)) else None,
                "s0_compile": s0.get("compile_success"),
                "s1_compile": s1.get("compile_success"),
            }
        rows.append(entry)

    gt_dir = os.environ.get("WF_GT_IMAGE_DIR", "")
    cards = []
    for entry in rows:
        sid = entry["sample_id"]
        gt_rel = f"{gt_dir}/{sid}.png"
        cells = ""
        for policy in policies:
            p = entry[policy]
            if p is None:
                cells += f'<td class="missing" colspan="4">no result</td>'
                continue
            cells += (
                f'<td><a href="{p["s0_render"]}" target="_blank"><img src="{p["s0_render"]}" '
                f'class="thumb" title="S0 rsim={_fmt(p["s0_rsim"])} compile={p["s0_compile"]}"></a>'
                f'<div class="cap">S0 {_fmt(p["s0_rsim"])}</div></td>'
                f'<td><a href="{p["s1_render"]}" target="_blank"><img src="{p["s1_render"]}" '
                f'class="thumb" title="S1 rsim={_fmt(p["s1_rsim"])} compile={p["s1_compile"]}"></a>'
                f'<div class="cap">S1 {_fmt(p["s1_rsim"])}</div></td>'
                f'<td class="num">{_fmt(p["delta"])}</td>'
                f'<td class="trans">{p["transition"]}</td>'
            )
        p0, p1 = policies
        card = (
            f'<div class="card" data-sample="{sid}" '
            f'data-p1-delta="{_js(entry.get(p1, {}).get("delta"))}" '
            f'data-p0-delta="{_js(entry.get(p0, {}).get("delta"))}" '
            f'data-p1-trans="{entry.get(p1, {}).get("transition")}" '
            f'data-p0-trans="{entry.get(p0, {}).get("transition")}" '
            f'data-p1-s0="{_js(entry.get(p1, {}).get("s0_rsim"))}">'
            f'<h3>{sid}</h3>'
            f'<table><tr><th></th>'
            f'<th>{p0} S0</th><th>{p0} S1</th><th>Δ</th><th>T</th>'
            f'<th>{p1} S0</th><th>{p1} S1</th><th>Δ</th><th>T</th></tr>'
            f'<tr><td><a href="{gt_rel}" target="_blank"><img src="{gt_rel}" class="thumb" '
            f'title="GT"></a><div class="cap">GT</div></td>'
            f'{cells}</tr></table></div>'
        )
        cards.append(card)

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Workflow-GRPO Evaluation Gallery</title>
<style>
body {{ font-family: sans-serif; margin: 16px; }}
.controls {{ margin-bottom: 12px; }}
.card {{ border: 1px solid #ccc; border-radius: 8px; padding: 8px; margin-bottom: 12px; }}
.card h3 {{ margin: 4px 0; font-size: 14px; }}
table {{ border-collapse: collapse; }}
td, th {{ border: 1px solid #ddd; padding: 3px; text-align: center; font-size: 12px; }}
img.thumb {{ width: 96px; height: 96px; object-fit: contain; background: #fff; }}
.cap {{ font-size: 11px; color: #555; }}
.trans {{ font-weight: bold; }}
.missing {{ color: #999; }}
</style></head><body>
<h1>Workflow-GRPO evaluation — {p0} vs {p1}</h1>
<div class="controls">
<label>Sort / filter: <select id="sort">
  <option value="manifest">manifest order</option>
  <option value="ss-improve">largest SS improvement ({p1})</option>
  <option value="ss-regress">largest SS regression ({p1})</option>
  <option value="fs-rescue">FS rescue</option>
  <option value="sf-break">SF break</option>
  <option value="high-s0-regress">high-S0 regression</option>
</select></label>
</div>
<div id="gallery">{''.join(cards)}</div>
<script>
const EPS = 0.005;
function num(el, attr) {{ const v = el.getAttribute(attr); return v === "null" || v === null ? null : parseFloat(v); }}
function sortCards(mode) {{
  const cards = Array.from(document.querySelectorAll('.card'));
  const cmp = (a, b) => 0;
  const sorters = {{
    'manifest': (a, b) => 0,
    'ss-improve': (a, b) => (num(b, 'data-p1-delta') ?? -1e9) - (num(a, 'data-p1-delta') ?? -1e9),
    'ss-regress': (a, b) => (num(a, 'data-p1-delta') ?? 1e9) - (num(b, 'data-p1-delta') ?? 1e9),
    'fs-rescue': (a, b) => (a.getAttribute('data-p1-trans') === 'FS' ? 0 : 1) - (b.getAttribute('data-p1-trans') === 'FS' ? 0 : 1),
    'sf-break': (a, b) => (a.getAttribute('data-p1-trans') === 'SF' ? 0 : 1) - (b.getAttribute('data-p1-trans') === 'SF' ? 0 : 1),
    'high-s0-regress': (a, b) => {{
      const fa = (num(a, 'data-p1-s0') ?? 0) >= 0.7 && (num(a, 'data-p1-delta') ?? 0) < -EPS;
      const fb = (num(b, 'data-p1-s0') ?? 0) >= 0.7 && (num(b, 'data-p1-delta') ?? 0) < -EPS;
      return (fa ? 0 : 1) - (fb ? 0 : 1) || (num(a, 'data-p1-delta') ?? 0) - (num(b, 'data-p1-delta') ?? 0);
    }}
  }};
  cards.sort(sorters[mode] || sorters['manifest']);
  const g = document.getElementById('gallery');
  cards.forEach(c => g.appendChild(c));
}}
document.getElementById('sort').addEventListener('change', e => sortCards(e.target.value));
</script>
</body></html>
"""
    out = eval_root / "summary" / f"gallery_{p0}_vs_{p1}.html"
    out.write_text(html, encoding="utf-8")
    print(f"[finalizer] gallery written: {out}", flush=True)
    return out


def _fmt(v: Any) -> str:
    return "—" if v is None else f"{float(v):.4f}"


def _js(v: Any) -> str:
    return "null" if v is None else f"{float(v):.6f}"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--eval-root", default=str(EVAL_ROOT))
    p.add_argument(
        "--policies",
        default=os.environ.get("WF_POLICY", ""),
        help="Comma-separated policy labels (directories holding results/); "
        "defaults to $WF_POLICY, or to every policy found under --eval-root.",
    )
    args = p.parse_args(argv)
    eval_root = Path(args.eval_root).resolve()
    policies = [x for x in args.policies.split(",") if x]
    if not policies:
        policies = discover_policies(eval_root)
    if not policies:
        raise SystemExit(
            f"No policy found under {eval_root} (expected a directory with results/). "
            "Pass --policies explicitly."
        )

    manifest_path = manifest_copy_path(eval_root)
    if manifest_path is None:
        raise FileNotFoundError(
            f"No manifest copy under {eval_root / '00_manifest'}; run the controller for this "
            "evaluation root first (it writes the manifest it used)."
        )
    manifest_order = [
        str(json.loads(line).get("sample_id") or json.loads(line).get("file_id"))
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    n = len(manifest_order)
    results: dict[str, list[dict[str, Any]]] = {}
    metrics: dict[str, dict[str, Any]] = {}
    for policy in policies:
        rows = read_results(eval_root / policy, manifest_order)
        results[policy] = rows
        metrics[policy] = policy_metrics(rows, n)
        pred_path = write_predictions(eval_root / policy, rows)
        summary_path = eval_root / "summary" / f"{policy}.summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(metrics[policy], ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (eval_root / policy).mkdir(parents=True, exist_ok=True)
        (eval_root / policy / "summary.json").write_text(
            json.dumps(metrics[policy], ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        done = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "policy": policy,
            "manifest": str(manifest_path),
            "n_samples": n,
            "results": len(rows),
            "duplicates": len(rows) - len({r["sample_id"] for r in rows}),
            "missing": n - len({r["sample_id"] for r in rows}),
            "predictions": str(pred_path),
            "summary": str(summary_path),
        }
        (eval_root / policy / "DONE").write_text(
            json.dumps(done, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"[finalizer] {policy}: results={len(rows)}/{n} "
            f"SS={metrics[policy]['transitions']['SS']} "
            f"SF={metrics[policy]['transitions']['SF']} "
            f"FS={metrics[policy]['transitions']['FS']} "
            f"FF={metrics[policy]['transitions']['FF']}",
            flush=True,
        )

    if len(policies) == 2:
        p0, p1 = policies[0], policies[1]
        comparison = build_comparison(metrics, p0, p1)
        write_comparison_files(eval_root, comparison, (p0, p1))
        build_gallery(eval_root, results, manifest_order, (p0, p1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
