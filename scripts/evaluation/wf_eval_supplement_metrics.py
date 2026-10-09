"""Supplemental DINO / DreamSim / CLIPImg scoring for completed workflow
evaluations (Dev-150 model selection or held-out Final-945 reporting).

Reuses the already-rendered Stage0/Stage1 PNGs (no regeneration, no
re-render).  Four resident per-GPU MetricSuite workers (one per GPU) consume a
filesystem request queue; results are merged back into each policy's
``results/<sample>.json`` (atomic rewrite) so the finalizer can report the
three extra metrics with the same render-only / fixed-zero / SS-conditional
semantics as RSim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from wf_eval_common import PYTHON, atomic_json, manifest_copy_path  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
METRIC_NAMES = ("dino", "dreamsim", "clipimg")


def manifest_gt_map(eval_root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    manifest = manifest_copy_path(eval_root)
    if manifest is None:
        return out
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        sid = str(row.get("sample_id") or row.get("file_id"))
        gt = str(row.get("gt_image_path") or row.get("asset_image_path") or row.get("resolved_image_path") or "")
        out[sid] = gt
    return out


def build_tasks(
    eval_root: Path, policies: list[str], gt_map: dict[str, str]
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for policy in policies:
        results_dir = eval_root / policy / "results"
        if not results_dir.exists():
            continue
        for fp in sorted(results_dir.glob("*.json")):
            try:
                row = json.loads(fp.read_text(encoding="utf-8"))
            except Exception:
                continue
            sid = str(row.get("sample_id"))
            gt = gt_map.get(sid)
            if not gt or not Path(gt).exists():
                continue
            for stage in ("stage0", "stage1"):
                st = row.get(stage) or {}
                if not (st.get("render_success") and st.get("render_path")):
                    continue
                if Path(st["render_path"]).exists() and st.get("metrics_status") == "success":
                    continue  # already scored
                if not Path(st["render_path"]).exists():
                    continue
                request_id = f"{policy}:{sid}:{stage}"
                tasks.append(
                    {
                        "request_id": request_id,
                        "policy": policy,
                        "sample_id": sid,
                        "stage": stage,
                        "gt_path": gt,
                        "render_path": st["render_path"],
                        "result_path": str(fp),
                    }
                )
    return tasks


def _gpu_for(request_id: str, gpus: list[int]) -> int:
    h = int(hashlib.sha1(request_id.encode("utf-8")).hexdigest()[:8], 16)
    return gpus[h % len(gpus)]


def spawn_worker(gpu: int, queue_dir: Path, logs: Path, env: dict[str, str]) -> subprocess.Popen:
    log_path = logs / f"metrics_gpu{gpu}.log"
    fh = open(log_path, "ab", buffering=0)
    cmd = [
        PYTHON,
        "-u",
        str(REPO / "scripts" / "evaluation" / "wf_eval_metric_worker.py"),
        "--queue-dir",
        str(queue_dir),
        "--heartbeat-path",
        str(logs / f"metrics_gpu{gpu}.heartbeat.json"),
        "--ready-path",
        str(logs / f"metrics_gpu{gpu}.ready"),
        "--stop-file",
        str(queue_dir.parent / "metrics.stop"),
        "--gpu",
        str(gpu),
        "--device",
        "cuda:0",
    ]
    return subprocess.Popen(
        cmd,
        stdout=fh,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        env=env,
        start_new_session=True,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--eval-root", required=True)
    p.add_argument("--policies", required=True)
    p.add_argument("--gpus", default="0,1,2,3")
    args = p.parse_args(argv)
    eval_root = Path(args.eval_root).resolve()
    policies = [x for x in args.policies.split(",") if x]
    gpus = [int(x) for x in args.gpus.split(",") if x]

    queue_root = eval_root / "controller" / "metrics_queue"
    logs = eval_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    for gpu in gpus:
        (queue_root / f"gpu{gpu}").mkdir(parents=True, exist_ok=True)
    stop_file = queue_root / "metrics.stop"
    try:
        stop_file.unlink(missing_ok=True)
    except Exception:
        pass

    gt_map = manifest_gt_map(eval_root)
    tasks = build_tasks(eval_root, policies, gt_map)
    if not tasks:
        print(f"[supplement] no pending metric tasks for {eval_root} {policies}", flush=True)
        return 0
    print(f"[supplement] tasks={len(tasks)} policies={policies} gpus={gpus}", flush=True)

    env = dict(os.environ)
    env["PYTHONPATH"] = f"{REPO}/src" + (
        f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else ""
    )
    procs: dict[int, subprocess.Popen] = {}
    for gpu in gpus:
        proc = spawn_worker(gpu, queue_root / f"gpu{gpu}", logs, env)
        procs[gpu] = proc
        print(f"[supplement] started metric worker gpu={gpu} pid={proc.pid}", flush=True)

    deadline = time.time() + 600
    ready = {g: False for g in gpus}
    while time.time() < deadline and not all(ready.values()):
        for gpu in gpus:
            if (logs / f"metrics_gpu{gpu}.ready").exists():
                ready[gpu] = True
        time.sleep(2)
    if not all(ready.values()):
        print("[supplement] FATAL metric workers not ready in time", flush=True)
        for proc in procs.values():
            try:
                proc.kill()
            except Exception:
                pass
        return 1
    print("[supplement] all metric workers ready", flush=True)

    for task in tasks:
        req = dict(task)
        req["created_at"] = time.time()
        gpu = _gpu_for(task["request_id"], gpus)
        qdir = queue_root / f"gpu{gpu}"
        req_path = qdir / f"req_{task['request_id'].replace(':', '_')}.json"
        tmp = req_path.with_name(f".{req_path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}")
        tmp.write_text(json.dumps(req, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, req_path)

    expected = {t["request_id"]: t for t in tasks}
    merged = 0
    failed = 0
    result_deadline = time.time() + 1800
    while expected and time.time() < result_deadline:
        remaining = list(expected.keys())
        for gpu in gpus:
            for res_path in sorted((queue_root / f"gpu{gpu}").glob("res_*.json")):
                escaped = res_path.name[len("res_"):-len(".json")]
                matched = None
                for rid in remaining:
                    if rid.replace(":", "_") == escaped:
                        matched = rid
                        break
                if matched is None:
                    continue
                try:
                    data = json.loads(res_path.read_text(encoding="utf-8"))
                except Exception:
                    time.sleep(0.2)
                    continue
                task = expected.pop(matched)
                res_path.unlink(missing_ok=True)
                if data.get("status") != "success":
                    failed += 1
                    print(
                        f"[supplement] FAILED {task['policy']} {task['sample_id']} "
                        f"stage={task['stage']} {data.get('error')}",
                        flush=True,
                    )
                    continue
                result_path = Path(task["result_path"])
                row = json.loads(result_path.read_text(encoding="utf-8"))
                st = row.get(task["stage"])
                if not isinstance(st, dict):
                    st = {}
                    row[task["stage"]] = st
                for m in METRIC_NAMES:
                    st[m] = float(data["scores"].get(m, 0.0))
                st["metrics_status"] = "success"
                st["metrics_elapsed"] = data.get("score_elapsed_s")
                atomic_json(result_path, row)
                merged += 1
        time.sleep(1)

    stop_file.write_text(str(time.time()) + "\n", encoding="utf-8")
    time.sleep(3)
    for proc in procs.values():
        if proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass
    time.sleep(3)
    for proc in procs.values():
        if proc.poll() is None:
            try:
                proc.kill()
            except Exception:
                pass

    print(
        f"[supplement] done merged={merged} failed={failed} remaining={len(expected)}",
        flush=True,
    )
    return 0 if merged > 0 and not expected else 2


if __name__ == "__main__":
    raise SystemExit(main())
