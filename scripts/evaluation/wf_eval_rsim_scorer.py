"""Resident per-GPU vision-only RSim-v2 scorer service for the workflow
Workflow-GRPO evaluation.

One process per GPU, shared by the 3 Qwen workers resident on that GPU.  The
DeTikZify SigLIP vision-only model is loaded exactly once per process; score
requests arrive as atomic JSON files under a per-GPU queue directory and
results are written back as ``res_<request_id>.json``.  GT features are cached
persistently (same key scheme as the v0.3.0 training RSimV2Scorer) so Stage0
and Stage1 of the same sample never re-encode the ground-truth image.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from wf_eval_common import RSIM_CONFIG  # noqa: E402


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Unique tmp per write: pid-only suffixes race inside one process.
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _heartbeat_loop(heartbeat_path: Path, gpu: int, state: list[dict[str, Any]], stop: threading.Event) -> None:
    while not stop.wait(10):
        try:
            payload = dict(state[0])
            payload.update({"time": time.time(), "pid": os.getpid(), "gpu": gpu})
            _atomic_json(heartbeat_path, payload)
        except Exception:
            pass


def _scorer_ready(scorer) -> None:
    print(f"[rsim-scorer gpu={scorer['gpu']}] loaded model_path={scorer['model_path']} "
          f"load_mode={scorer['scorer'].load_mode}", flush=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--queue-dir", required=True)
    p.add_argument("--model-path", default=RSIM_CONFIG["model_path"])
    p.add_argument("--feature-cache-dir", required=True)
    p.add_argument("--heartbeat-path", required=True)
    p.add_argument("--ready-path", required=True)
    p.add_argument("--stop-file", default=None)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--poll-seconds", type=float, default=0.5)
    p.add_argument("--cpu-threads", type=int, default=8)
    args = p.parse_args(argv)

    import torch

    torch.set_num_threads(max(1, int(args.cpu_threads)))
    queue_dir = Path(args.queue_dir)
    queue_dir.mkdir(parents=True, exist_ok=True)
    ready_path = Path(args.ready_path)
    heartbeat_path = Path(args.heartbeat_path)
    stop_file = Path(args.stop_file) if args.stop_file else None
    started_at = time.time()

    try:
        from i2t_workflow_grpo.rewards.rsim import RSimV2Scorer

        scorer = RSimV2Scorer(
            model_path=args.model_path,
            cache_dir=args.feature_cache_dir,
            detikzify_repo=None,
            batch_size=RSIM_CONFIG["score_batch_size"],
            mem_cache_size=RSIM_CONFIG["score_mem_cache_size"],
            emd_workers=RSIM_CONFIG["emd_workers"],
            device=args.device,
            require_vision_only=RSIM_CONFIG["require_vision_only"],
        )
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            used = round((total - free) / 2**30, 2)
        else:
            used = 0.0
        state: list[dict[str, Any]] = [
            {"phase": "ready", "queue_size": 0, "scored": 0, "gpu_used_gib": used}
        ]
        _atomic_json(ready_path, {"gpu": args.gpu, "pid": os.getpid(), "ready_at": time.time()})
        stop = threading.Event()
        threading.Thread(
            target=_heartbeat_loop, args=(heartbeat_path, args.gpu, state, stop), daemon=True
        ).start()
        print(
            f"[rsim-scorer gpu={args.gpu}] ready model=vision_only gpu_used_gib={used} "
            f"queue={queue_dir}",
            flush=True,
        )

        processed = 0
        while True:
            reqs = sorted(queue_dir.glob("req_*.json"), key=lambda p: p.stat().st_mtime)
            for req_path in reqs:
                request_id = req_path.name[len("req_"):-len(".json")]
                res_path = queue_dir / f"res_{request_id}.json"
                try:
                    req = json.loads(req_path.read_text(encoding="utf-8"))
                    t0 = time.monotonic()
                    if not req.get("render_path") or not Path(req["render_path"]).exists():
                        raise FileNotFoundError(f"render missing: {req.get('render_path')}")
                    if not req.get("gt_path") or not Path(req["gt_path"]).exists():
                        raise FileNotFoundError(f"gt missing: {req.get('gt_path')}")
                    scores = scorer.score_paths([req["render_path"]], [req["gt_path"]])
                    rsim = float(scores[0]) if scores else 0.0
                    elapsed = round(time.monotonic() - t0, 3)
                    _atomic_json(
                        res_path,
                        {
                            "request_id": request_id,
                            "sample_id": req.get("sample_id"),
                            "stage": req.get("stage"),
                            "policy": req.get("policy"),
                            "worker_id": req.get("worker_id"),
                            "rsim": rsim,
                            "score_elapsed_s": elapsed,
                            "status": "success",
                            "error": None,
                        },
                    )
                    processed += 1
                    state[0] = {
                        "phase": "scoring",
                        "queue_size": len(list(queue_dir.glob("req_*.json"))),
                        "scored": processed,
                        "gpu_used_gib": used,
                    }
                    print(
                        f"[rsim-scorer gpu={args.gpu}] {req.get('policy')} {req.get('sample_id')} "
                        f"stage={req.get('stage')} rsim={rsim:.4f} elapsed={elapsed}s",
                        flush=True,
                    )
                except Exception as exc:
                    _atomic_json(
                        res_path,
                        {
                            "request_id": request_id,
                            "status": "failed",
                            "error": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc()[-4000:],
                        },
                    )
                    print(
                        f"[rsim-scorer gpu={args.gpu}] ERROR req={request_id} {type(exc).__name__}: {exc}",
                        flush=True,
                    )
                finally:
                    try:
                        req_path.unlink(missing_ok=True)
                    except Exception:
                        pass
            if (
                stop_file
                and stop_file.exists()
                and stop_file.stat().st_mtime > started_at
                and not reqs
            ):
                print(f"[rsim-scorer gpu={args.gpu}] stop file present, queue drained; exiting processed={processed}", flush=True)
                break
            time.sleep(float(args.poll_seconds))
        stop.set()
        _atomic_json(
            heartbeat_path,
            {"phase": "stopped", "pid": os.getpid(), "gpu": args.gpu, "time": time.time()},
        )
        return 0
    except BaseException as exc:
        print(f"[rsim-scorer gpu={args.gpu}] FATAL {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
