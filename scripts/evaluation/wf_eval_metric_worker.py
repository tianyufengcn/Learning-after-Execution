"""Resident per-GPU DINO/DreamSim/CLIPImg scorer for supplemental metrics.

One process per GPU loads the validated ``MetricSuite`` (the same image-metric
protocol the reported evaluations use) exactly once and serves score requests through the same
atomic filesystem request/result queue used by the RSim scorer.  It never
loads RSim/DeTikZify and never touches the Qwen workers.
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

from i2t_workflow_grpo.eval_metrics import MetricSuite  # noqa: E402


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--queue-dir", required=True)
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
        suite = MetricSuite(device=args.device, enabled=("dino", "dreamsim", "clipimg"))
        suite.prepare()
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
            f"[metric-w gpu={args.gpu}] ready enabled={suite.enabled} gpu_used_gib={used} "
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
                    scores = suite.compare(req["gt_path"], req["render_path"])
                    elapsed = round(time.monotonic() - t0, 3)
                    _atomic_json(
                        res_path,
                        {
                            "request_id": request_id,
                            "policy": req.get("policy"),
                            "sample_id": req.get("sample_id"),
                            "stage": req.get("stage"),
                            "scores": scores,
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
                        f"[metric-w gpu={args.gpu}] {req.get('policy')} {req.get('sample_id')} "
                        f"stage={req.get('stage')} "
                        f"{ {k: round(v, 4) for k, v in scores.items()} } elapsed={elapsed}s",
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
                        f"[metric-w gpu={args.gpu}] ERROR req={request_id} {type(exc).__name__}: {exc}",
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
                print(f"[metric-w gpu={args.gpu}] stop file present; exiting processed={processed}", flush=True)
                break
            time.sleep(float(args.poll_seconds))
        stop.set()
        _atomic_json(
            heartbeat_path,
            {"phase": "stopped", "pid": os.getpid(), "gpu": args.gpu, "time": time.time()},
        )
        return 0
    except BaseException as exc:
        print(f"[metric-w gpu={args.gpu}] FATAL {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
