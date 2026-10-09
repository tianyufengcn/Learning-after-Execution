from __future__ import annotations

import json
import os
import shutil
import threading
import time
import random
import uuid

import torch
import torch.distributed as dist
from pathlib import Path
from typing import Any


from .transport import (
    assign_group_credit,
    balanced_fixed_count_partition,
    read_record,
    trajectory_to_record,
    write_record_atomic,
)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Unique tmp name per write: the heartbeat thread and the main process share
    # the same pid, so a pid-only suffix would race (one os.replace moving the
    # other's tmp away -> FileNotFoundError).
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


class Heartbeat:
    def __init__(self, path: Path, interval_s: float) -> None:
        self.path = path
        self.interval_s = max(1.0, float(interval_s))
        self._lock = threading.Lock()
        self._io_lock = threading.Lock()
        self._state: dict[str, Any] = {"phase": "initializing"}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"heartbeat-{os.getpid()}", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def update(self, **state: Any) -> None:
        with self._lock:
            self._state.update(state)
        self._write()

    def _write(self) -> None:
        with self._io_lock:
            with self._lock:
                payload = dict(self._state)
            payload.update({"time": time.time(), "pid": os.getpid()})
            _atomic_json(self.path, payload)

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            try:
                self._write()
            except Exception:
                pass

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=max(2.0, self.interval_s + 1.0))
        try:
            self._write()
        except Exception:
            pass


class DistributedTrajectoryCoordinator:
    """Shared-filesystem control plane for global rollout work stealing.

    Long, stochastic rollout waits never enter an NCCL collective. Every rank
    atomically claims trajectory tasks from one global queue. Once all results
    exist, ranks host-wait on ready markers and only then enter a short DDP
    synchronization before backward.
    """

    def __init__(
        self,
        accelerator,
        rollout_engine,
        processor,
        workflow_cfg: dict[str, Any],
        *,
        output_dir: str | Path,
        loss_microbatch: int,
    ) -> None:
        self.accelerator = accelerator
        self.rollout_engine = rollout_engine
        self.processor = processor
        self.rank = int(accelerator.process_index)
        self.world_size = int(accelerator.num_processes)
        self.group_size = int(workflow_cfg.get("num_generations", 8))
        self.reward_scale = float(workflow_cfg.get("reward_scale", 10.0))
        dcfg = workflow_cfg.get("distributed_rollout", {})
        self.enabled = bool(dcfg.get("enabled", True)) and self.world_size > 1
        self.claim_batch_size = int(dcfg.get("claim_batch_size", 4))
        self.poll_interval_s = float(dcfg.get("poll_interval_s", 0.5))
        self.heartbeat_interval_s = float(dcfg.get("heartbeat_interval_s", 10.0))
        self.heartbeat_stale_s = float(dcfg.get("heartbeat_stale_s", 3600.0))
        self.cleanup_completed_steps = bool(dcfg.get("cleanup_completed_steps", True))
        self.loss_microbatch = int(loss_microbatch)
        base = Path(dcfg.get("control_dir") or (Path(output_dir).parent / "workflow_control"))

        # One short launch-time NCCL broadcast establishes a fresh namespace.
        # No long rollout wait uses NCCL; this happens while all ranks are in
        # trainer initialization and completes immediately.
        nonce = random.randrange(1, 2**62) if self.accelerator.is_main_process else 0
        token = torch.tensor([nonce], dtype=torch.int64, device=self.accelerator.device)
        if dist.is_initialized() and self.world_size > 1:
            dist.broadcast(token, src=0)
        self.run_id = f"{int(token.item()):016x}"
        self.run_root = base / f"run_{self.run_id}"
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.heartbeat = Heartbeat(
            self.run_root / f"heartbeat_rank{self.rank}.json",
            self.heartbeat_interval_s,
        )
        self.heartbeat.start()
        self.global_log = Path(workflow_cfg.get("log_dir", Path(output_dir).parent / "rollout_logs")) / "workflow_rollouts_global.jsonl"

    def _step_root(self, global_step: int) -> Path:
        return self.run_root / f"step_{int(global_step):06d}"

    def _wait_for(self, predicate, description: str) -> None:
        while not predicate():
            self._check_heartbeats(description)
            time.sleep(self.poll_interval_s)

    def _check_heartbeats(self, description: str) -> None:
        now = time.time()
        stale: list[tuple[int, float]] = []
        for rank in range(self.world_size):
            path = self.run_root / f"heartbeat_rank{rank}.json"
            if not path.exists():
                continue
            age = now - path.stat().st_mtime
            if age > self.heartbeat_stale_s:
                stale.append((rank, age))
        if stale:
            detail = ", ".join(f"rank{rank}:{age:.0f}s" for rank, age in stale)
            raise RuntimeError(f"Stale rollout heartbeat while waiting for {description}: {detail}")

    def prepare_tasks(self, rows: list[dict[str, Any]], global_step: int) -> tuple[Path, int]:
        step_root = self._step_root(global_step)
        manifest = step_root / "manifest.json"
        if self.accelerator.is_main_process:
            if step_root.exists():
                shutil.rmtree(step_root)
            for name in ("pending", "running", "done", "update_ready"):
                (step_root / name).mkdir(parents=True, exist_ok=True)
            tasks: list[dict[str, Any]] = []
            for root_index, row in enumerate(rows):
                sample_id = str(row["sample_id"])
                gt_image_path = str(row.get("gt_image_path") or row.get("image"))
                for rollout_index in range(self.group_size):
                    task_id = f"r{root_index:04d}_g{rollout_index:02d}"
                    task = {
                        "task_id": task_id,
                        "root_index": root_index,
                        "rollout_index": rollout_index,
                        "sample_id": sample_id,
                        "gt_image_path": gt_image_path,
                    }
                    tasks.append(task)
                    _atomic_json(step_root / "pending" / f"{task_id}.json", task)
            _atomic_json(
                manifest,
                {
                    "global_step": int(global_step),
                    "group_size": self.group_size,
                    "root_count": len(rows),
                    "task_count": len(tasks),
                    "created_at": time.time(),
                },
            )
        self._wait_for(manifest.exists, f"step {global_step} manifest")
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        return step_root, int(payload["task_count"])

    def _claim_one(self, step_root: Path) -> tuple[Path, dict[str, Any]] | None:
        for pending in sorted((step_root / "pending").glob("*.json")):
            claimed = step_root / "running" / f"{pending.stem}.rank{self.rank}.json"
            try:
                os.rename(pending, claimed)
            except FileNotFoundError:
                continue
            except OSError:
                continue
            task = json.loads(claimed.read_text(encoding="utf-8"))
            return claimed, task
        return None

    def claim_batch(self, step_root: Path) -> list[tuple[Path, dict[str, Any]]]:
        batch: list[tuple[Path, dict[str, Any]]] = []
        while len(batch) < self.claim_batch_size:
            item = self._claim_one(step_root)
            if item is None:
                break
            batch.append(item)
        return batch

    def _done_count(self, step_root: Path) -> int:
        return sum(1 for _ in (step_root / "done").glob("*.json.gz"))

    def run_global_rollout(self, rows: list[dict[str, Any]], global_step: int) -> tuple[list[dict[str, Any]], dict[str, float]]:
        if not self.enabled:
            trajectories, metrics = self.rollout_engine.rollout(rows, global_step)
            records = [
                trajectory_to_record(t, root_index=i // self.group_size, task_id=f"local_{i}", worker_rank=self.rank)
                for i, t in enumerate(trajectories)
            ]
            return records, metrics

        step_started = time.monotonic()
        step_root, task_count = self.prepare_tasks(rows, global_step)
        generated_here = 0
        local_rollout_wall = 0.0
        local_stage0_generation = 0.0
        local_stage1_generation = 0.0

        while self._done_count(step_root) < task_count:
            claimed = self.claim_batch(step_root)
            if not claimed:
                self.heartbeat.update(global_step=global_step, phase="waiting_for_work_or_results")
                time.sleep(self.poll_interval_s)
                self._check_heartbeats(f"step {global_step} rollout results")
                continue

            tasks = [task for _, task in claimed]
            task_ids = [str(task["task_id"]) for task in tasks]
            self.heartbeat.update(global_step=global_step, phase="rollout_batch", task_ids=task_ids)
            trajectories, metrics = self.rollout_engine.rollout_tasks(
                tasks,
                global_step,
                assign_credit=False,
                log_trajectories=False,
            )
            if len(trajectories) != len(tasks):
                raise RuntimeError("rollout task/result length mismatch")
            local_rollout_wall += float(metrics.get("rollout_wall_s", 0.0))
            local_stage0_generation += float(metrics.get("stage0_generation_s", 0.0))
            local_stage1_generation += float(metrics.get("stage1_generation_s", 0.0))
            for (claim_path, task), trajectory in zip(claimed, trajectories, strict=True):
                record = trajectory_to_record(
                    trajectory,
                    root_index=int(task["root_index"]),
                    task_id=str(task["task_id"]),
                    worker_rank=self.rank,
                )
                write_record_atomic(step_root / "done" / f"{task['task_id']}.json.gz", record)
                claim_path.unlink(missing_ok=True)
                generated_here += 1

        self._wait_for(lambda: self._done_count(step_root) == task_count, f"step {global_step} all trajectory results")
        records = [read_record(path) for path in sorted((step_root / "done").glob("*.json.gz"))]
        if len(records) != task_count:
            raise RuntimeError(f"Expected {task_count} results, found {len(records)}")
        assign_group_credit(records, self.group_size, self.reward_scale)
        self.heartbeat.update(global_step=global_step, phase="rollout_complete", generated_here=generated_here)
        return records, {
            "rollout_wall_s": time.monotonic() - step_started,
            "local_claimed_trajectories": float(generated_here),
            "local_rollout_compute_s": local_rollout_wall,
            "stage0_generation_s": local_stage0_generation,
            "stage1_generation_s": local_stage1_generation,
        }

    def partition_for_backward(
        self, records: list[dict[str, Any]], stage: int
    ) -> list[dict[str, Any]]:
        bins = balanced_fixed_count_partition(records, self.world_size, stage)
        local = bins[self.rank]
        if len(local) % self.loss_microbatch != 0:
            raise RuntimeError(
                f"rank{self.rank} gets {len(local)} stage{stage} actions; not divisible by loss microbatch {self.loss_microbatch}"
            )
        return local

    def wait_update_ready(self, global_step: int) -> None:
        if not self.enabled:
            self.accelerator.wait_for_everyone()
            return
        step_root = self._step_root(global_step)
        _atomic_json(
            step_root / "update_ready" / f"rank{self.rank}.json",
            {"rank": self.rank, "time": time.time(), "global_step": int(global_step)},
        )
        self.heartbeat.update(global_step=global_step, phase="update_ready")
        self._wait_for(
            lambda: sum(1 for _ in (step_root / "update_ready").glob("rank*.json")) == self.world_size,
            f"step {global_step} update-ready markers",
        )
        # All ranks are now known to be close to the update boundary, so this
        # NCCL barrier is short and cannot be held open by a 30-minute rollout.
        self.accelerator.wait_for_everyone()

    def finish_step(self, global_step: int) -> None:
        self.heartbeat.update(global_step=global_step, phase="step_complete")
        self.accelerator.wait_for_everyone()
        if self.enabled and self.cleanup_completed_steps and self.accelerator.is_main_process:
            shutil.rmtree(self._step_root(global_step), ignore_errors=True)
        self.accelerator.wait_for_everyone()

    def write_global_log(self, records: list[dict[str, Any]], global_step: int) -> None:
        if not self.accelerator.is_main_process:
            return
        self.global_log.parent.mkdir(parents=True, exist_ok=True)
        by_root: dict[int, list[dict[str, Any]]] = {}
        with open(self.global_log, "a", encoding="utf-8") as handle:
            for record in sorted(records, key=lambda row: (int(row["root_index"]), int(row["rollout_index"]))):
                by_root.setdefault(int(record["root_index"]), []).append(record)
                if getattr(self.rollout_engine, "persist_artifacts", False):
                    safe_sid = self.rollout_engine._safe_sample_id(str(record["sample_id"]))
                    artifact_root = (
                        self.rollout_engine.artifact_root
                        / f"step_{int(global_step):06d}"
                        / safe_sid
                        / f"rollout_{int(record['rollout_index']):02d}"
                    )
                    artifact_root.mkdir(parents=True, exist_ok=True)
                    stage0_tex = artifact_root / "stage0.tex"
                    stage1_tex = artifact_root / "stage1.tex"
                    stage0_tex.write_text(str(record["stage0"]["text"]), encoding="utf-8")
                    stage1_tex.write_text(str(record["stage1"]["text"]), encoding="utf-8")
                    record["stage0_tex_path"] = str(stage0_tex)
                    record["stage1_tex_path"] = str(stage1_tex)
                row = {
                    "event": "trajectory",
                    "time": time.time(),
                    "global_step": int(global_step),
                    **record,
                }
                # Keep logs compact: token id arrays are recoverable from the
                # control-plane result during the step but need not live forever.
                row["stage0"] = {k: v for k, v in record["stage0"].items() if k != "completion_ids"}
                row["stage1"] = {k: v for k, v in record["stage1"].items() if k != "completion_ids"}
                handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            for root_index, group in sorted(by_root.items()):
                handle.write(
                    json.dumps(
                        {
                            "event": "group_summary",
                            "time": time.time(),
                            "global_step": int(global_step),
                            "root_index": root_index,
                            "sample_id": group[0]["sample_id"],
                            "group_size": len(group),
                            "stage0_rsim": [r["stage0_score"] for r in group],
                            "stage1_rsim": [r["stage1_score"] for r in group],
                            "delta_rsim": [r["delta_score"] for r in group],
                            "stage0_compile": [r["stage0_ok"] for r in group],
                            "stage1_compile": [r["stage1_ok"] for r in group],
                            "worker_ranks": [r["worker_rank"] for r in group],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            handle.flush()

    def close(self) -> None:
        self.heartbeat.close()
