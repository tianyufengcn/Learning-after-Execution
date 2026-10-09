"""Controller for the Workflow-GRPO two-stage evaluation.

Drives one policy (a warm-start adapter or a trained Workflow-GRPO checkpoint)
over whichever JSONL split manifest is requested by ``--test-manifest``: the
Dev-150 split used for model selection or the held-out Final-945 split used for
reporting. Every sample count, progress total and completion condition is
derived from that manifest, never from a fixed number.

SQLite dynamic claims, heartbeat/stale recovery, resident workers, restart
watchdog and finalizer hooks, scaled to 4 GPUs x 3 resident Qwen workers = 12
logical workers plus 4 resident vision-only RSim scorers (one per GPU).

The controller never loads a Qwen model and never performs inference.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from wf_eval_common import (  # noqa: E402
    BASE_MODEL,
    EVAL_ROOT,
    GENERATION,
    MANIFEST_COPY_NAME,
    PYTHON,
    RENDERER,
    RSIM_CONFIG,
    RSIM_MODEL,
    TEST_MANIFEST,
    TEXLIVE_BIN,
    atomic_json,
    load_manifest,
    policy_adapter,
)
from wf_eval_state import EvalState  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def gpu_memory_gib(device: int) -> tuple[float, float]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        lines = [l for l in out.strip().splitlines() if l.strip()]
        used, total = (float(x) for x in lines[device].split(","))
        return used / 1024.0, total / 1024.0
    except Exception:
        return 0.0, 0.0


class EvalController:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.eval_root = Path(args.eval_root).resolve()
        self.policy = args.policy
        # The manifest given on the command line is the single source of truth
        # for this run: every sample count and completion condition below is
        # derived from it.
        self.test_manifest = Path(args.test_manifest) if args.test_manifest else TEST_MANIFEST
        self.gpus = [int(x) for x in str(args.gpus).split(",") if x]
        self.replicas_per_gpu = int(args.replicas_per_gpu)
        self.smoke_limit = int(args.smoke_limit or 0)
        self.adapter = args.adapter or policy_adapter(self.policy)
        self.base_model = args.base_model or BASE_MODEL
        self.rsim_model = args.rsim_model or RSIM_MODEL

        self.logs = self.eval_root / "logs" / self.policy
        self.pids = self.eval_root / "pids" / self.policy
        self.controller_dir = self.eval_root / "controller"
        self.manifest_dir = self.eval_root / "00_manifest"
        self.contract_dir = self.eval_root / "run_contract"
        self.cache_dir = self.eval_root / "cache"
        self.render_cache = Path(args.render_cache_dir) if args.render_cache_dir else (self.cache_dir / "render_cache")
        self.render_tmp = Path(args.render_tmp_dir) if args.render_tmp_dir else (self.cache_dir / "render_tmp")
        self.gt_feats = Path(args.feature_cache_dir) if args.feature_cache_dir else (self.cache_dir / "gt_rsim_feats")
        for d in (
            self.logs,
            self.pids,
            self.controller_dir,
            self.manifest_dir,
            self.contract_dir,
            self.cache_dir,
            self.render_cache,
            self.render_tmp,
            self.gt_feats,
        ):
            d.mkdir(parents=True, exist_ok=True)

        # One SQLite DB per policy: two controllers run concurrently on the
        # same eval root (e.g. checkpoint50 on GPU0/1, checkpoint75 on GPU2/3)
        # and must never contend on a shared WAL journal.
        self.state_db = self.controller_dir / f"eval_state.{self.policy}.sqlite3"
        # Policy-scoped stop file so two policies running concurrently (e.g.
        # checkpoint50 on GPU0/1 and checkpoint75 on GPU2/3) never kill each
        # other's resident scorers.
        self.scorer_stop = self.controller_dir / f"scorer.stop.{self.policy}"
        # A stop file left by a previous policy must never kill this run's
        # freshly started scorers; remove it before spawning anything.
        try:
            self.scorer_stop.unlink(missing_ok=True)
        except Exception:
            pass
        self.start = time.time()

    # ------------------------------------------------------------------ logs
    def log(self, msg: str) -> None:
        print(f"[controller] {iso_now()} {msg}", flush=True)

    def log_err(self, msg: str) -> None:
        print(f"[controller][ERROR] {iso_now()} {msg}", flush=True)

    def env_for_gpu(self, gpu: int) -> dict[str, str]:
        env = dict(os.environ)
        env["PYTHONPATH"] = f"{REPO}/src" + (
            f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else ""
        )
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        # Children must agree with the controller about which split is running.
        env["WF_EVAL_ROOT"] = str(self.eval_root)
        env["WF_TEST_MANIFEST"] = str(self.test_manifest)
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env.setdefault(var, str(self.args.cpu_threads))
        return env

    def spawn(self, cmd: list[str], log_path: Path, env: dict[str, str]) -> subprocess.Popen:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(log_path, "ab", buffering=0)
        proc = subprocess.Popen(
            cmd,
            stdout=fh,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=env,
            start_new_session=True,
        )
        return proc

    # ------------------------------------------------------------ orchestrate
    def write_manifest_copy(self, rows: list[dict[str, Any]]) -> None:
        out = self.manifest_dir / MANIFEST_COPY_NAME
        with out.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.log(f"manifest copy written: {out} rows={len(rows)}")

    def write_run_contract(self, rows: list[dict[str, Any]]) -> None:
        from i2t_workflow_grpo.prompt_contract import (
            PROMPT_VERSION,
        )

        payload = {
            "repo": str(REPO),
            "repo_version": "0.3.0",
            "python_env": PYTHON,
            "backend": "transformers",
            "base_model": self.base_model,
            "policy": self.policy,
            "adapter": self.adapter,
            "test_manifest": str(self.test_manifest),
            "n_samples": len(rows),
            "gpus": self.gpus,
            "replicas_per_gpu": self.replicas_per_gpu,
            "logical_workers": len(self.gpus) * self.replicas_per_gpu,
            "prompt_version": PROMPT_VERSION,
            "routing": {
                "stage0_render_success": "visual_revision",
                "stage0_render_failure": "repair_or_complete",
            },
            "generation": GENERATION,
            "renderer": RENDERER,
            "reward": RSIM_CONFIG,
            "gt_feature_cache": str(self.gt_feats),
            "evaluation": {
                "per_sample_generations": 1,
                "same_policy_stage0_stage1": True,
                "worker_owns_full_sample": True,
                "dynamic_claim": True,
                "incremental_results": True,
                "resume": True,
            },
        }
        atomic_json(self.contract_dir / f"run_contract.{self.policy}.json", payload)
        self.log(f"run_contract written: {self.contract_dir / f'run_contract.{self.policy}.json'}")

    def scorer_cmd(self, gpu: int) -> list[str]:
        return [
            PYTHON,
            "-u",
            str(REPO / "scripts" / "evaluation" / "wf_eval_rsim_scorer.py"),
            "--queue-dir",
            str(self.controller_dir / "rsim_queue" / f"gpu{gpu}"),
            "--model-path",
            self.rsim_model,
            "--feature-cache-dir",
            str(self.gt_feats),
            "--heartbeat-path",
            str(self.logs / f"rsim_scorer_gpu{gpu}.heartbeat.json"),
            "--ready-path",
            str(self.logs / f"rsim_scorer_gpu{gpu}.ready"),
            "--stop-file",
            str(self.scorer_stop),
            "--gpu",
            str(gpu),
            "--device",
            "cuda:0",
            "--cpu-threads",
            str(self.args.cpu_threads),
        ]

    def worker_cmd(self, worker_id: str, gpu: int, policy_root: Path) -> list[str]:
        return [
            PYTHON,
            "-u",
            # Orchestration-only indirection: a Stage1-only control can point the
            # controller at an injection wrapper that reuses the identical worker
            # (same routing / prompts / renderer / metrics), replacing only the
            # Stage0 generation call with an external Stage0 record.
            os.environ.get("WF_WORKER_SCRIPT")
            or str(REPO / "scripts" / "evaluation" / "wf_eval_worker.py"),
            "--state-db",
            str(self.state_db),
            "--policy",
            self.policy,
            "--worker-id",
            worker_id,
            "--gpu",
            str(gpu),
            "--base-model",
            self.base_model,
            "--adapter",
            self.adapter,
            "--results-root",
            str(policy_root / "results"),
            "--tex-root",
            str(policy_root / "tex"),
            "--render-root",
            str(policy_root / "render"),
            "--rsim-queue-dir",
            str(self.controller_dir / "rsim_queue" / f"gpu{gpu}"),
            "--render-cache-dir",
            str(self.render_cache),
            "--render-tmp-dir",
            str(self.render_tmp),
            "--seed",
            str(self.args.seed),
            "--lease-seconds",
            str(self.args.lease_seconds),
            "--max-attempts",
            str(self.args.max_attempts),
            "--rsim-timeout",
            str(self.args.rsim_timeout),
            "--cpu-threads",
            str(self.args.cpu_threads),
        ]

    def spawn_scorers(self) -> list[dict[str, Any]]:
        scorers = []
        for gpu in self.gpus:
            log = self.logs / f"rsim_scorer_gpu{gpu}.log"
            proc = self.spawn(self.scorer_cmd(gpu), log, self.env_for_gpu(gpu))
            pidfile = self.pids / f"rsim_scorer_gpu{gpu}.pid"
            pidfile.write_text(str(proc.pid) + "\n", encoding="utf-8")
            scorers.append({"gpu": gpu, "proc": proc, "restarts": 0, "log": log})
            self.log(f"started rsim scorer gpu={gpu} pid={proc.pid} log={log}")
        return scorers

    def wait_scorers_ready(self, scorers: list[dict[str, Any]], timeout_s: float = 900.0) -> None:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            ready = all(
                (self.logs / f"rsim_scorer_gpu{s['gpu']}.ready").exists()
                and s["proc"].poll() is None
                for s in scorers
            )
            if ready:
                self.log("all 4 resident RSim scorers ready (vision-only load once)")
                return
            time.sleep(5)
        raise RuntimeError("RSim scorers not ready within timeout")

    def spawn_workers(self, policy_root: Path) -> list[dict[str, Any]]:
        workers = []
        n = len(self.gpus) * self.replicas_per_gpu
        for w in range(n):
            gpu = self.gpus[w // self.replicas_per_gpu]
            wid = f"worker{w:02d}"
            log = self.logs / f"{wid}.log"
            proc = self.spawn(self.worker_cmd(wid, gpu, policy_root), log, self.env_for_gpu(gpu))
            pidfile = self.pids / f"{wid}.pid"
            pidfile.write_text(str(proc.pid) + "\n", encoding="utf-8")
            workers.append(
                {
                    "worker_id": wid,
                    "gpu": gpu,
                    "proc": proc,
                    "restarts": 0,
                    "log": log,
                }
            )
            self.log(f"started worker {wid} gpu={gpu} pid={proc.pid} log={log}")
        return workers

    # --------------------------------------------------------------- watchdog
    def watchdog(self, workers: list[dict[str, Any]], scorers: list[dict[str, Any]], state: EvalState) -> None:
        counts = state.counts(self.policy)
        work_remains = int(counts.get("pending", 0)) + int(counts.get("running", 0)) > 0
        for w in workers:
            proc = w["proc"]
            if proc.poll() is None:
                continue
            if not work_remains:
                continue
            w["restarts"] += 1
            if w["restarts"] > self.args.max_worker_restarts:
                self.log_err(f"worker {w['worker_id']} exceeded restart cap ({w['restarts']}); leaving stopped")
                continue
            log = self.logs / f"{w['worker_id']}.log"
            new_proc = self.spawn(
                self.worker_cmd(w["worker_id"], w["gpu"], self.policy_root),
                log,
                self.env_for_gpu(w["gpu"]),
            )
            w["proc"] = new_proc
            pidfile = self.pids / f"{w['worker_id']}.pid"
            pidfile.write_text(str(new_proc.pid) + "\n", encoding="utf-8")
            self.log(f"restarted worker {w['worker_id']} gpu={w['gpu']} pid={new_proc.pid} restarts={w['restarts']}")
        for s in scorers:
            proc = s["proc"]
            if proc.poll() is None:
                continue
            s["restarts"] += 1
            if s["restarts"] > 10:
                self.log_err(f"rsim scorer gpu={s['gpu']} exceeded restart cap")
                continue
            log = self.logs / f"rsim_scorer_gpu{s['gpu']}.log"
            new_proc = self.spawn(self.scorer_cmd(s["gpu"]), log, self.env_for_gpu(s["gpu"]))
            s["proc"] = new_proc
            pidfile = self.pids / f"rsim_scorer_gpu{s['gpu']}.pid"
            pidfile.write_text(str(new_proc.pid) + "\n", encoding="utf-8")
            self.log(f"restarted rsim scorer gpu={s['gpu']} pid={new_proc.pid} restarts={s['restarts']}")

    def print_progress(self, workers: list[dict[str, Any]], scorers: list[dict[str, Any]], state: EvalState) -> None:
        counts = state.counts(self.policy)
        # Denominator is the number of tasks actually registered for this policy
        # (equal to len(manifest rows), or to the smoke prefix when one is used).
        total = sum(int(counts.get(key, 0)) for key in ("pending", "running", "done", "failed"))
        terminal = int(counts.get("done", 0)) + int(counts.get("failed", 0))
        print("[state]", flush=True)
        print(f"  {self.policy}: total={total} pending={counts.get('pending',0)} "
              f"running={counts.get('running',0)} done={counts.get('done',0)} "
              f"failed={counts.get('failed',0)}", flush=True)
        print("[inference slots]", flush=True)
        for gpu in self.gpus:
            slots = [w for w in workers if w["gpu"] == gpu]
            print(
                f"  GPU{gpu}: " + " ".join(
                    f"{s['worker_id']}={'running' if s['proc'].poll() is None else 'DEAD'}"
                    for s in slots
                ),
                flush=True,
            )
        for gpu in self.gpus:
            used, total_gib = gpu_memory_gib(gpu)
            print(f"  GPU{gpu} vram_used={used:.1f}/{total_gib:.0f} GiB", flush=True)
        alive = sum(1 for s in scorers if s["proc"].poll() is None)
        print(f"[rsim scorers] alive={alive}/{len(scorers)}", flush=True)
        wall = time.time() - self.start
        print(
            f"[global] terminal={terminal}/{total} elapsed={wall:.0f}s "
            f"throughput={terminal/max(wall,1e-6):.3f}/s",
            flush=True,
        )

    def stop_scorers(self, scorers: list[dict[str, Any]]) -> None:
        self.scorer_stop.write_text(iso_now() + "\n", encoding="utf-8")
        deadline = time.time() + 60
        while time.time() < deadline:
            if all(s["proc"].poll() is not None for s in scorers):
                break
            time.sleep(2)
        for s in scorers:
            if s["proc"].poll() is None:
                try:
                    os.killpg(os.getpgid(s["proc"].pid), signal.SIGTERM)
                except Exception:
                    try:
                        s["proc"].terminate()
                    except Exception:
                        pass
        time.sleep(5)
        for s in scorers:
            if s["proc"].poll() is None:
                try:
                    s["proc"].kill()
                except Exception:
                    pass
        self.log("rsim scorers stopped")

    # -------------------------------------------------------------------- run
    def run(self) -> int:
        rows = load_manifest(self.test_manifest)
        if self.smoke_limit:
            rows = rows[: self.smoke_limit]
        n_samples = len(rows)
        self.write_manifest_copy(rows)
        self.write_run_contract(rows)
        self.policy_root = self.eval_root / self.policy
        (self.policy_root / "results").mkdir(parents=True, exist_ok=True)
        (self.policy_root / "tex" / "stage0").mkdir(parents=True, exist_ok=True)
        (self.policy_root / "tex" / "stage1").mkdir(parents=True, exist_ok=True)
        (self.policy_root / "render" / "stage0").mkdir(parents=True, exist_ok=True)
        (self.policy_root / "render" / "stage1").mkdir(parents=True, exist_ok=True)
        self.log(
            f"controller start policy={self.policy} manifest={self.test_manifest} "
            f"n_samples={n_samples} "
            f"gpus={self.gpus} replicas_per_gpu={self.replicas_per_gpu} "
            f"logical_workers={len(self.gpus)*self.replicas_per_gpu} adapter={self.adapter}"
        )
        state = EvalState(self.state_db)
        inserted = state.init_tasks(self.policy, rows)
        self.log(f"tasks initialized policy={self.policy} inserted={inserted} expected={n_samples}")

        scorers = self.spawn_scorers()
        self.wait_scorers_ready(scorers)
        workers = self.spawn_workers(self.policy_root)

        last_log = 0.0
        while True:
            time.sleep(5)
            self.watchdog(workers, scorers, state)
            now = time.time()
            if now - last_log >= 30:
                last_log = now
                self.print_progress(workers, scorers, state)
            counts = state.counts(self.policy)
            if int(counts.get("pending", 0)) == 0 and int(counts.get("running", 0)) == 0:
                self.log(
                    f"policy {self.policy} terminal: done={counts.get('done',0)} "
                    f"failed={counts.get('failed',0)}"
                )
                break
        time.sleep(10)
        self.stop_scorers(scorers)
        done_note = {
            "timestamp": iso_now(),
            "policy": self.policy,
            "test_manifest": str(self.test_manifest),
            "n_samples": n_samples,
            "smoke_limit": self.smoke_limit,
            "gpus": self.gpus,
            "replicas_per_gpu": self.replicas_per_gpu,
            "logical_workers": len(self.gpus) * self.replicas_per_gpu,
            "done": int(counts.get("done", 0)),
            "failed": int(counts.get("failed", 0)),
        }
        atomic_json(self.policy_root / "DONE.controller", done_note)
        self.log("controller finished; DONE.controller written")
        return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--eval-root", default=str(EVAL_ROOT))
    p.add_argument("--policy", required=True)
    p.add_argument("--test-manifest", default=str(TEST_MANIFEST))
    p.add_argument("--base-model", default=BASE_MODEL)
    p.add_argument("--adapter", default=None)
    p.add_argument("--rsim-model", default=RSIM_MODEL)
    p.add_argument("--gpus", default="0,1,2,3")
    p.add_argument("--replicas-per-gpu", type=int, default=3)
    p.add_argument("--smoke-limit", type=int, default=0)
    p.add_argument("--seed", type=int, default=20260825)
    p.add_argument("--lease-seconds", type=float, default=1800.0)
    p.add_argument("--max-attempts", type=int, default=3)
    p.add_argument("--max-worker-restarts", type=int, default=10)
    p.add_argument("--rsim-timeout", type=float, default=900.0)
    p.add_argument("--cpu-threads", type=int, default=8)
    p.add_argument("--render-cache-dir", default=None)
    p.add_argument("--render-tmp-dir", default=None)
    p.add_argument("--feature-cache-dir", default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return EvalController(args).run()
    except Exception as exc:
        print(f"[controller][ERROR] {iso_now()} {exc!r}", flush=True)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
