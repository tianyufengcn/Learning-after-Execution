from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .environment import EnvironmentResult
from .generation import GeneratedAction, PromptSpec, QwenWorkflowGenerator
from .renderer import TikZRenderer
from .workflow import CreditConfig, failure_status, grouped_stage_advantages


@dataclass
class WorkflowTrajectory:
    sample_id: str
    rollout_index: int
    gt_image_path: str
    stage0: GeneratedAction
    stage0_env: EnvironmentResult
    stage1: GeneratedAction
    stage1_env: EnvironmentResult
    stage0_score: float
    stage1_score: float
    delta_score: float
    stage0_return: float = 0.0
    stage1_return: float = 0.0
    stage0_advantage: float = 0.0
    stage1_advantage: float = 0.0

    @property
    def stage0_ok(self) -> bool:
        return bool(self.stage0_env.render.ok)

    @property
    def stage1_ok(self) -> bool:
        return bool(self.stage1_env.render.ok)

    @property
    def route(self) -> str:
        return "visual_revision" if self.stage0_ok else "repair_or_complete"


@dataclass
class _TrajectoryState:
    sample_id: str
    rollout_index: int
    gt_image_path: str
    stage0: GeneratedAction | None = None
    stage0_env: EnvironmentResult | None = None
    stage1: GeneratedAction | None = None
    stage1_env: EnvironmentResult | None = None


class WorkflowEnvironment:
    def __init__(self, cfg: dict[str, Any], event_callback=None) -> None:
        rcfg = cfg["renderer"]
        renderer = TikZRenderer(
            cache_dir=rcfg["cache_dir"],
            tmp_dir=rcfg["tmp_dir"],
            texlive_bin=rcfg.get("texlive_bin"),
            pdftoppm_bin=rcfg.get("pdftoppm_bin"),
            compilers=rcfg.get("compilers"),
            compile_timeout_s=rcfg.get("compile_timeout_s", 30),
            image_size=rcfg.get("image_size", 448),
            dpi=rcfg.get("dpi", 150),
            keep_failed_logs=rcfg.get("keep_failed_logs", True),
            cache_failures=rcfg.get("cache_failures", True),
            cache_transient_failures=rcfg.get("cache_transient_failures", False),
        )
        from .rewards.rsim import RSimV2Scorer

        scorer = RSimV2Scorer(
            model_path=cfg["model_path"],
            cache_dir=cfg["feature_cache_dir"],
            detikzify_repo=cfg.get("detikzify_repo"),
            batch_size=cfg.get("score_batch_size", 16),
            mem_cache_size=cfg.get("score_mem_cache_size", 4096),
            emd_workers=cfg.get("emd_workers", 2),
            device=cfg.get("device"),
            require_vision_only=cfg.get("require_vision_only", True),
        )
        ecfg = cfg.get("environment", {})
        render_workers = ecfg.get("render_workers", "auto")
        if render_workers in (None, "auto"):
            world = max(1, int(os.environ.get("WORLD_SIZE", "1")))
            cpu = max(1, os.cpu_count() or 1)
            render_workers = max(2, min(8, max(1, cpu // world - 1)))
        from .environment import AsyncRenderScorePipeline

        self.pipeline = AsyncRenderScorePipeline(
            renderer,
            scorer,
            compile_failure_reward=0.0,
            render_workers=int(render_workers),
            max_pending=ecfg.get("max_pending", 96),
            score_batch_size=ecfg.get("score_batch_size", cfg.get("score_batch_size", 16)),
            score_max_wait_ms=ecfg.get("score_max_wait_ms", 120),
            score_queue_size=ecfg.get("score_queue_size", 128),
            event_callback=event_callback,
        )
        self.scorer = scorer

    def submit(self, completion: str, gt_path: str, metadata: dict[str, Any]) -> Future[EnvironmentResult]:
        return self.pipeline.submit(completion, gt_path, metadata)

    def telemetry(self) -> dict[str, Any]:
        return self.pipeline.telemetry()

    def close(self) -> None:
        self.pipeline.close()


class OnPolicyWorkflowRollout:
    """Two-stage on-policy rollout executor.

    v0.3.0 can execute arbitrary explicit trajectory tasks, which lets the
    distributed coordinator globally work-steal trajectories across ranks. The
    environment remains asynchronous and resident inside each rank. Group
    credit may be deferred until all distributed trajectory results are ready.
    """

    def __init__(
        self,
        generator: QwenWorkflowGenerator,
        reward_cfg: dict[str, Any],
        workflow_cfg: dict[str, Any],
        *,
        environment: WorkflowEnvironment | Any | None = None,
    ) -> None:
        self.generator = generator
        self.group_size = int(workflow_cfg.get("num_generations", 8))
        self.credit = CreditConfig(scale=float(workflow_cfg.get("reward_scale", 10.0)))
        log_dir = Path(workflow_cfg.get("log_dir", reward_cfg["renderer"]["cache_dir"]))
        log_dir.mkdir(parents=True, exist_ok=True)
        rank = os.environ.get("RANK", "0")
        self.log_dir = log_dir
        self.rank = str(rank)
        self.rollout_log = log_dir / f"workflow_rollouts_rank{rank}.jsonl"
        self.pipeline_log = log_dir / f"workflow_pipeline_rank{rank}.jsonl"
        self.status_path = log_dir / f"workflow_status_rank{rank}.json"
        self.artifact_root = log_dir / "artifacts"
        self.persist_artifacts = bool(workflow_cfg.get("persist_artifacts", True))
        self.status_interval_s = float(workflow_cfg.get("status_interval_s", 2.0))
        if self.persist_artifacts:
            self.artifact_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.environment = environment if environment is not None else WorkflowEnvironment(reward_cfg, self._pipeline_event)

    def _write(self, path: Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        with self._lock, open(path, "a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            handle.flush()

    def _pipeline_event(self, row: dict[str, Any]) -> None:
        self._write(self.pipeline_log, [row])

    def _status(self, global_step: int, phase: str, **extra: Any) -> None:
        payload = {
            "time": time.time(),
            "rank": self.rank,
            "global_step": int(global_step),
            "phase": phase,
            **extra,
        }
        tmp = self.status_path.with_suffix(f".tmp.{os.getpid()}.json")
        with self._lock:
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
            os.replace(tmp, self.status_path)

    @staticmethod
    def _safe_sample_id(value: str) -> str:
        safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(value))
        return safe[:120] or "sample"

    def _persist_trajectory_artifacts(self, trajectory: WorkflowTrajectory, global_step: int) -> tuple[str | None, str | None]:
        if not self.persist_artifacts:
            return None, None
        root = (
            self.artifact_root
            / f"step_{global_step:06d}"
            / self._safe_sample_id(trajectory.sample_id)
            / f"rollout_{trajectory.rollout_index:02d}"
        )
        root.mkdir(parents=True, exist_ok=True)
        stage0_path = root / "stage0.tex"
        stage1_path = root / "stage1.tex"
        stage0_path.write_text(trajectory.stage0.text, encoding="utf-8")
        stage1_path.write_text(trajectory.stage1.text, encoding="utf-8")
        return str(stage0_path), str(stage1_path)

    def _expanded_rows(self, rows: list[dict[str, Any]]) -> list[tuple[str, int, str]]:
        expanded: list[tuple[str, int, str]] = []
        for row in rows:
            sample_id = str(row["sample_id"])
            gt_path = str(row.get("gt_image_path") or row.get("image"))
            for rollout_index in range(self.group_size):
                expanded.append((sample_id, rollout_index, gt_path))
        return expanded

    @staticmethod
    def explicit_tasks_to_expanded(tasks: list[dict[str, Any]]) -> list[tuple[str, int, str]]:
        return [
            (str(task["sample_id"]), int(task["rollout_index"]), str(task["gt_image_path"]))
            for task in tasks
        ]

    def _route_stage1_spec(self, state: _TrajectoryState, global_step: int) -> tuple[PromptSpec, dict[str, Any]]:
        assert state.stage0 is not None and state.stage0_env is not None
        if state.stage0_env.render.ok and state.stage0_env.render.png_path:
            spec = PromptSpec(
                "revision",
                state.gt_image_path,
                current_image_path=state.stage0_env.render.png_path,
                current_tikz=state.stage0.text,
            )
            route = "visual_revision"
        else:
            spec = PromptSpec(
                "repair",
                state.gt_image_path,
                current_tikz=state.stage0.text,
                failure_status=failure_status(
                    state.stage0_env.render.error,
                    state.stage0.hit_max_tokens,
                    state.stage0.ends_with_document,
                ),
            )
            route = "repair_or_complete"
        return spec, {
            "global_step": global_step,
            "stage": 1,
            "sample_id": state.sample_id,
            "rollout_index": state.rollout_index,
            "route": route,
        }

    def _flush_stage1_batch(
        self,
        pending: list[tuple[int, PromptSpec, dict[str, Any]]],
        states: dict[int, _TrajectoryState],
        stage1_futures: dict[Future[EnvironmentResult], int],
        stage1_generation_s: list[float],
        global_step: int,
        completed_stage0: int,
        n_total: int,
    ) -> int:
        """Generate ready Stage-1 work, grouped by route/image cardinality."""
        if not pending:
            return 0
        groups: dict[str, list[tuple[int, PromptSpec, dict[str, Any]]]] = {}
        for item in pending:
            groups.setdefault(item[1].kind, []).append(item)
        submitted = 0
        stage1_micro = max(1, int(getattr(self.generator, "stage1_microbatch", 2)))
        for kind in sorted(groups):
            items = groups[kind]
            for start in range(0, len(items), stage1_micro):
                batch = items[start : start + stage1_micro]
                specs = [item[1] for item in batch]
                self._pipeline_event(
                    {
                        "event": "stage1_generation_start",
                        "time": time.time(),
                        "global_step": global_step,
                        "stage": 1,
                        "kind": kind,
                        "batch_size": len(batch),
                        "completed_stage0": int(completed_stage0),
                        "n_total": int(n_total),
                    }
                )
                started = time.monotonic()
                actions = self.generator.generate(specs, stage=1)
                elapsed = time.monotonic() - started
                stage1_generation_s[0] += elapsed
                self._pipeline_event(
                    {
                        "event": "stage1_generation_end",
                        "time": time.time(),
                        "global_step": global_step,
                        "stage": 1,
                        "kind": kind,
                        "batch_size": len(batch),
                        "elapsed_s": elapsed,
                    }
                )
                for (idx, _spec, meta), action in zip(batch, actions, strict=True):
                    states[idx].stage1 = action
                    stage1_futures[self.environment.submit(action.text, states[idx].gt_image_path, meta)] = idx
                    submitted += 1
        pending.clear()
        return submitted

    def _emit_progress(
        self,
        global_step: int,
        phase: str,
        *,
        n_total: int,
        stage0_actions: int,
        stage0_done: int,
        pending_stage1: int,
        stage1_submitted: int,
        stage1_done: int,
        routes: dict[str, int],
        last_status_t: list[float],
    ) -> None:
        now = time.monotonic()
        if phase != "rollout_done" and now - last_status_t[0] < self.status_interval_s:
            return
        last_status_t[0] = now
        telemetry = self.environment.telemetry() if hasattr(self.environment, "telemetry") else {}
        self._status(
            global_step,
            phase,
            trajectories=n_total,
            stage0_actions=stage0_actions,
            stage0_done=stage0_done,
            pending_stage1=pending_stage1,
            stage1_submitted=stage1_submitted,
            stage1_done=stage1_done,
            revision_routes=routes.get("visual_revision", 0),
            repair_routes=routes.get("repair_or_complete", 0),
            telemetry=telemetry,
        )

    def rollout_tasks(
        self,
        tasks: list[dict[str, Any]],
        global_step: int,
        *,
        assign_credit: bool = False,
        log_trajectories: bool = False,
    ) -> tuple[list[WorkflowTrajectory], dict[str, float]]:
        expanded = self.explicit_tasks_to_expanded(tasks)
        return self._rollout_expanded(
            expanded,
            global_step,
            assign_credit=assign_credit,
            log_trajectories=log_trajectories,
        )

    def rollout(self, rows: list[dict[str, Any]], global_step: int) -> tuple[list[WorkflowTrajectory], dict[str, float]]:
        return self._rollout_expanded(
            self._expanded_rows(rows),
            global_step,
            assign_credit=True,
            log_trajectories=True,
        )

    def _rollout_expanded(
        self,
        expanded: list[tuple[str, int, str]],
        global_step: int,
        *,
        assign_credit: bool,
        log_trajectories: bool,
    ) -> tuple[list[WorkflowTrajectory], dict[str, float]]:
        if not expanded:
            return [], {"rollout_wall_s": 0.0, "stage0_generation_s": 0.0, "stage1_generation_s": 0.0}
        if not hasattr(self.environment, "submit") or not hasattr(self.generator, "generate_iter"):
            raise RuntimeError("v0.3.0 distributed rollout requires async environment.submit and generator.generate_iter")

        started_all = time.monotonic()
        n_total = len(expanded)
        states = {i: _TrajectoryState(sample_id, rollout_index, gt_path) for i, (sample_id, rollout_index, gt_path) in enumerate(expanded)}
        stage0_specs = [PromptSpec("direct", gt_path) for _, _, gt_path in expanded]
        self._status(global_step, "stage0_generation", trajectories=n_total, sample_ids=[x[0] for x in expanded])

        stage0_futures: dict[Future[EnvironmentResult], int] = {}
        stage1_futures: dict[Future[EnvironmentResult], int] = {}
        pending_stage1: list[tuple[int, PromptSpec, dict[str, Any]]] = []
        stage0_generation_s = 0.0
        stage1_generation_s = [0.0]
        routes = {"visual_revision": 0, "repair_or_complete": 0}
        last_status_t = [0.0]
        submitted_stage0 = completed_stage0 = submitted_stage1 = completed_stage1 = 0

        def drain_ready(timeout: float) -> None:
            nonlocal completed_stage0, completed_stage1
            watched = set(stage0_futures) | set(stage1_futures)
            if not watched:
                return
            done, _ = wait(watched, timeout=timeout, return_when=FIRST_COMPLETED)
            for future in done:
                if future in stage0_futures:
                    idx = stage0_futures.pop(future)
                    env = future.result()
                    states[idx].stage0_env = env
                    spec, meta = self._route_stage1_spec(states[idx], global_step)
                    routes[meta["route"]] += 1
                    pending_stage1.append((idx, spec, meta))
                    completed_stage0 += 1
                elif future in stage1_futures:
                    idx = stage1_futures.pop(future)
                    states[idx].stage1_env = future.result()
                    completed_stage1 += 1

        cursor = 0
        for chunk in self.generator.generate_iter(stage0_specs, stage=0):
            chunk_start = cursor
            chunk_end = cursor + len(chunk)
            elapsed = max((a.generation_batch_elapsed_s for a in chunk), default=0.0)
            stage0_generation_s += elapsed
            self._pipeline_event(
                {
                    "event": "stage0_generation_end",
                    "time": time.time(),
                    "global_step": global_step,
                    "stage": 0,
                    "batch_size": len(chunk),
                    "elapsed_s": elapsed,
                }
            )
            for idx, action in zip(range(chunk_start, chunk_end), chunk, strict=True):
                state = states[idx]
                state.stage0 = action
                meta = {
                    "global_step": global_step,
                    "stage": 0,
                    "sample_id": state.sample_id,
                    "rollout_index": state.rollout_index,
                }
                stage0_futures[self.environment.submit(action.text, state.gt_image_path, meta)] = idx
                submitted_stage0 += 1
            cursor = chunk_end
            drain_ready(0.0)
            stage1_micro = max(1, int(getattr(self.generator, "stage1_microbatch", 2)))
            if len(pending_stage1) >= stage1_micro:
                submitted_stage1 += self._flush_stage1_batch(
                    pending_stage1,
                    states,
                    stage1_futures,
                    stage1_generation_s,
                    global_step,
                    completed_stage0,
                    n_total,
                )
            self._emit_progress(
                global_step,
                "workflow_running",
                n_total=n_total,
                stage0_actions=submitted_stage0,
                stage0_done=completed_stage0,
                pending_stage1=len(pending_stage1),
                stage1_submitted=submitted_stage1,
                stage1_done=completed_stage1,
                routes=routes,
                last_status_t=last_status_t,
            )

        while stage0_futures or pending_stage1 or stage1_futures:
            drain_ready(0.05)
            stage1_micro = max(1, int(getattr(self.generator, "stage1_microbatch", 2)))
            if pending_stage1 and (len(pending_stage1) >= stage1_micro or not stage0_futures):
                submitted_stage1 += self._flush_stage1_batch(
                    pending_stage1,
                    states,
                    stage1_futures,
                    stage1_generation_s,
                    global_step,
                    completed_stage0,
                    n_total,
                )
            self._emit_progress(
                global_step,
                "workflow_running",
                n_total=n_total,
                stage0_actions=submitted_stage0,
                stage0_done=completed_stage0,
                pending_stage1=len(pending_stage1),
                stage1_submitted=submitted_stage1,
                stage1_done=completed_stage1,
                routes=routes,
                last_status_t=last_status_t,
            )

        trajectories: list[WorkflowTrajectory] = []
        for idx in range(n_total):
            state = states[idx]
            assert state.stage0 is not None and state.stage0_env is not None
            assert state.stage1 is not None and state.stage1_env is not None
            stage0_score = float(state.stage0_env.reward) if state.stage0_env.render.ok else 0.0
            stage1_score = float(state.stage1_env.reward) if state.stage1_env.render.ok else 0.0
            trajectories.append(
                WorkflowTrajectory(
                    sample_id=state.sample_id,
                    rollout_index=state.rollout_index,
                    gt_image_path=state.gt_image_path,
                    stage0=state.stage0,
                    stage0_env=state.stage0_env,
                    stage1=state.stage1,
                    stage1_env=state.stage1_env,
                    stage0_score=stage0_score,
                    stage1_score=stage1_score,
                    delta_score=stage1_score - stage0_score,
                )
            )

        if assign_credit:
            if len(trajectories) % self.group_size != 0:
                raise ValueError("assign_credit=True requires complete G-sized root groups")
            s0 = [trajectory.stage0_score for trajectory in trajectories]
            s1 = [trajectory.stage1_score for trajectory in trajectories]
            r0, r1, a0, a1 = grouped_stage_advantages(s0, s1, self.group_size, self.credit)
            for i, trajectory in enumerate(trajectories):
                trajectory.stage0_return = r0[i]
                trajectory.stage1_return = r1[i]
                trajectory.stage0_advantage = a0[i]
                trajectory.stage1_advantage = a1[i]

        if log_trajectories:
            self._log_trajectories(trajectories, global_step)
        self._emit_progress(
            global_step,
            "rollout_done",
            n_total=n_total,
            stage0_actions=submitted_stage0,
            stage0_done=completed_stage0,
            pending_stage1=0,
            stage1_submitted=submitted_stage1,
            stage1_done=completed_stage1,
            routes=routes,
            last_status_t=[0.0],
        )
        return trajectories, {
            "rollout_wall_s": time.monotonic() - started_all,
            "stage0_generation_s": stage0_generation_s,
            "stage1_generation_s": stage1_generation_s[0],
        }

    def _log_trajectories(self, trajectories: list[WorkflowTrajectory], global_step: int) -> None:
        rows: list[dict[str, Any]] = []
        groups: dict[str, list[WorkflowTrajectory]] = {}
        for trajectory in trajectories:
            groups.setdefault(trajectory.sample_id, []).append(trajectory)
            stage0_tex_path, stage1_tex_path = self._persist_trajectory_artifacts(trajectory, global_step)
            rows.append(
                {
                    "event": "trajectory",
                    "time": time.time(),
                    "global_step": global_step,
                    "sample_id": trajectory.sample_id,
                    "rollout_index": trajectory.rollout_index,
                    "route": trajectory.route,
                    "stage0_compile_ok": trajectory.stage0_ok,
                    "stage1_compile_ok": trajectory.stage1_ok,
                    "stage0_rsim": trajectory.stage0_score,
                    "stage1_rsim": trajectory.stage1_score,
                    "delta_rsim": trajectory.delta_score,
                    "stage0_return": trajectory.stage0_return,
                    "stage1_return": trajectory.stage1_return,
                    "stage0_advantage": trajectory.stage0_advantage,
                    "stage1_advantage": trajectory.stage1_advantage,
                    "stage0_tokens": trajectory.stage0.completion_tokens,
                    "stage1_tokens": trajectory.stage1.completion_tokens,
                    "stage0_prompt_tokens": trajectory.stage0.prompt.prompt_len,
                    "stage1_prompt_tokens": trajectory.stage1.prompt.prompt_len,
                    "stage0_full_sequence_tokens": trajectory.stage0.full_sequence_tokens,
                    "stage1_full_sequence_tokens": trajectory.stage1.full_sequence_tokens,
                    "stage0_hit_max": trajectory.stage0.hit_max_tokens,
                    "stage1_hit_max": trajectory.stage1.hit_max_tokens,
                    "stage0_end_document": trajectory.stage0.ends_with_document,
                    "stage1_end_document": trajectory.stage1.ends_with_document,
                    "stage0_stopped_on_end_document": trajectory.stage0.stopped_on_end_document,
                    "stage1_stopped_on_end_document": trajectory.stage1.stopped_on_end_document,
                    "stage0_generation_batch_elapsed_s": trajectory.stage0.generation_batch_elapsed_s,
                    "stage1_generation_batch_elapsed_s": trajectory.stage1.generation_batch_elapsed_s,
                    "stage0_error": trajectory.stage0_env.render.error,
                    "stage1_error": trajectory.stage1_env.render.error,
                    "stage0_tikz_hash": trajectory.stage0_env.render.code_hash,
                    "stage1_tikz_hash": trajectory.stage1_env.render.code_hash,
                    "stage0_tex_path": stage0_tex_path,
                    "stage1_tex_path": stage1_tex_path,
                    "stage0_png_path": trajectory.stage0_env.render.png_path,
                    "stage1_png_path": trajectory.stage1_env.render.png_path,
                    "gt_image_path": trajectory.gt_image_path,
                    "stage0_preview": trajectory.stage0.text[:200].replace("\n", "\\n"),
                    "stage1_preview": trajectory.stage1.text[:200].replace("\n", "\\n"),
                }
            )
        for sample_id, group in groups.items():
            rows.append(
                {
                    "event": "group_summary",
                    "time": time.time(),
                    "global_step": global_step,
                    "sample_id": sample_id,
                    "group_size": len(group),
                    "stage0_rsim": [x.stage0_score for x in group],
                    "stage1_rsim": [x.stage1_score for x in group],
                    "delta_rsim": [x.delta_score for x in group],
                    "stage0_compile": [x.stage0_ok for x in group],
                    "stage1_compile": [x.stage1_ok for x in group],
                    "routes": [x.route for x in group],
                }
            )
        self._write(self.rollout_log, rows)

    def close(self) -> None:
        self.environment.close()
