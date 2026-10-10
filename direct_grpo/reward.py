from __future__ import annotations

import json
import os
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from .environment import AsyncRenderScorePipeline
from .renderer import TikZRenderer
from .rewards.rsim import RSimV2Scorer
from .tikz import completion_to_text


class RenderAwareReward:
    """TRL-compatible terminal reward with a pipelined TikZ environment.

    The callable still returns one scalar per completed rollout, as required by
    GRPOTrainer. Internally there is no render->RSim stage barrier: successful
    PNGs are sent to the dynamic RSim micro-batcher as soon as individual TeX
    workers finish.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        renderer_cfg = cfg["renderer"]
        self.renderer = TikZRenderer(
            cache_dir=renderer_cfg["cache_dir"],
            tmp_dir=renderer_cfg["tmp_dir"],
            texlive_bin=renderer_cfg.get("texlive_bin"),
            pdftoppm_bin=renderer_cfg.get("pdftoppm_bin"),
            compilers=renderer_cfg.get("compilers"),
            compile_timeout_s=renderer_cfg.get("compile_timeout_s", 30),
            image_size=renderer_cfg.get("image_size", 448),
            dpi=renderer_cfg.get("dpi", 150),
            keep_failed_logs=renderer_cfg.get("keep_failed_logs", True),
            cache_failures=renderer_cfg.get("cache_failures", True),
            cache_transient_failures=renderer_cfg.get("cache_transient_failures", False),
        )
        backend = cfg.get("backend", "rsim_v2")
        if backend != "rsim_v2":
            raise ValueError(f"Initial implementation supports backend='rsim_v2', got {backend!r}")
        self.scorer = RSimV2Scorer(
            model_path=cfg["model_path"],
            cache_dir=cfg["feature_cache_dir"],
            detikzify_repo=cfg.get("detikzify_repo"),
            batch_size=cfg.get("score_batch_size", 16),
            mem_cache_size=cfg.get("score_mem_cache_size", 4096),
            emd_workers=cfg.get("emd_workers", 4),
            device=cfg.get("device"),
        )
        self.compile_failure_reward = float(cfg.get("compile_failure_reward", 0.0))
        self.max_completion_length = int(cfg.get("max_completion_length", 8192))
        self.prompt_version = str(cfg.get("prompt_version", "unknown"))
        self.eos_token_id = int(cfg.get("eos_token_id") or -1)
        self.rank = str(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))

        log_dir = Path(cfg.get("log_dir", renderer_cfg["cache_dir"]))
        log_dir.mkdir(parents=True, exist_ok=True)
        rank = os.environ.get("RANK", "0")
        self.rollout_log_path = log_dir / f"rollouts_rank{rank}.jsonl"
        self.pipeline_log_path = log_dir / f"pipeline_rank{rank}.jsonl"
        self.log_pipeline_events = bool(cfg.get("log_pipeline_events", True))
        self._lock = threading.Lock()
        self._call_counter = 0

        env_cfg = cfg.get("environment", {})
        render_workers = env_cfg.get("render_workers", "auto")
        if render_workers in (None, "auto"):
            world = max(1, int(os.environ.get("WORLD_SIZE", "1")))
            cpu = max(1, os.cpu_count() or 1)
            # Leave roughly one CPU core per rank for Python/trainer/runtime.
            render_workers = max(2, min(8, max(1, cpu // world - 1)))
        render_workers = int(render_workers)
        print(
            f"[environment] rank={os.environ.get('RANK','0')} render_workers={render_workers} "
            f"world_size={os.environ.get('WORLD_SIZE','1')} cpu_count={os.cpu_count()}"
        )
        self.pipeline = AsyncRenderScorePipeline(
            self.renderer,
            self.scorer,
            compile_failure_reward=self.compile_failure_reward,
            render_workers=render_workers,
            max_pending=env_cfg.get("max_pending", 96),
            score_batch_size=env_cfg.get("score_batch_size", cfg.get("score_batch_size", 16)),
            score_max_wait_ms=env_cfg.get("score_max_wait_ms", 120),
            score_queue_size=env_cfg.get("score_queue_size", 128),
            event_callback=self._pipeline_event if self.log_pipeline_events else None,
        )

    @staticmethod
    def _as_list(value: Any, n: int, default: Any = None) -> list[Any]:
        if value is None:
            return [default] * n
        if isinstance(value, (list, tuple)):
            if len(value) == n:
                return list(value)
            if len(value) == 1:
                return list(value) * n
        return [value] * n

    def _write_jsonl(self, path: Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        with self._lock, open(path, "a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            f.flush()

    def _pipeline_event(self, row: dict[str, Any]) -> None:
        self._write_jsonl(self.pipeline_log_path, [row])

    @staticmethod
    def _completion_lengths(completion_ids: Any, n: int) -> list[int | None]:
        if completion_ids is None:
            return [None] * n
        try:
            if len(completion_ids) != n:
                return [None] * n
            out: list[int | None] = []
            for ids in completion_ids:
                try:
                    out.append(len(ids))
                except Exception:
                    out.append(None)
            return out
        except Exception:
            return [None] * n

    def _finish_reasons(self, completion_ids: Any, n: int) -> list[str]:
        if completion_ids is None:
            return ["unknown"] * n
        out: list[str] = []
        try:
            if len(completion_ids) != n:
                return ["unknown"] * n
            for ids in completion_ids:
                try:
                    length = len(ids)
                except Exception:
                    out.append("unknown")
                    continue
                if length == 0:
                    out.append("empty")
                elif self.eos_token_id >= 0 and int(ids[-1]) == self.eos_token_id:
                    out.append("eos")
                elif length >= self.max_completion_length:
                    out.append("max_tokens")
                else:
                    out.append("length")
            return out
        except Exception:
            return ["unknown"] * n

    def __call__(
        self,
        prompts: list[Any] | None = None,
        completions: list[Any] | None = None,
        completion_ids: Any = None,
        gt_image_path: list[str] | str | None = None,
        sample_id: list[str] | str | None = None,
        trainer_state: Any = None,
        log_extra=None,
        log_metric=None,
        **kwargs: Any,
    ) -> list[float]:
        completions = completions or []
        n = len(completions)
        if n == 0:
            return []

        gt_paths = self._as_list(gt_image_path, n)
        sample_ids = [str(x) for x in self._as_list(sample_id, n, "unknown")]
        texts = [completion_to_text(c) for c in completions]
        token_lengths = self._completion_lengths(completion_ids, n)
        finish_reasons = self._finish_reasons(completion_ids, n)
        gt_cache_hits = self.scorer.gt_cache_hits(gt_paths)
        global_step = int(getattr(trainer_state, "global_step", -1)) if trainer_state is not None else -1

        with self._lock:
            self._call_counter += 1
            call_id = self._call_counter

        # GRPO repeats the same prompt G times. Record a stable rollout index within
        # each sample group for sample-level debugging.
        occurrence: Counter[str] = Counter()
        metadata: list[dict[str, Any]] = []
        for i, sid in enumerate(sample_ids):
            rollout_idx = occurrence[sid]
            occurrence[sid] += 1
            metadata.append(
                {
                    "global_step": global_step,
                    "reward_call": call_id,
                    "sample_id": sid,
                    "group_id": sid,
                    "rollout_id": f"{sid}:{rollout_idx}",
                    "rank": self.rank,
                    "rollout_index": rollout_idx,
                    "completion_tokens": token_lengths[i],
                    "hit_max_tokens": bool(token_lengths[i] is not None and token_lengths[i] >= self.max_completion_length),
                    "prompt_version": self.prompt_version,
                    "finish_reason": finish_reasons[i],
                    "gt_cache_hit": gt_cache_hits[i],
                }
            )

        start = time.monotonic()
        env_results = self.pipeline.evaluate_many(texts, gt_paths, metadata)
        elapsed = time.monotonic() - start
        rewards = [float(x.reward) for x in env_results]
        compile_rate = sum(x.render.ok for x in env_results) / max(1, n)
        valid_scores = [x.reward for x in env_results if x.render.ok]
        batch_reward_mean = mean(valid_scores) if valid_scores else self.compile_failure_reward
        cache_hit_rate = sum(x.render.cache_hit for x in env_results) / max(1, n)

        if log_metric is not None:
            try:
                log_metric("reward/compile_rate", compile_rate)
                log_metric("reward/visual_mean_valid", batch_reward_mean)
                log_metric("reward/environment_wall_s", elapsed)
                log_metric("reward/render_cache_hit_rate", cache_hit_rate)
            except Exception:
                pass
        if log_extra is not None:
            try:
                log_extra("compile_ok", [x.render.ok for x in env_results])
                log_extra("tikz_hash", [x.render.code_hash for x in env_results])
                log_extra("completion_tokens", token_lengths)
                log_extra("hit_max_tokens", [m["hit_max_tokens"] for m in metadata])
            except Exception:
                pass

        rows: list[dict[str, Any]] = []
        grouped: dict[str, list[int]] = defaultdict(list)
        for i, (result, reward) in enumerate(zip(env_results, rewards, strict=True)):
            grouped[sample_ids[i]].append(i)
            rows.append(
                {
                    "event": "rollout_result",
                    "time": time.time(),
                    **metadata[i],
                    "tikz_hash": result.render.code_hash,
                    "cache_key": result.render.cache_key,
                    "compile_ok": result.render.ok,
                    "compiler": result.render.compiler,
                    "reward": float(reward),
                    "render_s": result.render.elapsed_s,
                    "compile_time": result.render.compile_s,
                    "render_time": max(0.0, result.render.elapsed_s - result.render.compile_s),
                    "score_s": result.score_elapsed_s,
                    "environment_total_s": result.total_elapsed_s,
                    "cache_hit": result.render.cache_hit,
                    "error": result.render.error,
                    "completion_chars": len(texts[i]),
                    "completion_preview": texts[i][:240].replace("\n", "\\n"),
                }
            )

        for sid, idxs in grouped.items():
            group_rewards = [rewards[i] for i in idxs]
            group_tokens = [token_lengths[i] for i in idxs if token_lengths[i] is not None]
            token_stats = {
                "token_min": min(group_tokens) if group_tokens else None,
                "token_mean": float(mean(group_tokens)) if group_tokens else None,
                "token_max": max(group_tokens) if group_tokens else None,
            }
            rows.append(
                {
                    "event": "group_summary",
                    "time": time.time(),
                    "global_step": global_step,
                    "reward_call": call_id,
                    "sample_id": sid,
                    "group_id": sid,
                    "G": len(idxs),
                    "group_size_seen": len(idxs),
                    "rewards": group_rewards,
                    "reward_mean": float(mean(group_rewards)),
                    "reward_min": float(min(group_rewards)),
                    "reward_max": float(max(group_rewards)),
                    "reward_range": float(max(group_rewards) - min(group_rewards)),
                    "zero_range": bool(max(group_rewards) == min(group_rewards)),
                    "compile_ok": [env_results[i].render.ok for i in idxs],
                    "compile_count": sum(1 for i in idxs if env_results[i].render.ok),
                    "completion_tokens": [token_lengths[i] for i in idxs],
                    "hit_max_tokens": [metadata[i]["hit_max_tokens"] for i in idxs],
                    **token_stats,
                }
            )

        self._write_jsonl(self.rollout_log_path, rows)
        return rewards
