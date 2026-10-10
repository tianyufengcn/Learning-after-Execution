from __future__ import annotations

import atexit
import queue
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .renderer import RenderResult, TikZRenderer


EventCallback = Callable[[dict[str, Any]], None]


@dataclass
class EnvironmentResult:
    render: RenderResult
    reward: float
    score_elapsed_s: float = 0.0
    total_elapsed_s: float = 0.0


@dataclass
class _ScoreRequest:
    generated_path: str
    gt_path: str
    future: Future[tuple[float, float]]
    metadata: dict[str, Any]
    enqueued_at: float


class DynamicScoreBatcher:
    """Single-GPU dynamic micro-batcher for RSim-like scorers.

    Render workers enqueue PNGs as soon as they are ready. The scorer thread
    flushes when either ``batch_size`` items are available or ``max_wait_ms`` has
    elapsed since the oldest item arrived. This removes the stage-level barrier
    "render everything, then score everything" while retaining efficient batched
    GPU image encoding.
    """

    def __init__(
        self,
        scorer: Any,
        *,
        batch_size: int = 16,
        max_wait_ms: int = 120,
        max_queue: int = 128,
        event_callback: EventCallback | None = None,
    ) -> None:
        self.scorer = scorer
        self.batch_size = max(1, int(batch_size))
        self.max_wait_s = max(0.0, float(max_wait_ms) / 1000.0)
        self.queue: queue.Queue[_ScoreRequest | None] = queue.Queue(maxsize=max(1, int(max_queue)))
        self.event_callback = event_callback
        self._closed = threading.Event()
        self._thread = threading.Thread(target=self._run, name="i2t-rsim-batcher", daemon=True)
        self._thread.start()

    def _emit(self, event: str, metadata: dict[str, Any], **extra: Any) -> None:
        if self.event_callback is None:
            return
        try:
            self.event_callback({"event": event, "time": time.time(), **metadata, **extra})
        except Exception:
            pass

    def submit(self, generated_path: str, gt_path: str, metadata: dict[str, Any] | None = None) -> Future[tuple[float, float]]:
        if self._closed.is_set():
            raise RuntimeError("DynamicScoreBatcher is closed")
        fut: Future[tuple[float, float]] = Future()
        meta = dict(metadata or {})
        req = _ScoreRequest(generated_path, gt_path, fut, meta, time.monotonic())
        self.queue.put(req)  # bounded queue => natural backpressure
        self._emit("score_queued", meta, queue_size=self.queue.qsize())
        return fut

    def _run(self) -> None:
        while not self._closed.is_set():
            try:
                first = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if first is None:
                self.queue.task_done()
                break

            batch = [first]
            deadline = first.enqueued_at + self.max_wait_s
            while len(batch) < self.batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    item = self.queue.get(timeout=remaining)
                except queue.Empty:
                    break
                if item is None:
                    self.queue.task_done()
                    self._closed.set()
                    break
                batch.append(item)

            start = time.monotonic()
            for req in batch:
                self._emit("score_batch_start", req.metadata, batch_size=len(batch))
            try:
                scores = self.scorer.score_paths(
                    [r.generated_path for r in batch],
                    [r.gt_path for r in batch],
                )
                if len(scores) != len(batch):
                    raise RuntimeError(f"scorer returned {len(scores)} scores for batch of {len(batch)}")
                elapsed = time.monotonic() - start
                for req, score in zip(batch, scores, strict=True):
                    if not req.future.done():
                        req.future.set_result((float(score), elapsed))
                    self._emit("score_done", req.metadata, reward=float(score), score_batch_s=elapsed, batch_size=len(batch))
            except Exception as exc:
                elapsed = time.monotonic() - start
                for req in batch:
                    if not req.future.done():
                        req.future.set_exception(exc)
                    self._emit("score_error", req.metadata, error=repr(exc), score_batch_s=elapsed, batch_size=len(batch))
            finally:
                for _ in batch:
                    self.queue.task_done()

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        try:
            self.queue.put_nowait(None)
        except queue.Full:
            # Worker will exit after draining; don't block process shutdown.
            pass
        self._thread.join(timeout=5.0)


class AsyncRenderScorePipeline:
    """Bounded render -> dynamic-score pipeline.

    The policy/trainer remains synchronous at the GRPO update boundary, but the
    expensive environment is pipelined:

      completion -> CPU TeX worker -> PNG-ready event -> dynamic RSim batcher

    Each rollout is independent. A bad/slow TeX program cannot block other
    render workers, and RSim begins as soon as successful PNGs appear instead of
    waiting for the full render batch.
    """

    def __init__(
        self,
        renderer: TikZRenderer,
        scorer: Any,
        *,
        compile_failure_reward: float = 0.0,
        render_workers: int = 12,
        max_pending: int = 96,
        score_batch_size: int = 16,
        score_max_wait_ms: int = 120,
        score_queue_size: int = 128,
        event_callback: EventCallback | None = None,
    ) -> None:
        self.renderer = renderer
        self.compile_failure_reward = float(compile_failure_reward)
        self.event_callback = event_callback
        self.render_pool = ThreadPoolExecutor(max_workers=max(1, int(render_workers)), thread_name_prefix="i2t-tex")
        self.pending = threading.BoundedSemaphore(max(1, int(max_pending)))
        self.scorer = DynamicScoreBatcher(
            scorer,
            batch_size=score_batch_size,
            max_wait_ms=score_max_wait_ms,
            max_queue=score_queue_size,
            event_callback=event_callback,
        )
        self._closed = False
        atexit.register(self.close)

    def _emit(self, event: str, metadata: dict[str, Any], **extra: Any) -> None:
        if self.event_callback is None:
            return
        try:
            self.event_callback({"event": event, "time": time.time(), **metadata, **extra})
        except Exception:
            pass

    def submit(self, completion: str, gt_path: str | None, metadata: dict[str, Any] | None = None) -> Future[EnvironmentResult]:
        if self._closed:
            raise RuntimeError("AsyncRenderScorePipeline is closed")
        meta = dict(metadata or {})
        final: Future[EnvironmentResult] = Future()
        self.pending.acquire()  # producer backpressure if environment falls behind
        submit_t = time.monotonic()
        self._emit("render_queued", meta)
        render_future = self.render_pool.submit(self.renderer.render, completion)

        def finish(result: EnvironmentResult) -> None:
            if not final.done():
                final.set_result(result)
            self.pending.release()

        def on_render_done(fut: Future[RenderResult]) -> None:
            try:
                render = fut.result()
            except Exception as exc:
                render = RenderResult(False, None, "unknown", error=f"renderer_exception:{exc}")
            self._emit(
                "render_done",
                meta,
                compile_ok=render.ok,
                compiler=render.compiler,
                render_s=render.elapsed_s,
                render_error=render.error,
                cache_hit=render.cache_hit,
                code_hash=render.code_hash,
            )
            if not render.ok or not render.png_path or not gt_path or not Path(gt_path).exists():
                finish(
                    EnvironmentResult(
                        render=render,
                        reward=self.compile_failure_reward,
                        total_elapsed_s=time.monotonic() - submit_t,
                    )
                )
                return

            try:
                score_future = self.scorer.submit(render.png_path, gt_path, meta)
            except Exception:
                finish(
                    EnvironmentResult(
                        render=render,
                        reward=self.compile_failure_reward,
                        total_elapsed_s=time.monotonic() - submit_t,
                    )
                )
                return

            def on_score_done(sfut: Future[tuple[float, float]]) -> None:
                try:
                    score, score_s = sfut.result()
                except Exception as exc:
                    self._emit("score_fallback", meta, error=repr(exc))
                    score, score_s = self.compile_failure_reward, 0.0
                finish(
                    EnvironmentResult(
                        render=render,
                        reward=float(score),
                        score_elapsed_s=float(score_s),
                        total_elapsed_s=time.monotonic() - submit_t,
                    )
                )

            score_future.add_done_callback(on_score_done)

        render_future.add_done_callback(on_render_done)
        return final

    def evaluate_many(
        self,
        completions: list[str],
        gt_paths: list[str | None],
        metadata: list[dict[str, Any]] | None = None,
    ) -> list[EnvironmentResult]:
        if len(completions) != len(gt_paths):
            raise ValueError("completions and gt_paths must have equal length")
        metas = metadata or [{} for _ in completions]
        if len(metas) != len(completions):
            raise ValueError("metadata and completions must have equal length")
        futures = [self.submit(c, gt, m) for c, gt, m in zip(completions, gt_paths, metas, strict=True)]
        # Synchronization occurs only here, at the reward/update boundary. Render
        # and score stages are already overlapped while these futures are active.
        return [f.result() for f in futures]

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.render_pool.shutdown(wait=False, cancel_futures=False)
        self.scorer.close()
