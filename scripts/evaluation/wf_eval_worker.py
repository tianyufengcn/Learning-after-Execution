"""Resident Qwen3-VL workflow evaluation worker.

12 independent processes (4 GPUs x 3).  Each worker loads its full policy
(base model + same-policy PEFT adapter) exactly once and stays resident for the
whole evaluation.  A worker claims one sample at a time from the shared SQLite
queue and owns the complete lifecycle:

    Stage0 (direct) -> compile/render -> RSim score
    -> route (visual_revision | repair_or_complete)
    -> Stage1 (revision/repair) -> compile/render -> RSim score
    -> atomic result file -> mark done

The generation path reuses the v0.3.0 training code (encode_prompt,
StopStringCriteria on the first \\end{document}, _trim_completion,
_truncate_to_first_end_document) with microbatch = 1 per worker.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# sqlite3 must load before torch: torch's import manipulates LD_LIBRARY_PATH
# and then _sqlite3 fails to resolve libstdc++ (CXXABI_1.3.15).
from wf_eval_common import GENERATION, RENDERER, atomic_json, safe_sample_id, stable_seed
from wf_eval_state import EvalState

import torch


def _heartbeat_loop(
    state: EvalState,
    worker_id: str,
    policy: str,
    current: list[tuple[str, str] | None],
    stop: threading.Event,
) -> None:
    while not stop.wait(15):
        try:
            key = current[0]
            state.worker_heartbeat(worker_id, key[1] if key else None)
            if key:
                state.heartbeat(policy, key[1], worker_id)
        except Exception:
            pass


class ResidentWorkflowGenerator:
    """Same-policy resident Qwen3-VL generator (transformers, microbatch 1)."""

    def __init__(
        self,
        base_model: str,
        adapter: str | None,
        device: torch.device,
        generation_cfg: dict[str, Any],
    ) -> None:
        from transformers import (
            AutoProcessor,
            Qwen3VLForConditionalGeneration,
            StopStringCriteria,
            StoppingCriteriaList,
        )

        self.device = device
        self.temperature = float(generation_cfg["temperature"])
        self.top_p = float(generation_cfg["top_p"])
        self.repetition_penalty = float(generation_cfg["repetition_penalty"])
        self.stage0_max = int(generation_cfg["stage0_max_completion_length"])
        self.stage1_max = int(generation_cfg["stage1_max_completion_length"])
        self.stop_on_end_document = bool(generation_cfg["stop_on_end_document"])

        t0 = time.time()
        self.processor = AutoProcessor.from_pretrained(
            base_model,
            trust_remote_code=True,
            min_pixels=200704,
            max_pixels=200704,
        )
        tok = getattr(self.processor, "tokenizer", None)
        if tok is not None:
            if tok.pad_token_id is None:
                tok.pad_token = tok.eos_token
            tok.padding_side = "left"
        print(f"[worker] processor loaded in {time.time()-t0:.1f}s", flush=True)

        t0 = time.time()
        kwargs: dict[str, Any] = {
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": True,
            "device_map": {"": device},
            "low_cpu_mem_usage": True,
        }
        try:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                base_model, attn_implementation="flash_attention_2", **kwargs
            )
            self.attn = "flash_attention_2"
        except Exception as exc:
            print(f"[worker] flash_attention_2 unavailable ({exc!r}); falling back to default", flush=True)
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(base_model, **kwargs)
            self.attn = "default"
        print(f"[worker] base model loaded in {time.time()-t0:.1f}s attn={self.attn}", flush=True)

        if adapter:
            from peft import PeftModel

            t0 = time.time()
            self.model = PeftModel.from_pretrained(self.model, adapter, is_trainable=False)
            print(f"[worker] adapter {adapter} loaded in {time.time()-t0:.1f}s", flush=True)
        self.model.eval()
        self.model.config.use_cache = True
        self.ready = True

    @property
    def tokenizer(self):
        return self.processor.tokenizer

    def _generate_one(self, spec, stage: int) -> Any:
        from i2t_workflow_grpo.generation import (
            END_DOCUMENT,
            _trim_completion,
            _truncate_to_first_end_document,
            GeneratedAction,
            collate_prompt_encodings,
            encode_prompt,
        )
        from transformers import StopStringCriteria, StoppingCriteriaList

        max_new = self.stage0_max if stage == 0 else self.stage1_max
        enc = encode_prompt(self.processor, spec)
        pad_id = int(self.tokenizer.pad_token_id)
        batch = collate_prompt_encodings([enc], pad_id, side="left")
        batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
        padded_prompt_len = int(batch["input_ids"].shape[1])
        kwargs: dict[str, Any] = {
            **batch,
            "do_sample": True,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
            "max_new_tokens": max_new,
            "use_cache": True,
            "pad_token_id": pad_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if self.stop_on_end_document:
            kwargs["stopping_criteria"] = StoppingCriteriaList(
                [StopStringCriteria(self.tokenizer, [END_DOCUMENT])]
            )
        started = time.monotonic()
        with torch.inference_mode():
            out = self.model.generate(**kwargs)
        elapsed = time.monotonic() - started
        new = out[:, padded_prompt_len:]
        row = new[0]
        ids, hit_max = _trim_completion(row, self.tokenizer.eos_token_id, pad_id, max_new)
        ids, text, complete = _truncate_to_first_end_document(ids, self.tokenizer)
        if self.stop_on_end_document and complete:
            hit_max = False
        return GeneratedAction(
            spec=spec,
            prompt=enc,
            completion_ids=ids,
            text=text,
            hit_max_tokens=hit_max,
            ends_with_document=complete,
            stopped_on_end_document=self.stop_on_end_document and complete,
            generation_batch_elapsed_s=float(elapsed),
        )

    def generate(self, spec, stage: int) -> Any:
        return self._generate_one(spec, stage)


class RSimClient:
    """Filesystem request/result queue client for the per-GPU resident scorer."""

    def __init__(self, queue_dir: str | Path, timeout_s: float = 900.0) -> None:
        self.queue_dir = Path(queue_dir)
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        self.timeout_s = float(timeout_s)

    def score(
        self,
        *,
        sample_id: str,
        stage: str,
        policy: str,
        worker_id: str,
        gt_path: str,
        render_path: str,
    ) -> dict[str, Any]:
        request_id = f"{worker_id}-{uuid.uuid4().hex[:12]}"
        req = {
            "request_id": request_id,
            "sample_id": sample_id,
            "stage": stage,
            "policy": policy,
            "worker_id": worker_id,
            "gt_path": str(gt_path),
            "render_path": str(render_path),
            "created_at": time.time(),
        }
        req_tmp = self.queue_dir / f".{request_id}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}.json"
        req_tmp.write_text(json.dumps(req, ensure_ascii=False), encoding="utf-8")
        os.replace(req_tmp, self.queue_dir / f"req_{request_id}.json")
        res_path = self.queue_dir / f"res_{request_id}.json"
        deadline = time.time() + self.timeout_s
        while time.time() < deadline:
            if res_path.exists():
                try:
                    data = json.loads(res_path.read_text(encoding="utf-8"))
                    res_path.unlink(missing_ok=True)
                    return data
                except Exception:
                    time.sleep(0.2)
                    continue
            time.sleep(0.5)
        raise TimeoutError(f"RSim score timeout for {sample_id} stage={stage} (req {request_id})")


def _generation_error_stage(error: str) -> dict[str, Any]:
    return {
        "prompt_kind": None,
        "generation": {"status": "failed", "error": error, "elapsed": None},
        "compile_success": False,
        "compiler": None,
        "render_success": False,
        "render_path": None,
        "render_error": error,
        "render_elapsed": None,
        "rsim": None,
        "rsim_status": "not_scored",
        "rsim_elapsed": None,
        "tex_path": None,
        "prompt_path": None,
    }


def _is_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda oom" in msg or "cuda error: out of memory" in msg


def _save_prompt(processor, spec, path: Path) -> None:
    try:
        text = processor.apply_chat_template(
            spec.messages(), tokenize=False, add_generation_prompt=True
        )
        path.write_text(text, encoding="utf-8")
    except Exception as exc:
        path.write_text(f"prompt render failed: {exc!r}\n{json.dumps(spec.to_dict())}", encoding="utf-8")


def run_sample(
    *,
    args: argparse.Namespace,
    generator: ResidentWorkflowGenerator,
    renderer,
    rsim: RSimClient,
    policy: str,
    sample: dict[str, Any],
) -> dict[str, Any]:
    sid = str(sample["sample_id"])
    gt = str(sample["gt_image_path"])
    safe = safe_sample_id(sid)
    started = time.time()

    from i2t_workflow_grpo.generation import PromptSpec
    from i2t_workflow_grpo.workflow import failure_status

    tex0_dir = Path(args.tex_root) / "stage0"
    tex1_dir = Path(args.tex_root) / "stage1"
    ren0_dir = Path(args.render_root) / "stage0"
    ren1_dir = Path(args.render_root) / "stage1"
    for d in (tex0_dir, tex1_dir, ren0_dir, ren1_dir):
        d.mkdir(parents=True, exist_ok=True)

    record: dict[str, Any] = {
        "sample_id": sid,
        "policy": policy,
        "worker_id": args.worker_id,
        "gpu": args.gpu,
        "route": None,
        "transition": None,
        "started_at": started,
        "finished_at": None,
        "elapsed_sec": None,
        "error": None,
        "stage0": None,
        "stage1": None,
    }

    # ------------------------------------------------------------------ stage0
    t0 = time.time()
    stage0 = _generation_error_stage("not_started")
    action0 = None
    try:
        spec0 = PromptSpec("direct", gt)
        action0 = generator.generate(spec0, stage=0)
        stage0 = {
            "prompt_kind": "direct",
            "generation": {
                "status": "success",
                "elapsed": action0.generation_batch_elapsed_s,
                "tokens": action0.completion_tokens,
                "hit_max_tokens": action0.hit_max_tokens,
                "ends_with_document": action0.ends_with_document,
                "stopped_on_end_document": action0.stopped_on_end_document,
            },
            "compile_success": False,
            "compiler": None,
            "render_success": False,
            "render_path": None,
            "render_error": None,
            "render_elapsed": None,
            "rsim": None,
            "rsim_status": "pending",
            "rsim_elapsed": None,
            "score_error": None,
            "tex_path": None,
            "prompt_path": None,
        }
        tex_path = tex0_dir / f"{safe}.tex"
        tex_path.write_text(action0.text, encoding="utf-8")
        prompt_path = tex0_dir / f"{safe}.prompt.txt"
        _save_prompt(generator.processor, spec0, prompt_path)
        gen_json = {
            "sample_id": sid,
            "policy": policy,
            "stage": 0,
            "worker_id": args.worker_id,
            "seed": args.seed_for_sample,
            "tokens": action0.completion_tokens,
            "hit_max_tokens": action0.hit_max_tokens,
            "ends_with_document": action0.ends_with_document,
            "stopped_on_end_document": action0.stopped_on_end_document,
            "temperature": generator.temperature,
            "top_p": generator.top_p,
            "repetition_penalty": generator.repetition_penalty,
            "max_new_tokens": generator.stage0_max,
        }
        atomic_json(tex0_dir / f"{safe}.generation.json", gen_json)
        stage0["tex_path"] = str(tex_path)
        stage0["prompt_path"] = str(prompt_path)
        print(
            f"[{args.worker_id} gpu={args.gpu}] stage0_generate sid={sid} "
            f"tokens={action0.completion_tokens} hit_max={action0.hit_max_tokens} "
            f"complete={action0.ends_with_document}",
            flush=True,
        )
        rr = renderer.render(action0.text)
        stage0["compile_success"] = bool(rr.compiler is not None)
        stage0["compiler"] = rr.compiler
        stage0["render_success"] = bool(rr.ok and rr.png_path)
        stage0["render_path"] = str(rr.png_path) if rr.png_path else None
        stage0["render_error"] = rr.error
        stage0["render_elapsed"] = round(rr.elapsed_s, 3)
        if stage0["render_success"]:
            ren0 = ren0_dir / f"{safe}.png"
            shutil.copy2(rr.png_path, ren0)
            stage0["render_path"] = str(ren0)
            t_score = time.time()
            resp = rsim.score(
                sample_id=sid,
                stage="stage0",
                policy=policy,
                worker_id=args.worker_id,
                gt_path=gt,
                render_path=str(ren0),
            )
            stage0["rsim_elapsed"] = round(time.time() - t_score, 3)
            if resp.get("status") == "success":
                stage0["rsim"] = float(resp["rsim"])
                stage0["rsim_status"] = "success"
            else:
                stage0["rsim"] = 0.0
                stage0["rsim_status"] = "failed"
                stage0["score_error"] = resp.get("error")
        else:
            stage0["rsim"] = 0.0
            stage0["rsim_status"] = "not_scored"
        print(
            f"[{args.worker_id} gpu={args.gpu}] stage0_render sid={sid} "
            f"compile={stage0['compile_success']} render={stage0['render_success']} "
            f"error={stage0['render_error']} rsim={stage0['rsim']}",
            flush=True,
        )
    except BaseException as exc:
        if _is_oom(exc):
            print(
                f"[{args.worker_id} gpu={args.gpu}] stage0 OOM sid={sid} "
                f"tokens={getattr(action0, 'completion_tokens', None)}; empty_cache + retry once",
                flush=True,
            )
            torch.cuda.empty_cache()
            try:
                spec0 = PromptSpec("direct", gt)
                action0 = generator.generate(spec0, stage=0)
                rr = renderer.render(action0.text)
                stage0 = {
                    "prompt_kind": "direct",
                    "generation": {
                        "status": "success",
                        "elapsed": action0.generation_batch_elapsed_s,
                        "tokens": action0.completion_tokens,
                        "hit_max_tokens": action0.hit_max_tokens,
                        "ends_with_document": action0.ends_with_document,
                        "stopped_on_end_document": action0.stopped_on_end_document,
                    },
                    "compile_success": bool(rr.compiler is not None),
                    "compiler": rr.compiler,
                    "render_success": bool(rr.ok and rr.png_path),
                    "render_path": str(rr.png_path) if rr.png_path else None,
                    "render_error": rr.error,
                    "render_elapsed": round(rr.elapsed_s, 3),
                    "rsim": None,
                    "rsim_status": "pending",
                    "rsim_elapsed": None,
                    "score_error": None,
                    "tex_path": None,
                    "prompt_path": None,
                }
                tex_path = tex0_dir / f"{safe}.tex"
                tex_path.write_text(action0.text, encoding="utf-8")
                prompt_path = tex0_dir / f"{safe}.prompt.txt"
                _save_prompt(generator.processor, spec0, prompt_path)
                stage0["tex_path"] = str(tex_path)
                stage0["prompt_path"] = str(prompt_path)
                if stage0["render_success"]:
                    ren0 = ren0_dir / f"{safe}.png"
                    shutil.copy2(rr.png_path, ren0)
                    stage0["render_path"] = str(ren0)
                    resp = rsim.score(
                        sample_id=sid, stage="stage0", policy=policy,
                        worker_id=args.worker_id, gt_path=gt, render_path=str(ren0),
                    )
                    if resp.get("status") == "success":
                        stage0["rsim"] = float(resp["rsim"])
                        stage0["rsim_status"] = "success"
                    else:
                        stage0["rsim"] = 0.0
                        stage0["rsim_status"] = "failed"
                        stage0["score_error"] = resp.get("error")
                else:
                    stage0["rsim"] = 0.0
                    stage0["rsim_status"] = "not_scored"
            except BaseException as exc2:
                stage0 = _generation_error_stage(f"stage0 OOM retry failed: {type(exc2).__name__}: {exc2}")
                record["error"] = stage0["generation"]["error"]
        else:
            stage0 = _generation_error_stage(f"{type(exc).__name__}: {exc}")
            record["error"] = stage0["generation"]["error"]
            print(
                f"[{args.worker_id} gpu={args.gpu}] stage0 ERROR sid={sid} {type(exc).__name__}: {exc}",
                flush=True,
            )
    stage0["elapsed_sec"] = round(time.time() - t0, 3)

    # ------------------------------------------------------------------ route
    s0_ok = bool(stage0["render_success"] and stage0["render_path"])
    route = "visual_revision" if s0_ok else "repair_or_complete"
    record["route"] = route
    record["stage0"] = stage0

    stage1 = _generation_error_stage("not_started")
    action1 = None
    if stage0["generation"]["status"] == "success" and action0 is not None:
        t1 = time.time()
        try:
            if s0_ok:
                spec1 = PromptSpec(
                    "revision",
                    gt,
                    current_image_path=stage0["render_path"],
                    current_tikz=action0.text,
                )
            else:
                status = failure_status(
                    stage0["render_error"],
                    stage0["generation"].get("hit_max_tokens", False),
                    stage0["generation"].get("ends_with_document", False),
                )
                spec1 = PromptSpec(
                    "repair",
                    gt,
                    current_tikz=action0.text,
                    failure_status=status,
                )
            action1 = generator.generate(spec1, stage=1)
            stage1 = {
                "prompt_kind": "revision" if s0_ok else "repair",
                "generation": {
                    "status": "success",
                    "elapsed": action1.generation_batch_elapsed_s,
                    "tokens": action1.completion_tokens,
                    "hit_max_tokens": action1.hit_max_tokens,
                    "ends_with_document": action1.ends_with_document,
                    "stopped_on_end_document": action1.stopped_on_end_document,
                },
                "compile_success": False,
                "compiler": None,
                "render_success": False,
                "render_path": None,
                "render_error": None,
                "render_elapsed": None,
                "rsim": None,
                "rsim_status": "pending",
                "rsim_elapsed": None,
                "score_error": None,
                "tex_path": None,
                "prompt_path": None,
            }
            tex_path = tex1_dir / f"{safe}.tex"
            tex_path.write_text(action1.text, encoding="utf-8")
            prompt_path = tex1_dir / f"{safe}.prompt.txt"
            _save_prompt(generator.processor, spec1, prompt_path)
            gen_json = {
                "sample_id": sid,
                "policy": policy,
                "stage": 1,
                "worker_id": args.worker_id,
                "seed": args.seed_for_sample,
                "route": route,
                "failure_status": None if s0_ok else status,
                "tokens": action1.completion_tokens,
                "hit_max_tokens": action1.hit_max_tokens,
                "ends_with_document": action1.ends_with_document,
                "stopped_on_end_document": action1.stopped_on_end_document,
                "temperature": generator.temperature,
                "top_p": generator.top_p,
                "repetition_penalty": generator.repetition_penalty,
                "max_new_tokens": generator.stage1_max,
            }
            atomic_json(tex1_dir / f"{safe}.generation.json", gen_json)
            stage1["tex_path"] = str(tex_path)
            stage1["prompt_path"] = str(prompt_path)
            print(
                f"[{args.worker_id} gpu={args.gpu}] stage1_generate sid={sid} "
                f"route={route} tokens={action1.completion_tokens} "
                f"hit_max={action1.hit_max_tokens} complete={action1.ends_with_document}",
                flush=True,
            )
            rr = renderer.render(action1.text)
            stage1["compile_success"] = bool(rr.compiler is not None)
            stage1["compiler"] = rr.compiler
            stage1["render_success"] = bool(rr.ok and rr.png_path)
            stage1["render_path"] = str(rr.png_path) if rr.png_path else None
            stage1["render_error"] = rr.error
            stage1["render_elapsed"] = round(rr.elapsed_s, 3)
            if stage1["render_success"]:
                ren1 = ren1_dir / f"{safe}.png"
                shutil.copy2(rr.png_path, ren1)
                stage1["render_path"] = str(ren1)
                t_score = time.time()
                resp = rsim.score(
                    sample_id=sid,
                    stage="stage1",
                    policy=policy,
                    worker_id=args.worker_id,
                    gt_path=gt,
                    render_path=str(ren1),
                )
                stage1["rsim_elapsed"] = round(time.time() - t_score, 3)
                if resp.get("status") == "success":
                    stage1["rsim"] = float(resp["rsim"])
                    stage1["rsim_status"] = "success"
                else:
                    stage1["rsim"] = 0.0
                    stage1["rsim_status"] = "failed"
                    stage1["score_error"] = resp.get("error")
            else:
                stage1["rsim"] = 0.0
                stage1["rsim_status"] = "not_scored"
            print(
                f"[{args.worker_id} gpu={args.gpu}] stage1_render sid={sid} "
                f"compile={stage1['compile_success']} render={stage1['render_success']} "
                f"error={stage1['render_error']} rsim={stage1['rsim']}",
                flush=True,
            )
        except BaseException as exc:
            if _is_oom(exc):
                print(
                    f"[{args.worker_id} gpu={args.gpu}] stage1 OOM sid={sid}; empty_cache + retry once",
                    flush=True,
                )
                torch.cuda.empty_cache()
                try:
                    action1 = generator.generate(spec1, stage=1)
                    rr = renderer.render(action1.text)
                    stage1 = {
                        "prompt_kind": "revision" if s0_ok else "repair",
                        "generation": {
                            "status": "success",
                            "elapsed": action1.generation_batch_elapsed_s,
                            "tokens": action1.completion_tokens,
                            "hit_max_tokens": action1.hit_max_tokens,
                            "ends_with_document": action1.ends_with_document,
                            "stopped_on_end_document": action1.stopped_on_end_document,
                        },
                        "compile_success": bool(rr.compiler is not None),
                        "compiler": rr.compiler,
                        "render_success": bool(rr.ok and rr.png_path),
                        "render_path": str(rr.png_path) if rr.png_path else None,
                        "render_error": rr.error,
                        "render_elapsed": round(rr.elapsed_s, 3),
                        "rsim": None,
                        "rsim_status": "pending",
                        "rsim_elapsed": None,
                        "score_error": None,
                        "tex_path": None,
                        "prompt_path": None,
                    }
                    tex_path = tex1_dir / f"{safe}.tex"
                    tex_path.write_text(action1.text, encoding="utf-8")
                    prompt_path = tex1_dir / f"{safe}.prompt.txt"
                    _save_prompt(generator.processor, spec1, prompt_path)
                    stage1["tex_path"] = str(tex_path)
                    stage1["prompt_path"] = str(prompt_path)
                    if stage1["render_success"]:
                        ren1 = ren1_dir / f"{safe}.png"
                        shutil.copy2(rr.png_path, ren1)
                        stage1["render_path"] = str(ren1)
                        resp = rsim.score(
                            sample_id=sid, stage="stage1", policy=policy,
                            worker_id=args.worker_id, gt_path=gt, render_path=str(ren1),
                        )
                        if resp.get("status") == "success":
                            stage1["rsim"] = float(resp["rsim"])
                            stage1["rsim_status"] = "success"
                        else:
                            stage1["rsim"] = 0.0
                            stage1["rsim_status"] = "failed"
                            stage1["score_error"] = resp.get("error")
                    else:
                        stage1["rsim"] = 0.0
                        stage1["rsim_status"] = "not_scored"
                except BaseException as exc2:
                    stage1 = _generation_error_stage(f"stage1 OOM retry failed: {type(exc2).__name__}: {exc2}")
                    record["error"] = stage1["generation"]["error"]
            else:
                stage1 = _generation_error_stage(f"{type(exc).__name__}: {exc}")
                record["error"] = stage1["generation"]["error"]
                print(
                    f"[{args.worker_id} gpu={args.gpu}] stage1 ERROR sid={sid} {type(exc).__name__}: {exc}",
                    flush=True,
                )
        stage1["elapsed_sec"] = round(time.time() - t1, 3)
    else:
        stage1 = _generation_error_stage("stage0_generation_failed; stage1 skipped")
        stage1["elapsed_sec"] = 0.0
        if record["error"] is None:
            record["error"] = "stage0_generation_failed"

    record["stage1"] = stage1
    s1_ok = bool(stage1["render_success"] and stage1["render_path"])
    record["transition"] = ("SS" if s0_ok else "FS") if s1_ok else ("SF" if s0_ok else "FF")
    s0_fz = float(stage0["rsim"] or 0.0)
    s1_fz = float(stage1["rsim"] or 0.0)
    record["stage0_rsim_fixed_zero"] = s0_fz
    record["stage1_rsim_fixed_zero"] = s1_fz
    record["delta_fixed_zero"] = round(s1_fz - s0_fz, 6)
    record["finished_at"] = time.time()
    record["elapsed_sec"] = round(time.time() - started, 3)
    return record


def worker_main(args: argparse.Namespace) -> int:
    torch.set_num_threads(max(1, int(args.cpu_threads)))
    state = EvalState(args.state_db)
    state.register_worker(args.worker_id, role="qwen", pid=args.pid, gpu=args.gpu, policy=args.policy)
    stop = threading.Event()
    current: list[tuple[str, str] | None] = [None]
    threading.Thread(
        target=_heartbeat_loop,
        args=(state, args.worker_id, args.policy, current, stop),
        daemon=True,
    ).start()

    generator: ResidentWorkflowGenerator | None = None
    renderer = None
    rsim = RSimClient(args.rsim_queue_dir, timeout_s=float(args.rsim_timeout))
    results_root = Path(args.results_root)
    results_root.mkdir(parents=True, exist_ok=True)
    processed = 0
    # Progress denominators come from the task queue, i.e. from the manifest the
    # controller scheduled for this policy (Dev-150, Final-945, or a smoke prefix).
    def policy_total() -> int:
        counts = state.counts(args.policy)
        return sum(int(counts.get(key, 0)) for key in ("pending", "running", "done", "failed"))

    try:
        while True:
            task = state.claim(
                args.policy,
                args.worker_id,
                lease_seconds=float(args.lease_seconds),
                max_attempts=int(args.max_attempts),
            )
            if task is None:
                counts = state.counts(args.policy)
                if counts.get("pending", 0) == 0 and counts.get("running", 0) == 0:
                    print(
                        f"[{args.worker_id} gpu={args.gpu}] policy {args.policy} drained; "
                        f"exiting processed={processed}",
                        flush=True,
                    )
                    break
                time.sleep(float(args.poll_seconds))
                continue

            sid = str(task["sample_id"])
            safe = safe_sample_id(sid)
            result_path = results_root / f"{safe}.json"
            if result_path.exists():
                # Crash-resume: the result was fully written before a restart.
                try:
                    existing = json.loads(result_path.read_text(encoding="utf-8"))
                    state.complete(
                        args.policy, sid, args.worker_id,
                        ok=True, result_path=str(result_path), error=None,
                    )
                    print(
                        f"[{args.worker_id} gpu={args.gpu}] resume skip existing result sid={sid}",
                        flush=True,
                    )
                    processed += 1
                    continue
                except Exception:
                    pass

            current[0] = (args.policy, sid)
            print(
                f"[{args.worker_id} gpu={args.gpu}] claim sample={sid} "
                f"completed={processed}/{policy_total()}",
                flush=True,
            )
            t0 = time.time()
            try:
                args.seed_for_sample = stable_seed(int(args.seed), sid)
                if generator is None:
                    device = torch.device("cuda:0")
                    generator = ResidentWorkflowGenerator(
                        args.base_model,
                        adapter=None if args.adapter in (None, "", "none") else args.adapter,
                        device=device,
                        generation_cfg=GENERATION,
                    )
                    from i2t_workflow_grpo.renderer import TikZRenderer

                    renderer = TikZRenderer(
                        cache_dir=args.render_cache_dir,
                        tmp_dir=args.render_tmp_dir,
                        texlive_bin=RENDERER["texlive_bin"],
                        compilers=RENDERER["compilers"],
                        compile_timeout_s=RENDERER["compile_timeout_s"],
                        image_size=RENDERER["image_size"],
                        dpi=RENDERER["dpi"],
                        keep_failed_logs=RENDERER["keep_failed_logs"],
                        cache_failures=RENDERER["cache_failures"],
                        cache_transient_failures=RENDERER["cache_transient_failures"],
                    )
                    free, total = torch.cuda.mem_get_info()
                    print(
                        f"[{args.worker_id} gpu={args.gpu}] model resident policy={args.policy} "
                        f"adapter={args.adapter} reserved_gib={torch.cuda.memory_reserved()/2**30:.2f} "
                        f"free_gib={(total-free)/2**30:.2f}",
                        flush=True,
                    )
                random.seed(args.seed_for_sample)
                torch.manual_seed(args.seed_for_sample)
                torch.cuda.manual_seed_all(args.seed_for_sample)
                record = run_sample(
                    args=args,
                    generator=generator,
                    renderer=renderer,
                    rsim=rsim,
                    policy=args.policy,
                    sample=task,
                )
                atomic_json(result_path, record)
                elapsed = round(time.time() - t0, 3)
                ok = record.get("stage0") is not None and record.get("stage1") is not None
                state.complete(
                    args.policy, sid, args.worker_id,
                    ok=bool(ok),
                    result_path=str(result_path),
                    error=record.get("error"),
                )
                processed += 1
                print(
                    f"[{args.worker_id} gpu={args.gpu}] done sample={sid} "
                    f"transition={record['transition']} route={record['route']} "
                    f"elapsed={elapsed}s completed={processed}/{policy_total()}",
                    flush=True,
                )
            except BaseException as exc:
                print(
                    f"[{args.worker_id} gpu={args.gpu}] SAMPLE FATAL sid={sid} "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                traceback.print_exc()
                try:
                    state.complete(
                        args.policy, sid, args.worker_id,
                        ok=False, result_path=None,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                except Exception:
                    pass
            finally:
                current[0] = None
    finally:
        stop.set()
        state.stop_worker(args.worker_id)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--state-db", required=True)
    p.add_argument("--policy", required=True)
    p.add_argument("--worker-id", required=True)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--base-model", required=True)
    p.add_argument("--adapter", default="none")
    p.add_argument("--results-root", required=True)
    p.add_argument("--tex-root", required=True)
    p.add_argument("--render-root", required=True)
    p.add_argument("--rsim-queue-dir", required=True)
    p.add_argument("--render-cache-dir", required=True)
    p.add_argument("--render-tmp-dir", required=True)
    p.add_argument("--seed", type=int, default=20260825)
    p.add_argument("--poll-seconds", type=float, default=3.0)
    p.add_argument("--lease-seconds", type=float, default=1800.0)
    p.add_argument("--max-attempts", type=int, default=3)
    p.add_argument("--rsim-timeout", type=float, default=900.0)
    p.add_argument("--cpu-threads", type=int, default=8)
    p.add_argument("--pid", type=int, default=0)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return worker_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
