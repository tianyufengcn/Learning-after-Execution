from __future__ import annotations

import copy
import inspect
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .modeling import load_policy_and_processor
from .prompt_contract import DIRECT_I2T_PROMPT, IMAGE_SIZE, PROMPT_VERSION, contract_dict
from .reward import RenderAwareReward


def _supported_kwargs(cls, values: dict[str, Any]) -> dict[str, Any]:
    """Filter config keys for modest TRL API drift while keeping settings explicit."""
    try:
        sig = inspect.signature(cls)
        params = sig.parameters
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
            return values
        accepted = set(params)
        dropped = sorted(set(values) - accepted)
        if dropped:
            print(f"[compat] {cls.__name__} does not expose these config keys; ignoring: {dropped}")
        return {k: v for k, v in values.items() if k in accepted}
    except Exception:
        return values


def _validate_dataset_contract(dataset, max_checks: int = 64) -> None:
    required = {"prompt", "image", "gt_image_path", "sample_id"}
    missing = required - set(dataset.column_names)
    if missing:
        raise KeyError(f"GRPO dataset is missing columns: {sorted(missing)}")
    if "prompt_version" in dataset.column_names:
        bad = []
        for row in dataset.select(range(min(max_checks, len(dataset)))):
            if row.get("prompt_version") != PROMPT_VERSION:
                bad.append((row.get("sample_id"), row.get("prompt_version")))
        if bad:
            raise ValueError(f"Dataset prompt contract does not match {PROMPT_VERSION}: {bad[:5]}")


def run_training(cfg: dict[str, Any]) -> None:
    from datasets import load_from_disk
    from trl import GRPOConfig, GRPOTrainer

    seed = int(cfg.get("seed", 20260821))
    # Reward model is constructed before GRPOTrainer creates Accelerator. Bind
    # each accelerate process first so every rank does not allocate RSim on GPU0.
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        print(f"[dist] LOCAL_RANK={local_rank} cuda_device={torch.cuda.current_device()}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    data_cfg = cfg["data"]
    dataset = load_from_disk(data_cfg["dataset_path"])
    dataset = dataset.shuffle(seed=seed)
    if data_cfg.get("max_samples"):
        dataset = dataset.select(range(min(int(data_cfg["max_samples"]), len(dataset))))
    _validate_dataset_contract(dataset)
    print(f"[data] GRPO prompts={len(dataset)} columns={dataset.column_names}")

    model_cfg = copy.deepcopy(cfg["model"])
    model_cfg.setdefault("image_size", IMAGE_SIZE)
    model_cfg.setdefault("min_pixels", IMAGE_SIZE * IMAGE_SIZE)
    model_cfg.setdefault("max_pixels", IMAGE_SIZE * IMAGE_SIZE)
    model, processor = load_policy_and_processor(model_cfg)

    g = cfg["grpo"]
    output_dir = Path(g["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    max_completion = int(g.get("max_completion_length", 8192))
    reward_cfg = copy.deepcopy(cfg["reward"])
    reward_cfg["max_completion_length"] = max_completion
    reward_cfg["prompt_version"] = PROMPT_VERSION
    try:
        eos = getattr(getattr(processor, "tokenizer", None), "eos_token_id", None)
        if eos is not None:
            reward_cfg["eos_token_id"] = int(eos)
    except Exception:
        pass
    reward_func = RenderAwareReward(reward_cfg)

    generation_kwargs = {
        "do_sample": True,
        "temperature": float(g.get("temperature", 1.0)),
        "top_p": float(g.get("top_p", 0.9)),
    }
    grpo_kwargs = {
        "output_dir": str(output_dir),
        "run_name": g.get("run_name", output_dir.name),
        "bf16": bool(g.get("bf16", True)),
        "optim": g.get("optim", "adamw_torch"),
        "learning_rate": float(g.get("learning_rate", 5e-6)),
        "weight_decay": float(g.get("weight_decay", 1e-2)),
        "warmup_ratio": float(g.get("warmup_ratio", 0.0)),
        "lr_scheduler_type": g.get("lr_scheduler_type", "constant"),
        "max_grad_norm": float(g.get("max_grad_norm", 1.0)),
        "per_device_train_batch_size": int(g.get("per_device_train_batch_size", 2)),
        "gradient_accumulation_steps": int(g.get("gradient_accumulation_steps", 8)),
        "num_train_epochs": float(g.get("num_train_epochs", 1)),
        "num_generations": int(g.get("num_generations", 8)),
        "max_prompt_length": g.get("max_prompt_length", 4096),
        "max_completion_length": max_completion,
        "loss_type": g.get("loss_type", "dr_grpo"),
        "scale_rewards": bool(g.get("scale_rewards", False)),
        "beta": float(g.get("beta", 0.0)),
        "epsilon": float(g.get("epsilon", 0.2)),
        "epsilon_high": float(g.get("epsilon_high", 0.28)),
        "mask_truncated_completions": bool(g.get("mask_truncated_completions", True)),
        "generation_kwargs": generation_kwargs,
        "temperature": float(g.get("temperature", 1.0)),
        "top_p": float(g.get("top_p", 0.9)),
        "save_strategy": "steps",
        "save_steps": int(g.get("save_steps", 25)),
        "save_total_limit": int(g.get("save_total_limit", 4)),
        "logging_steps": int(g.get("logging_steps", 1)),
        "report_to": g.get("report_to", ["tensorboard"]),
        "gradient_checkpointing": bool(g.get("gradient_checkpointing", False)),
        "remove_unused_columns": False,
        "log_completions": bool(g.get("log_completions", True)),
    }
    for optional_key in (
        "use_vllm",
        "vllm_mode",
        "use_transformers_continuous_batching",
        "steps_per_generation",
        "generation_batch_size",
    ):
        if g.get(optional_key) is not None:
            grpo_kwargs[optional_key] = g.get(optional_key)

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    num_generations = int(g.get("num_generations", 8))
    per_device = int(g.get("per_device_train_batch_size", 2))
    steps_per_generation = g.get("steps_per_generation")
    generation_batch_size = g.get("generation_batch_size")
    if generation_batch_size is None:
        steps = int(steps_per_generation if steps_per_generation is not None else g.get("gradient_accumulation_steps", 1))
        generation_batch_size = per_device * world_size * steps
    else:
        generation_batch_size = int(generation_batch_size)
    if generation_batch_size % num_generations != 0:
        raise ValueError(
            f"generation_batch_size={generation_batch_size} must be divisible by num_generations={num_generations}"
        )
    print(
        f"[grpo-wave] generation_batch_size={generation_batch_size} "
        f"num_generations={num_generations} unique_prompts_per_wave={generation_batch_size // num_generations}"
    )

    run_contract = {
        "seed": seed,
        "dataset_path": data_cfg["dataset_path"],
        "dataset_rows": len(dataset),
        "prompt_contract": contract_dict(),
        "grpo": {
            "num_generations": num_generations,
            "max_completion_length": max_completion,
            "temperature": float(g.get("temperature", 1.0)),
            "top_p": float(g.get("top_p", 0.9)),
            "loss_type": g.get("loss_type", "dr_grpo"),
            "mask_truncated_completions": bool(g.get("mask_truncated_completions", True)),
        },
        "environment": reward_cfg.get("environment", {}),
    }
    (output_dir / "run_contract.json").write_text(json.dumps(run_contract, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[contract]", json.dumps(run_contract["prompt_contract"], ensure_ascii=False))
    print("[generation]", json.dumps(run_contract["grpo"], ensure_ascii=False))

    training_args = GRPOConfig(**_supported_kwargs(GRPOConfig, grpo_kwargs))
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "reward_funcs": reward_func,
        "train_dataset": dataset,
        "processing_class": processor,
    }
    trainer = GRPOTrainer(**_supported_kwargs(GRPOTrainer, trainer_kwargs))

    try:
        from transformers import TrainerCallback

        class StepMetricsCallback(TrainerCallback):
            """Logging-only per-optimizer-step metrics (never touches the loss/reward path)."""

            def __init__(self, trainer_ref, out_path: Path) -> None:
                self.trainer_ref = trainer_ref
                self.out_path = Path(out_path)
                self._last_step_end = time.monotonic()
                self._step_delta = 0.0
                self._gen_wave_s = 0.0
                orig_gen = trainer_ref._generate_and_score_completions

                def timed_gen(*args: Any, **kwargs: Any) -> Any:
                    t0 = time.monotonic()
                    try:
                        return orig_gen(*args, **kwargs)
                    finally:
                        self._gen_wave_s = time.monotonic() - t0

                trainer_ref._generate_and_score_completions = timed_gen

            def on_step_end(self, args, state, control, **kwargs) -> None:
                now = time.monotonic()
                self._step_delta = now - self._last_step_end
                self._last_step_end = now

            def on_log(self, args, state, control, logs=None, **kwargs) -> None:
                try:
                    if not self.trainer_ref.accelerator.is_main_process:
                        return
                except Exception:
                    return
                logs = logs or {}
                gen_s = self._gen_wave_s
                self._gen_wave_s = 0.0
                train_step_s = logs.get("train/step_time") or 0.0
                backward_s = max(0.0, float(train_step_s) - gen_s)
                env_s = logs.get("reward/environment_wall_s")
                row = {
                    "event": "step_metrics",
                    "time": time.time(),
                    "global_step": int(getattr(state, "global_step", -1)),
                    "epoch": logs.get("epoch"),
                    "loss": logs.get("loss"),
                    "grad_norm": logs.get("grad_norm"),
                    "learning_rate": logs.get("learning_rate"),
                    "step_time": round(self._step_delta, 4),
                    "generation_time": round(gen_s, 4),
                    "environment_time": round(float(env_s), 4) if env_s is not None else None,
                    "backward_time": round(backward_s, 4),
                    "train_step_time_total": round(float(train_step_s), 4),
                    "completions_mean_length": logs.get("completions/mean_length"),
                    "completions_min_length": logs.get("completions/min_length"),
                    "completions_max_length": logs.get("completions/max_length"),
                    "completions_clipped_ratio": logs.get("completions/clipped_ratio"),
                    "frac_reward_zero_std": logs.get("frac_reward_zero_std"),
                    "reward": logs.get("reward"),
                    "reward_std": logs.get("reward_std"),
                    "kl": logs.get("kl"),
                    "clip_ratio": logs.get("clip_ratio"),
                }
                self.out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.out_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                    f.flush()

        step_metrics_path = Path(reward_cfg.get("log_dir", g["output_dir"])) / "step_metrics_rank0.jsonl"
        trainer.add_callback(StepMetricsCallback(trainer, step_metrics_path))
        print(f"[logging] step metrics -> {step_metrics_path}")
    except Exception as exc:
        print(f"[warn] step-metrics callback not installed: {exc}")

    resume = g.get("resume_from_checkpoint")
    trainer.train(resume_from_checkpoint=resume if resume else None)
    trainer.save_model(str(output_dir / "final"))
    try:
        processor.save_pretrained(str(output_dir / "final"))
    except Exception as exc:
        print(f"[warn] processor save failed: {exc}")
