from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import torch

from .checkpointing import CheckpointManager
from .config import validate_algorithm_contract
from .distributed_rollout import DistributedTrajectoryCoordinator
from .generation import GeneratedAction, QwenWorkflowGenerator, collate_action_inputs
from .loss import clipped_grpo_token_loss, forward_completion_logps
from .modeling import load_policy_and_processor
from .prompt_contract import IMAGE_SIZE, contract_dict
from .rollout import OnPolicyWorkflowRollout
from .transport import materialize_stage_actions, record_cost


class WorkflowGRPOTrainer:
    """Two-stage on-policy Dr.GRPO with a long-horizon rollout control plane.

    The learner remains synchronous, while rollout is globally load-balanced:
      * one global queue contains all complete trajectory tasks for the step;
      * ranks atomically work-steal tasks until the queue is drained;
      * group credit is computed by root GT, independent of physical worker;
      * Stage-0/Stage-1 actions are token-balanced back across DDP ranks;
      * only after host-side update-ready markers does a short NCCL sync occur.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        from accelerate import Accelerator
        from accelerate.utils import InitProcessGroupKwargs
        from datasets import load_from_disk
        from datetime import timedelta
        from transformers import get_scheduler

        validate_algorithm_contract(cfg)
        self.cfg = cfg
        workflow_cfg = dict(cfg["workflow"])
        timeout_s = int(
            os.environ.get(
                "I2T_WORKFLOW_DIST_TIMEOUT_S",
                workflow_cfg.get("distributed_timeout_s", 1200),
            )
        )
        self.accelerator = Accelerator(
            kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=timeout_s))]
        )
        if torch.cuda.is_available():
            torch.cuda.set_device(self.accelerator.local_process_index)

        seed = int(cfg.get("seed", 20260823))
        rank = int(self.accelerator.process_index)
        random.seed(seed + rank)
        np.random.seed(seed + rank)
        torch.manual_seed(seed + rank)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + rank)

        data_cfg = cfg["data"]
        dataset = load_from_disk(data_cfg["dataset_path"])
        if data_cfg.get("max_samples"):
            dataset = dataset.select(range(min(int(data_cfg["max_samples"]), len(dataset))))
        self._validate_dataset(dataset)
        # Every rank intentionally owns the same deterministic dataset order.
        # Global queue scheduling, not DistributedSampler, assigns rollout work.
        self.dataset = dataset.shuffle(seed=seed)
        self.drop_last = bool(data_cfg.get("drop_last", True))

        model_cfg = dict(cfg["model"])
        model_cfg.setdefault("image_size", IMAGE_SIZE)
        model_cfg.setdefault("min_pixels", IMAGE_SIZE * IMAGE_SIZE)
        model_cfg.setdefault("max_pixels", IMAGE_SIZE * IMAGE_SIZE)
        self.model, self.processor = load_policy_and_processor(model_cfg)

        opt_cfg = cfg["optimization"]
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable,
            lr=float(opt_cfg.get("learning_rate", 5e-6)),
            weight_decay=float(opt_cfg.get("weight_decay", 1e-2)),
        )
        # DeepSpeed needs a concrete per-GPU micro batch size when no dataloader
        # is passed to accelerate.prepare(). The loss microbatch is the natural
        # value (one backward call processes loss_microbatch actions).
        ds_plugin = getattr(self.accelerator.state, "deepspeed_plugin", None)
        if ds_plugin is not None:
            ds_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = int(
                cfg.get("loss", {}).get("microbatch_size", 2)
            )
        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)

        self.prompts_per_device = int(opt_cfg.get("prompts_per_device", 2))
        self.global_prompts_per_step = self.prompts_per_device * int(self.accelerator.num_processes)
        if self.drop_last:
            steps_per_epoch = len(self.dataset) // self.global_prompts_per_step
        else:
            steps_per_epoch = int(math.ceil(len(self.dataset) / self.global_prompts_per_step))
        self.num_train_epochs = float(opt_cfg.get("num_train_epochs", 1.0))
        total_steps = max(1, int(math.ceil(steps_per_epoch * self.num_train_epochs)))
        self.max_steps = int(opt_cfg["max_steps"]) if opt_cfg.get("max_steps") is not None else None
        if self.max_steps is not None:
            total_steps = min(total_steps, self.max_steps)
        self.total_steps = total_steps
        # Action-level optimisation schedule (metadata only). Each rank owns
        # ``prompts_per_device`` root groups of ``group_size`` complete
        # trajectories, so one stage contributes group_size * prompts_per_device
        # actions, consumed in loss microbatches. Under the formal 4-rank run
        # (2 groups/rank, K=8, loss microbatch 2) that is 8 Stage-0 plus
        # 8 Stage-1 microbatches, i.e. 16 real AdamW updates per iteration.
        actions_per_stage_per_rank = int(self.group_size) * int(self.prompts_per_device)
        self.microbatches_per_stage = int(
            math.ceil(actions_per_stage_per_rank / max(1, int(self.loss_microbatch)))
        )
        self.adam_updates_per_training_iteration = 2 * self.microbatches_per_stage
        warmup_steps = int(total_steps * float(opt_cfg.get("warmup_ratio", 0.0)))
        self.scheduler = get_scheduler(
            str(opt_cfg.get("lr_scheduler_type", "constant")),
            optimizer=self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        self.accelerator.register_for_checkpointing(self.scheduler)

        self.output_dir = Path(opt_cfg["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(opt_cfg.get("log_dir", self.output_dir / "logs"))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.step_log = self.log_dir / f"step_metrics_rank{self.accelerator.process_index}.jsonl"

        generation_cfg = dict(cfg["generation"])
        backend = str(generation_cfg.get("backend", "transformers"))
        if backend == "transformers":
            self.generator = QwenWorkflowGenerator(self.model, self.processor, self.accelerator, generation_cfg)
        else:
            raise ValueError(f"Unsupported generation.backend={backend!r}")
        workflow_cfg.setdefault("log_dir", str(self.log_dir))
        self.rollout_engine = OnPolicyWorkflowRollout(self.generator, cfg["reward"], workflow_cfg)

        self.group_size = int(workflow_cfg.get("num_generations", 8))
        self.reward_scale = float(workflow_cfg.get("reward_scale", 10.0))
        self.loss_cfg = cfg.get("loss", {})
        self.eps_low = float(self.loss_cfg.get("epsilon", 0.20))
        self.eps_high = float(self.loss_cfg.get("epsilon_high", 0.28))
        self.mask_truncated = bool(self.loss_cfg.get("mask_truncated_completions", True))
        self.loss_microbatch = int(self.loss_cfg.get("microbatch_size", 2))
        self.stage0_loss_weight = float(self.loss_cfg.get("stage0_weight", 0.5))
        self.stage1_loss_weight = float(self.loss_cfg.get("stage1_weight", 0.5))
        self.max_grad_norm = float(opt_cfg.get("max_grad_norm", 1.0))

        self.coordinator = DistributedTrajectoryCoordinator(
            self.accelerator,
            self.rollout_engine,
            self.processor,
            workflow_cfg,
            output_dir=self.output_dir,
            loss_microbatch=self.loss_microbatch,
        )

        checkpoint_cfg = dict(opt_cfg.get("checkpointing", {}))
        self.checkpoints = CheckpointManager(
            self.accelerator,
            self.model,
            self.processor,
            self.output_dir,
            rolling_every_steps=int(checkpoint_cfg.get("rolling_every_steps", 1)),
            rolling_keep=int(checkpoint_cfg.get("rolling_keep", 2)),
            milestone_every_steps=int(
                checkpoint_cfg.get("milestone_every_steps", opt_cfg.get("save_steps", 25))
            ),
        )

        self.resume_path = opt_cfg.get("resume_from_checkpoint")
        self.start_step = 0
        self._write_contract(model_cfg, total_steps, timeout_s)
        if self.resume_path:
            self._load_resume(Path(self.resume_path))

    def _validate_dataset(self, dataset) -> None:
        required = {"sample_id", "gt_image_path"}
        missing = required - set(dataset.column_names)
        if missing:
            raise KeyError(f"Workflow dataset missing columns: {sorted(missing)}")
        for row in dataset.select(range(min(32, len(dataset)))):
            path = Path(str(row["gt_image_path"]))
            if not path.exists():
                raise FileNotFoundError(path)

    def _write_contract(self, model_cfg: dict[str, Any], total_steps: int, timeout_s: int) -> None:
        if not self.accelerator.is_main_process:
            return
        adapter = model_cfg.get("adapter_path")
        adapter_weights = Path(adapter) / "adapter_model.safetensors" if adapter else None
        payload = {
            "format": "two_stage_on_policy_workflow_grpo_v0.3.0",
            "dataset_path": self.cfg["data"]["dataset_path"],
            "dataset_rows": len(self.dataset),
            "global_prompts_per_step": self.global_prompts_per_step,
            "prompt_contract": contract_dict(),
            "workflow": self.cfg["workflow"],
            "generation": self.cfg["generation"],
            "loss": self.cfg["loss"],
            "optimization": self.cfg["optimization"],
            "distributed_timeout_s": timeout_s,
            "model": {
                "base_model_path": model_cfg.get("base_model_path"),
                "adapter_path": adapter,
            },
            # Training-loop termination is driven by planned_training_iterations.
            # The Adam-update counts are reporting metadata describing the
            # per-microbatch update schedule of the formal stack (see README
            # "Important training semantics"); they never control the loop.
            "planned_training_iterations": total_steps,
            "loss_microbatches_per_stage": self.microbatches_per_stage,
            "adam_updates_per_training_iteration": self.adam_updates_per_training_iteration,
            "planned_adam_updates": total_steps * self.adam_updates_per_training_iteration,
        }
        (self.output_dir / "run_contract.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def _load_resume(self, checkpoint: Path) -> None:
        state_dir = checkpoint / "accelerate_state"
        state_file = checkpoint / "workflow_trainer_state.json"
        if not state_dir.exists() or not state_file.exists():
            raise FileNotFoundError(f"Incomplete workflow checkpoint: {checkpoint}")
        self.accelerator.load_state(str(state_dir))
        state = json.loads(state_file.read_text(encoding="utf-8"))
        self.start_step = int(state.get("global_step", 0))
        self.accelerator.print(f"[resume] checkpoint={checkpoint} global_step={self.start_step}")

    def _rows_for_step(self, step_num: int) -> list[dict[str, Any]]:
        steps_per_epoch = max(1, len(self.dataset) // self.global_prompts_per_step if self.drop_last else math.ceil(len(self.dataset) / self.global_prompts_per_step))
        within_epoch = (int(step_num) - 1) % int(steps_per_epoch)
        start = within_epoch * self.global_prompts_per_step
        end = min(len(self.dataset), start + self.global_prompts_per_step)
        if self.drop_last and end - start < self.global_prompts_per_step:
            raise RuntimeError("drop_last=True produced an incomplete global prompt batch")
        batch = self.dataset.select(range(start, end))
        return [dict(batch[i]) for i in range(len(batch))]

    def _action_batches(self, actions: list[GeneratedAction], advantages: list[float]):
        if len(actions) != len(advantages):
            raise ValueError("actions/advantages mismatch")
        order = sorted(range(len(actions)), key=lambda i: (actions[i].full_sequence_tokens, i))
        sorted_actions = [actions[i] for i in order]
        sorted_advantages = [advantages[i] for i in order]
        for start in range(0, len(sorted_actions), self.loss_microbatch):
            yield (
                sorted_actions[start : start + self.loss_microbatch],
                sorted_advantages[start : start + self.loss_microbatch],
            )

    def _stage_backward(
        self,
        actions: list[GeneratedAction],
        advantages: list[float],
        max_completion_length: int,
        stage_weight: float,
    ) -> dict[str, float]:
        if not actions:
            return {"loss": 0.0, "clip_ratio": 0.0, "active_tokens": 0.0}
        pad_id = int(self.processor.tokenizer.pad_token_id)
        n_total = len(actions)
        loss_sum = clipped_count = active_count = 0.0

        for original_actions, original_advantages in self._action_batches(actions, advantages):
            mb_actions = original_actions
            mb_advantages = original_advantages
            if self.mask_truncated:
                valid = [
                    (action, advantage)
                    for action, advantage in zip(mb_actions, mb_advantages, strict=True)
                    if not action.hit_max_tokens
                ]
                # Removing zero-mask rows is exactly loss-equivalent because the
                # denominator remains the original stage-wide n_total*max_len.
                # If every row is truncated, keep the real model graph so every
                # DDP rank still executes the same communication schedule.
                if valid:
                    mb_actions = [action for action, _ in valid]
                    mb_advantages = [advantage for _, advantage in valid]

            batch_cpu, mask_cpu = collate_action_inputs(mb_actions, pad_id)
            batch = {key: value.to(self.accelerator.device, non_blocking=True) for key, value in batch_cpu.items()}
            mask = mask_cpu.to(self.accelerator.device)
            if self.mask_truncated:
                keep = torch.tensor(
                    [0.0 if action.hit_max_tokens else 1.0 for action in mb_actions],
                    device=self.accelerator.device,
                    dtype=mask.dtype,
                )
                mask = mask * keep[:, None]
            advantage_tensor = torch.tensor(mb_advantages, device=self.accelerator.device, dtype=torch.float32)

            with torch.no_grad(), self.accelerator.autocast():
                old_logps = forward_completion_logps(self.model, batch, mask).detach()
            with self.accelerator.autocast():
                new_logps = forward_completion_logps(self.model, batch, mask)
                per_token, was_clipped = clipped_grpo_token_loss(
                    new_logps,
                    old_logps,
                    advantage_tensor,
                    self.eps_low,
                    self.eps_high,
                )
                denom = max(1.0, float(n_total * max_completion_length))
                loss = stage_weight * (per_token * mask).sum() / denom
            self.accelerator.backward(loss)
            loss_sum += float(loss.detach().item())
            clipped_count += float((was_clipped.float() * mask).sum().detach().item())
            active_count += float(mask.sum().detach().item())

        return {
            "loss": loss_sum,
            "clip_ratio": clipped_count / max(1.0, active_count),
            "active_tokens": active_count,
        }

    @staticmethod
    def _metrics_from_records(records: list[dict[str, Any]]) -> dict[str, float]:
        n = max(1, len(records))
        stage0_success = [row for row in records if row["stage0_ok"]]
        stage0_fail = [row for row in records if not row["stage0_ok"]]
        deltas = [float(row["delta_score"]) for row in stage0_success]
        positive = [delta for delta in deltas if delta > 0]
        negative = [delta for delta in deltas if delta < 0]
        regress = sum(delta < 0 for delta in deltas)
        improve = sum(delta > 0 for delta in deltas)
        rescue = sum((not row["stage0_ok"]) and row["stage1_ok"] for row in records)
        catastrophic = sum(float(row["delta_score"]) < -0.1 for row in stage0_success)
        broke_compile = sum(row["stage0_ok"] and not row["stage1_ok"] for row in records)
        return {
            "n_trajectories": float(len(records)),
            "stage0_compile_rate": sum(bool(row["stage0_ok"]) for row in records) / n,
            "stage1_compile_rate": sum(bool(row["stage1_ok"]) for row in records) / n,
            "stage0_rsim_mean": mean(float(row["stage0_score"]) for row in records) if records else 0.0,
            "stage1_rsim_mean": mean(float(row["stage1_score"]) for row in records) if records else 0.0,
            "delta_rsim_mean": mean(float(row["delta_score"]) for row in records) if records else 0.0,
            "revision_delta_mean": mean(deltas) if deltas else 0.0,
            "revision_improve_rate": improve / max(1, len(stage0_success)),
            "revision_regress_rate": regress / max(1, len(stage0_success)),
            "revision_positive_delta_mean": mean(positive) if positive else 0.0,
            "revision_negative_delta_mean": mean(negative) if negative else 0.0,
            "revision_catastrophic_regress_rate": catastrophic / max(1, len(stage0_success)),
            "revision_break_compile_rate": broke_compile / max(1, len(stage0_success)),
            "repair_rescue_rate": rescue / max(1, len(stage0_fail)),
            "stage0_tokens_mean": mean(len(row["stage0"]["completion_ids"]) for row in records) if records else 0.0,
            "stage1_tokens_mean": mean(len(row["stage1"]["completion_ids"]) for row in records) if records else 0.0,
            "stage0_full_tokens_mean": mean(record_cost(row, 0) for row in records) if records else 0.0,
            "stage1_full_tokens_mean": mean(record_cost(row, 1) for row in records) if records else 0.0,
            "stage0_hit_max_rate": sum(bool(row["stage0"]["hit_max_tokens"]) for row in records) / n,
            "stage1_hit_max_rate": sum(bool(row["stage1"]["hit_max_tokens"]) for row in records) / n,
            "stage0_end_document_stop_rate": sum(bool(row["stage0"].get("stopped_on_end_document")) for row in records) / n,
            "stage1_end_document_stop_rate": sum(bool(row["stage1"].get("stopped_on_end_document")) for row in records) / n,
        }

    def _reduce_mean_metrics(self, metrics: dict[str, float]) -> dict[str, float]:
        output: dict[str, float] = {}
        for key, value in metrics.items():
            tensor = torch.tensor(float(value), device=self.accelerator.device)
            output[key] = float(self.accelerator.reduce(tensor, reduction="mean").item())
        return output

    def _append_step_log(self, row: dict[str, Any]) -> None:
        with open(self.step_log, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            handle.flush()

    def train(self) -> None:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        self.accelerator.print(
            f"[workflow] ranks={self.accelerator.num_processes} global_prompts={self.global_prompts_per_step} "
            f"G={self.group_size} planned_steps={self.total_steps} start_step={self.start_step} "
            f"distributed_queue={self.coordinator.enabled} claim_batch={self.coordinator.claim_batch_size}"
        )
        try:
            for step_num in range(self.start_step + 1, self.total_steps + 1):
                step_started = time.monotonic()
                rows = self._rows_for_step(step_num)
                self.generator.sync_policy(step_num - 1)
                records, rollout_metrics = self.coordinator.run_global_rollout(rows, step_num)
                self.coordinator.write_global_log(records, step_num)

                stage0_records = self.coordinator.partition_for_backward(records, stage=0)
                stage1_records = self.coordinator.partition_for_backward(records, stage=1)
                stage0_actions, stage0_advantages = materialize_stage_actions(stage0_records, 0, self.processor)
                stage1_actions, stage1_advantages = materialize_stage_actions(stage1_records, 1, self.processor)
                partition_stats = {}
                for stage, recs in ((0, stage0_records), (1, stage1_records)):
                    total_tokens = sum(
                        int(record[f"stage{stage}"].get("prompt_len", 0))
                        + len(record[f"stage{stage}"].get("completion_ids", []))
                        for record in recs
                    )
                    partition_stats[f"stage{stage}_backward_actions"] = float(len(recs))
                    partition_stats[f"stage{stage}_backward_full_tokens"] = float(total_tokens)

                self.coordinator.wait_update_ready(step_num)
                self.optimizer.zero_grad(set_to_none=True)
                stage0_backward = self._stage_backward(
                    stage0_actions,
                    stage0_advantages,
                    self.generator.stage0_max,
                    self.stage0_loss_weight,
                )
                stage1_backward = self._stage_backward(
                    stage1_actions,
                    stage1_advantages,
                    self.generator.stage1_max,
                    self.stage1_loss_weight,
                )
                grad_norm = self.accelerator.clip_grad_norm_(
                    [parameter for parameter in self.model.parameters() if parameter.requires_grad],
                    self.max_grad_norm,
                )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

                global_metrics = self._metrics_from_records(records)
                local = {
                    **global_metrics,
                    "loss": stage0_backward["loss"] + stage1_backward["loss"],
                    "stage0_loss": stage0_backward["loss"],
                    "stage1_loss": stage1_backward["loss"],
                    "stage0_clip_ratio": stage0_backward["clip_ratio"],
                    "stage1_clip_ratio": stage1_backward["clip_ratio"],
                    "grad_norm": float(grad_norm.detach().item() if torch.is_tensor(grad_norm) else grad_norm),
                    "lr": float(self.optimizer.param_groups[0]["lr"]),
                    "step_wall_s": time.monotonic() - step_started,
                    **partition_stats,
                    **rollout_metrics,
                }
                reduced = self._reduce_mean_metrics(local)
                self._append_step_log({"global_step": step_num, "time": time.time(), **local})
                if self.accelerator.is_main_process:
                    print("[step] " + json.dumps({"global_step": step_num, **reduced}, ensure_ascii=False))

                self.checkpoints.save(
                    step_num,
                    extra_state={
                        "run_id": self.coordinator.run_id,
                        "dataset_step": step_num,
                    },
                )
                self.coordinator.finish_step(step_num)

            self._save_final(self.total_steps)
        finally:
            self.coordinator.close()
            self.rollout_engine.close()

    def _save_final(self, global_step: int) -> None:
        self.accelerator.wait_for_everyone()
        final = self.output_dir / "final"
        if self.accelerator.is_main_process:
            final.mkdir(parents=True, exist_ok=True)
            unwrapped = self.accelerator.unwrap_model(self.model)
            unwrapped.save_pretrained(str(final), safe_serialization=True)
            try:
                self.processor.save_pretrained(str(final / "processor"))
            except Exception:
                pass
            (self.output_dir / "TRAIN_DONE").write_text(
                json.dumps({"global_step": int(global_step), "time": time.time(), "exit_code": 0}) + "\n",
                encoding="utf-8",
            )
        self.accelerator.wait_for_everyone()


def run_training(cfg: dict[str, Any]) -> None:
    WorkflowGRPOTrainer(cfg).train()
