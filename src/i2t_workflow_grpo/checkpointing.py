from __future__ import annotations

import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any


_CKPT_RE = re.compile(r"^checkpoint-(\d+)$")


def checkpoint_step(path: Path) -> int | None:
    match = _CKPT_RE.match(path.name)
    return int(match.group(1)) if match else None


class CheckpointManager:
    """Step-named rotating checkpoints plus permanent milestones.

    Default policy requested for long-running workflow RL:
      * save every optimizer step;
      * always retain the newest two step checkpoints;
      * retain every 25th checkpoint permanently;
      * no ``latest`` alias/symlink/directories.
    """

    def __init__(
        self,
        accelerator,
        model,
        processor,
        output_dir: str | Path,
        *,
        rolling_every_steps: int = 1,
        rolling_keep: int = 2,
        milestone_every_steps: int = 25,
    ) -> None:
        self.accelerator = accelerator
        self.model = model
        self.processor = processor
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rolling_every_steps = max(1, int(rolling_every_steps))
        self.rolling_keep = max(1, int(rolling_keep))
        self.milestone_every_steps = max(1, int(milestone_every_steps))

    def should_save(self, step: int) -> bool:
        return step % self.rolling_every_steps == 0 or step % self.milestone_every_steps == 0

    def save(self, step: int, extra_state: dict[str, Any] | None = None) -> Path | None:
        step = int(step)
        if not self.should_save(step):
            return None
        final = self.output_dir / f"checkpoint-{step}"
        incomplete = self.output_dir / f"checkpoint-{step}.incomplete"

        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            shutil.rmtree(incomplete, ignore_errors=True)
            incomplete.mkdir(parents=True, exist_ok=True)
        self.accelerator.wait_for_everyone()

        self.accelerator.save_state(str(incomplete / "accelerate_state"))
        if self.accelerator.is_main_process:
            unwrapped = self.accelerator.unwrap_model(self.model)
            unwrapped.save_pretrained(str(incomplete), safe_serialization=True)
            try:
                self.processor.save_pretrained(str(incomplete / "processor"))
            except Exception as exc:
                print(f"[warn] processor save failed: {exc}")
            state = {"global_step": step, "time": time.time(), **(extra_state or {})}
            (incomplete / "workflow_trainer_state.json").write_text(
                json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            shutil.rmtree(final, ignore_errors=True)
            os.replace(incomplete, final)
            self._prune()
        self.accelerator.wait_for_everyone()
        return final

    def _prune(self) -> None:
        checkpoints: list[tuple[int, Path]] = []
        for path in self.output_dir.iterdir():
            if not path.is_dir():
                continue
            step = checkpoint_step(path)
            if step is not None:
                checkpoints.append((step, path))
        checkpoints.sort()
        if not checkpoints:
            return

        latest_steps = {step for step, _ in checkpoints[-self.rolling_keep :]}
        milestone_steps = {
            step for step, _ in checkpoints if step % self.milestone_every_steps == 0
        }
        keep = latest_steps | milestone_steps
        for step, path in checkpoints:
            if step not in keep:
                shutil.rmtree(path, ignore_errors=True)

    def retained_steps(self) -> list[int]:
        steps: list[int] = []
        for path in self.output_dir.iterdir():
            if path.is_dir():
                step = checkpoint_step(path)
                if step is not None:
                    steps.append(step)
        return sorted(steps)
