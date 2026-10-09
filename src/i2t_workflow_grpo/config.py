from __future__ import annotations

import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        def repl(match: re.Match[str]) -> str:
            name = match.group(1)
            if name not in os.environ:
                raise KeyError(f"Environment variable {name!r} is required by the config")
            return os.environ[name]
        return os.path.expanduser(_ENV_RE.sub(repl, value))
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    return value


def _set_dot(cfg: dict[str, Any], key: str, value: Any) -> None:
    cur = cfg
    parts = key.split(".")
    for part in parts[:-1]:
        cur = cur.setdefault(part, {})
    cur[parts[-1]] = value


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg = deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must be KEY=VALUE, got {item}")
        key, raw = item.split("=", 1)
        _set_dot(cfg, key, yaml.safe_load(raw))
    return _expand_env(cfg)


def validate_algorithm_contract(cfg: dict[str, Any]) -> None:
    """Reject silent drift from the frozen v0.3.0 scientific contract."""
    image_size = int(cfg.get("model", {}).get("image_size", 448))
    if image_size != 448:
        raise ValueError(f"v0.3.0 freezes image_size=448, got {image_size}")
    workflow = cfg.get("workflow", {})
    if int(workflow.get("num_generations", 8)) < 2:
        raise ValueError("workflow.num_generations must be >=2 for group-relative optimization")
    if float(workflow.get("reward_scale", 10.0)) <= 0:
        raise ValueError("workflow.reward_scale must be positive")

    generation = cfg.get("generation", {})
    backend = str(generation.get("backend", "transformers"))
    if backend != "transformers":
        raise ValueError(f"Unsupported generation.backend={backend!r}")
    if int(generation.get("stage0_generation_microbatch", 4)) <= 0 or int(generation.get("stage1_generation_microbatch", 2)) <= 0:
        raise ValueError("generation microbatches must be positive")
    distributed = workflow.get("distributed_rollout", {})
    if int(distributed.get("claim_batch_size", 4)) <= 0:
        raise ValueError("workflow.distributed_rollout.claim_batch_size must be positive")
    loss = cfg.get("loss", {})
    if str(loss.get("type", "dr_grpo")) != "dr_grpo":
        raise ValueError("v0.3.0 implements loss.type=dr_grpo only")
    if bool(loss.get("scale_rewards", False)):
        raise ValueError("v0.3.0 freezes scale_rewards=false")
    if float(loss.get("beta", 0.0)) != 0.0:
        raise ValueError("v0.3.0 freezes beta=0")
    if float(loss.get("stage0_weight", 0.5)) < 0 or float(loss.get("stage1_weight", 0.5)) < 0:
        raise ValueError("stage loss weights must be non-negative")
    reward = cfg.get("reward", {})
    if str(reward.get("backend", "rsim_v2")) != "rsim_v2":
        raise ValueError("v0.3.0 implements reward.backend=rsim_v2 only")
