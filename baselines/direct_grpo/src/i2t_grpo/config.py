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
    parts = key.split(".")
    cur = cfg
    for part in parts[:-1]:
        cur = cur.setdefault(part, {})
    cur[parts[-1]] = value


def _parse_override(value: str) -> Any:
    # Let YAML parse booleans/numbers/lists, while ordinary strings remain strings.
    return yaml.safe_load(value)


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg = deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must be KEY=VALUE, got: {item}")
        key, value = item.split("=", 1)
        _set_dot(cfg, key, _parse_override(value))
    return _expand_env(cfg)
