from __future__ import annotations

import hashlib
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any

# Layout: <repository>/scripts/evaluation/wf_eval_common.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]   # workflow_grpo project root
RELEASE_ROOT = Path(__file__).resolve().parents[3]   # release root (holds data/)

EVAL_ROOT = Path(
    os.environ.get("WF_EVAL_ROOT")
    or (Path(os.environ.get("OUTPUT_ROOT", ".")) / "eval_workflow_grpo")
)
# Split manifests shipped with this release (dataset-relative paths only).
DEV_MANIFEST = RELEASE_ROOT / "data" / "dev150_manifest.jsonl"
FINAL_MANIFEST = RELEASE_ROOT / "data" / "final945_manifest.jsonl"
TEST_MANIFEST = Path(
    os.environ.get("WF_TEST_MANIFEST")
    or os.environ.get("FINAL_MANIFEST")
    or os.environ.get("DEV_MANIFEST")
    or FINAL_MANIFEST
)
# Name of the manifest copy the controller writes into <eval_root>/00_manifest.
# Runs created before this naming was generalised used the legacy split-specific
# name; it is still accepted so archived evaluations remain readable.
MANIFEST_COPY_NAME = "test_manifest.jsonl"
LEGACY_MANIFEST_COPY_NAMES = ("manifest_test_150.jsonl",)
# Optional root used to resolve relative image/code paths recorded in the split
# manifests shipped under data/ (they carry dataset-relative paths only).
GT_ROOT = os.environ.get("WF_GT_IMAGE_ROOT", "")
REPLICAS_PER_GPU = int(os.environ.get("WF_REPLICAS_PER_GPU", "3"))

BASE_MODEL = os.environ.get("BASE_MODEL") or os.environ.get("I2T_MODEL") or ""
INITIALIZER_ADAPTER = (
    os.environ.get("WF_ADAPTER")
    or os.environ.get("INITIALIZER_ADAPTER")
    or os.environ.get("INIT_ADAPTER")
    or ""
)
RSIM_MODEL = os.environ.get("RSIM_MODEL_PATH") or os.environ.get("WF_RSIM_MODEL") or ""
TEXLIVE_BIN = os.environ.get("TEXLIVE_BIN", "")
PYTHON = os.environ.get("WF_PYTHON") or sys.executable
EVAL_SEED = 20260825

# Evaluation roots contain bookkeeping directories next to one directory per
# policy. These are never policies.
_NON_POLICY_DIRS = {
    "00_manifest",
    "cache",
    "controller",
    "logs",
    "pids",
    "run_contract",
    "summary",
    "offline_acceptance_rsim",
}

# Formal Workflow-GRPO v0.3.0 generation contract (configs/workflow_grpo_1k.yaml).
GENERATION = {
    "backend": "transformers",
    "temperature": 1.0,
    "top_p": 0.9,
    "repetition_penalty": 1.0,
    "stage0_max_completion_length": 8192,
    "stage1_max_completion_length": 8192,
    "stop_on_end_document": True,
}

RENDERER = {
    "texlive_bin": TEXLIVE_BIN,
    "compilers": ["pdflatex", "lualatex", "xelatex"],
    "compile_timeout_s": 30,
    "image_size": 448,
    "dpi": 150,
    "cache_failures": True,
    "cache_transient_failures": False,
    "keep_failed_logs": True,
}

RSIM_CONFIG = {
    "backend": "rsim_v2",
    "model_path": RSIM_MODEL,
    "require_vision_only": True,
    "score_batch_size": 16,
    "score_mem_cache_size": 4096,
    "emd_workers": 2,
}


def policy_adapter(policy: str) -> str:
    """Adapter for a policy label.

    Only the ``initializer`` policy has an implicit adapter (the warm-start
    adapter). Any other policy is a trained checkpoint and must be supplied
    explicitly through ``--adapter`` / ``WF_ADAPTER``; guessing a path here
    would silently evaluate the wrong model.
    """
    if policy == "initializer":
        return INITIALIZER_ADAPTER
    raise ValueError(
        f"No adapter configured for policy {policy!r}: pass --adapter (or set "
        "WF_ADAPTER) to the trained checkpoint you want to evaluate."
    )


def discover_policies(eval_root: str | Path) -> list[str]:
    """Policy labels present in an evaluation root (directories with results/)."""
    root = Path(eval_root)
    found = []
    if root.is_dir():
        for child in sorted(root.iterdir()):
            if not child.is_dir() or child.name in _NON_POLICY_DIRS:
                continue
            if (child / "results").is_dir():
                found.append(child.name)
    return found


def load_manifest(path: str | Path | None = None) -> list[dict[str, Any]]:
    """Load a JSONL split manifest.

    ``path`` (the controller's ``--test-manifest``) always wins. When it is not
    given, the environment-resolved ``TEST_MANIFEST`` is used.
    """
    manifest = Path(path) if path not in (None, "") else TEST_MANIFEST
    if not str(manifest).strip() or not manifest.is_file():
        raise FileNotFoundError(
            f"Evaluation manifest not found: {manifest!s}. Pass --test-manifest, or set "
            "WF_TEST_MANIFEST (or FINAL_MANIFEST / DEV_MANIFEST) to a JSONL split manifest, "
            "e.g. data/final945_manifest.jsonl."
        )
    rows = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    def _resolve(value: str) -> str:
        if not value:
            return value
        path = Path(value)
        if path.is_absolute() or not GT_ROOT:
            return str(path)
        return str(Path(GT_ROOT) / path)

    out = []
    for row in rows:
        sid = str(row.get("file_id") or row.get("sample_id"))
        gt = _resolve(str(row.get("asset_image_path") or row.get("resolved_image_path") or row.get("image_path") or ""))
        gt_tikz = _resolve(str(row.get("asset_code_path") or row.get("gt_tikz_path") or ""))
        out.append(
            {
                "sample_id": sid,
                "gt_image_path": gt,
                "gt_tikz_path": gt_tikz,
                "row_order": len(out),
            }
        )
    return out


def manifest_copy_path(eval_root: str | Path) -> Path | None:
    """Locate the manifest copy the controller wrote for an evaluation root."""
    base = Path(eval_root) / "00_manifest"
    for name in (MANIFEST_COPY_NAME, *LEGACY_MANIFEST_COPY_NAMES):
        candidate = base / name
        if candidate.is_file():
            return candidate
    return None


def policy_root(policy: str) -> Path:
    root = EVAL_ROOT / policy
    for sub in ("results", "tex/stage0", "tex/stage1", "render/stage0", "render/stage1", "logs", "pids"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    return root


def stable_seed(base_seed: int, sample_id: str) -> int:
    """Deterministic per-sample seed (same scheme as the Direct-4K evaluator)."""
    return int(base_seed) + int(hashlib.sha256(str(sample_id).encode("utf-8")).hexdigest()[:8], 16)


def safe_sample_id(value: Any) -> str:
    safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(value))
    return safe[:120] or "sample"


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, default=str, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        handle.flush()
