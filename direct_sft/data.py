from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import base64
import io
import json
import os

from PIL import Image

try:
    from datasets import Dataset as HFDataset
    from datasets import load_dataset
except Exception:  # pragma: no cover
    HFDataset = None
    load_dataset = None

from .config import DataConfig


_EXCLUDED_AUTODISCOVERY_PARTS = {"rollouts", "direct_sft", "revision", "reports", "logs"}


@dataclass
class NormalizedExample:
    sample_id: str
    image: Any
    tikz: str
    source_row: dict[str, Any]
    base_dir: Path


def _candidate_score(path: Path, size: int) -> tuple[int, int, str]:
    name = path.name.lower()
    score = 0
    if "train" in name: score += 20
    if "manifest" in name: score += 10
    if str(size) in name: score += 8
    if size == 1000 and "1k" in name: score += 12
    if size == 4000 and "4k" in name: score += 12
    if size == 16000 and "16k" in name: score += 12
    if path.suffix == ".jsonl": score += 5
    # Shorter paths are preferred after semantic score.
    return (-score, len(path.parts), str(path))


def discover_data_source(work_root: str | Path, cohort_size: int) -> Path:
    root = Path(work_root)
    if not root.exists():
        raise FileNotFoundError(f"work_root does not exist: {root}")

    exact = [
        root / "manifests" / f"train_{cohort_size}.jsonl",
        root / "manifests" / f"train_{cohort_size}.parquet",
        root / "manifests" / ({1000:"train_1k.jsonl",4000:"train_4k.jsonl",16000:"train_16k.jsonl"}.get(cohort_size, "")),
        root / ({1000:"train_1k.jsonl",4000:"train_4k.jsonl",16000:"train_16k.jsonl"}.get(cohort_size, "")),
    ]
    for p in exact:
        if p.name and p.exists():
            return p

    files: list[Path] = []
    for ext in ("*.jsonl", "*.parquet"):
        for p in root.rglob(ext):
            if any(part.lower() in _EXCLUDED_AUTODISCOVERY_PARTS for part in p.parts):
                continue
            lower = p.name.lower()
            size_tokens = [str(cohort_size), {1000:"1k",4000:"4k",16000:"16k"}.get(cohort_size, "")]
            if any(tok and tok in lower for tok in size_tokens):
                files.append(p)
    if files:
        return sorted(files, key=lambda p: _candidate_score(p, cohort_size))[0]

    # A frozen 16K master can be safely prefix-selected for 1K/4K nested experiments.
    if cohort_size < 16000:
        for pattern in ("*16k*.jsonl", "*16000*.jsonl", "*16k*.parquet", "*16000*.parquet"):
            masters = [p for p in root.rglob(pattern) if not any(part.lower() in _EXCLUDED_AUTODISCOVERY_PARTS for part in p.parts)]
            if masters:
                return sorted(masters, key=lambda p: _candidate_score(p, 16000))[0]
    raise FileNotFoundError(
        f"Could not auto-discover training data for cohort={cohort_size} under {root}. "
        "Set I2T_DATA_SOURCE=/absolute/path/to/manifest.jsonl or parquet."
    )


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows=[]
    with path.open("r", encoding="utf-8") as f:
        for lineno,line in enumerate(f,1):
            line=line.strip()
            if not line: continue
            obj=json.loads(line)
            if not isinstance(obj, dict):
                raise TypeError(f"{path}:{lineno} is not an object")
            rows.append(obj)
    return rows


def load_rows(source: Path):
    if source.suffix == ".jsonl":
        return _load_jsonl(source)
    if source.suffix == ".parquet":
        if load_dataset is None:
            raise RuntimeError("datasets is required for parquet input")
        siblings = sorted(source.parent.glob("*.parquet"))
        # Only treat siblings as one dataset when the chosen file looks like a shard.
        use = siblings if len(siblings) > 1 and any(token in source.name.lower() for token in ("shard", "part", "00000", "train-")) else [source]
        return load_dataset("parquet", data_files=[str(p) for p in use], split="train")
    raise ValueError(f"Unsupported source: {source}")


def _first_value(row: dict[str, Any], keys: Iterable[str]):
    for key in keys:
        if key in row and row[key] is not None and row[key] != "":
            return row[key], key
    return None, None


def _resolve_path(value: str, base_dir: Path) -> Path:
    p=Path(value)
    if p.is_absolute(): return p
    candidates=[base_dir/p, base_dir.parent/p]
    for c in candidates:
        if c.exists(): return c.resolve()
    return (base_dir/p).resolve()


def resolve_tikz(row: dict[str, Any], cfg: DataConfig, base_dir: Path) -> str:
    value,key=_first_value(row,cfg.tikz_value_keys)
    if value is not None:
        if isinstance(value, bytes): value=value.decode("utf-8")
        if not isinstance(value,str): value=str(value)
        if value.strip(): return value
    value,key=_first_value(row,cfg.tikz_path_keys)
    if value is not None:
        p=_resolve_path(str(value),base_dir)
        if not p.exists(): raise FileNotFoundError(f"TikZ path not found: {p}")
        text=p.read_text(encoding="utf-8")
        if not text.strip(): raise ValueError(f"Empty TikZ file: {p}")
        return text
    raise KeyError(f"Could not find TikZ in row; keys={list(row.keys())[:40]}")


def resolve_image(row: dict[str, Any], cfg: DataConfig, base_dir: Path):
    value,key=_first_value(row,cfg.image_path_keys)
    if value is not None:
        p=_resolve_path(str(value),base_dir)
        if not p.exists(): raise FileNotFoundError(f"Image path not found: {p}")
        return str(p)
    value,key=_first_value(row,cfg.image_value_keys)
    if value is None:
        raise KeyError(f"Could not find image in row; keys={list(row.keys())[:40]}")
    if isinstance(value, Image.Image): return value
    if isinstance(value, str):
        p=_resolve_path(value,base_dir)
        if p.exists(): return str(p)
        # Accept data URI / raw base64 only as a last resort.
        if value.startswith("data:image"):
            return base64.b64decode(value.split(",",1)[1])
    if isinstance(value, (bytes, bytearray, memoryview)): return bytes(value)
    if isinstance(value, dict):
        if value.get("path"):
            p=_resolve_path(str(value["path"]),base_dir)
            if p.exists(): return str(p)
        if value.get("bytes") is not None: return bytes(value["bytes"])
    raise TypeError(f"Unsupported image value type for key {key}: {type(value)}")


def open_image(value) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, str):
        with Image.open(value) as im: return im.convert("RGB")
    if isinstance(value, (bytes, bytearray, memoryview)):
        with Image.open(io.BytesIO(bytes(value))) as im: return im.convert("RGB")
    raise TypeError(type(value))


class I2TTikZDataset:
    """A light wrapper around JSONL or HF parquet rows. Prefix selection preserves 1K⊂4K⊂16K."""
    def __init__(self, source: str | Path, cfg: DataConfig):
        self.source=Path(source)
        self.cfg=cfg
        self.base_dir=self.source.parent
        self.rows=load_rows(self.source)
        total=len(self.rows)
        n=min(cfg.cohort_size,total)
        if cfg.limit is not None: n=min(n,cfg.limit)
        if n < min(cfg.cohort_size, cfg.limit or cfg.cohort_size):
            raise ValueError(f"Source has only {total} rows, need {cfg.cohort_size}")
        self.n=n

    def __len__(self): return self.n

    def __getitem__(self, idx: int) -> NormalizedExample:
        row=dict(self.rows[idx])
        sid=None
        for key in self.cfg.id_keys:
            if row.get(key) not in (None,""):
                sid=str(row[key]); break
        if sid is None: sid=f"row-{idx:06d}"
        return NormalizedExample(
            sample_id=sid,
            image=resolve_image(row,self.cfg,self.base_dir),
            tikz=resolve_tikz(row,self.cfg,self.base_dir),
            source_row=row,
            base_dir=self.base_dir,
        )
