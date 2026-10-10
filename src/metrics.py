"""TikZilla-aligned image-image metric suite (protocol v2).

The implementations in this module intentionally mirror the public TikZilla
metric code for the three generally available metrics:

* CLIPImg: SigLIP image-image cosine using ``get_image_features``.
* DINO: ``dino-vits16`` CLS-token cosine similarity.
* DreamSim: ensemble DreamSim, white-square pair padding, similarity=1-distance.

RSim is optional because it depends on the DeTikZify package and a large
DeTikZify checkpoint.  When enabled it calls DeTikZify's own
``ImageSim.from_detikzify(..., mode='emd')`` instead of reimplementing EMD.
This preserves the exact score transform used by the DeTikZify source code.

All metrics compare rendered GT and generated images and are higher-is-better.
"""

from __future__ import annotations

import gc
import json
import math
import os
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageOps

METRIC_PROTOCOL_VERSION = "tikzilla-image-metrics-v2-20260817"
# Model locations are environment-driven; nothing is hard-coded. The evaluator
# passes explicit paths, these defaults are only a convenience fallback.
DEFAULT_DINO_MODEL = os.environ.get("I2T_DINO_MODEL", "")
DEFAULT_SIGLIP_MODEL = os.environ.get("I2T_SIGLIP_MODEL", "")
DEFAULT_DREAMSIM_CACHE = os.environ.get("I2T_DREAMSIM_CACHE", "")
DEFAULT_RSIM_MODEL = os.environ.get("I2T_RSIM_MODEL", "")

# Keep the old default names as aliases for backwards compatibility.
_NAME_ALIASES = {
    "clip": "clipimg",
    "clipimg": "clipimg",
    "clip_img": "clipimg",
    "siglip": "clipimg",
    "siglip_img": "clipimg",
    "dino": "dino",
    "dreamsim": "dreamsim",
    "dsim": "dreamsim",
    "rsim": "rsim",
}


def _canonical_name(name: str) -> str:
    key = str(name).strip().lower().replace("-", "_")
    if key not in _NAME_ALIASES:
        raise ValueError(f"Unknown image metric: {name!r}")
    return _NAME_ALIASES[key]


def parse_metric_names(value: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        raw = [x for x in value.replace(";", ",").split(",") if x.strip()]
    else:
        raw = list(value)
    out: list[str] = []
    for name in raw:
        c = _canonical_name(str(name))
        if c not in out:
            out.append(c)
    return tuple(out)


def _check_local_model(path: str, label: str) -> None:
    p = Path(path).expanduser()
    if p.is_absolute() and not p.exists():
        raise FileNotFoundError(f"{label} model path does not exist: {p}")


def _resolve_torch_dtype(torch: Any, device: str, requested: str | None = "auto") -> Any:
    req = str(requested or "auto").lower()
    if req in {"float32", "fp32"}:
        return torch.float32
    if req in {"float16", "fp16", "half"}:
        return torch.float16
    if req in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if req != "auto":
        raise ValueError(f"Unsupported dtype: {requested}")
    if str(device).startswith("cuda") and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def _load_rgb(path: str | Path) -> Image.Image:
    with Image.open(path) as src:
        return src.convert("RGB")


@dataclass(frozen=True)
class MetricBackendInfo:
    metric: str
    higher_is_better: bool
    backend: str
    model_path: str | None
    implementation: str
    protocol_version: str = METRIC_PROTOCOL_VERSION


class MetricSuite:
    """Reusable image-image metrics with lazy model loading.

    By default this loads the three light/medium metrics suitable for the main
    Qwen/TikZilla environment.  RSim is deliberately opt-in because the
    DeTikZify checkpoint is large and its dependency stack is often isolated.
    """

    def __init__(
        self,
        *,
        dino_model_path: str = DEFAULT_DINO_MODEL,
        siglip_model_path: str = DEFAULT_SIGLIP_MODEL,
        dreamsim_cache_dir: str = DEFAULT_DREAMSIM_CACHE,
        rsim_model_path: str | None = None,
        rsim_mode: str = "emd",
        device: str = "cuda:0",
        enabled: tuple[str, ...] | list[str] | str = ("dino", "dreamsim", "clipimg"),
        dreamsim_dtype: str = "auto",
        siglip_dtype: str = "auto",
        rsim_dtype: str = "auto",
    ) -> None:
        self.enabled = list(parse_metric_names(enabled))
        self.device = str(device)
        self.dino_model_path = str(dino_model_path)
        self.siglip_model_path = str(siglip_model_path)
        self.dreamsim_cache_dir = str(dreamsim_cache_dir)
        self.rsim_model_path = str(rsim_model_path) if rsim_model_path else None
        self.rsim_mode = str(rsim_mode)
        self.dreamsim_dtype = str(dreamsim_dtype)
        self.siglip_dtype = str(siglip_dtype)
        self.rsim_dtype = str(rsim_dtype)

        self._torch = None
        self._F = None
        self._dino = None
        self._dino_processor = None
        self._siglip = None
        self._siglip_processor = None
        self._dreamsim = None
        self._dreamsim_preprocess = None
        self._dreamsim_torch_dtype = None
        self._rsim = None
        self._rsim_backend_label = None

    # ------------------------------------------------------------------
    # Backend metadata
    # ------------------------------------------------------------------
    def backend_info(self) -> dict[str, dict[str, Any]]:
        infos: dict[str, MetricBackendInfo] = {}
        for name in self.enabled:
            if name == "dino":
                infos[name] = MetricBackendInfo(
                    metric=name,
                    higher_is_better=True,
                    backend="ViTImageProcessor+ViTModel / CLS cosine",
                    model_path=self.dino_model_path,
                    implementation="TikZilla-aligned DinoScore(use_cls_token=True, normalize=True)",
                )
            elif name == "clipimg":
                infos[name] = MetricBackendInfo(
                    metric=name,
                    higher_is_better=True,
                    backend="SigLIP image-image cosine",
                    model_path=self.siglip_model_path,
                    implementation="TikZilla-aligned ClipScoreImg(get_image_features + L2 cosine)",
                )
            elif name == "dreamsim":
                infos[name] = MetricBackendInfo(
                    metric=name,
                    higher_is_better=True,
                    backend="DreamSim ensemble / 1-distance",
                    model_path=self.dreamsim_cache_dir,
                    implementation="TikZilla-aligned DreamSim(normalize_embeds=True, common white-square padding)",
                )
            elif name == "rsim":
                infos[name] = MetricBackendInfo(
                    metric=name,
                    higher_is_better=True,
                    backend=f"DeTikZify ImageSim mode={self.rsim_mode}",
                    model_path=self.rsim_model_path,
                    implementation="detikzify.evaluate.imagesim.ImageSim.from_detikzify (source score transform preserved)",
                )
        return {k: asdict(v) for k, v in infos.items()}

    def protocol_record(self) -> dict[str, Any]:
        return {
            "metric_protocol_version": METRIC_PROTOCOL_VERSION,
            "enabled": list(self.enabled),
            "device": self.device,
            "backends": self.backend_info(),
        }

    # ------------------------------------------------------------------
    # DINO -- exact public TikZilla convention: CLS token, normalize, cosine
    # ------------------------------------------------------------------
    def _load_dino(self) -> None:
        if self._dino is not None:
            return
        _check_local_model(self.dino_model_path, "DINO")
        try:
            import torch
            import torch.nn.functional as F
            from transformers import ViTImageProcessor, ViTModel
        except ImportError as exc:
            raise ImportError("DINO scoring requires torch and transformers") from exc
        self._torch, self._F = torch, F
        self._dino_processor = ViTImageProcessor.from_pretrained(self.dino_model_path)
        self._dino = ViTModel.from_pretrained(self.dino_model_path).to(self.device).eval()

    @lru_cache(maxsize=4096)
    def _dino_embedding(self, image_path: str) -> np.ndarray:
        self._load_dino()
        image = _load_rgb(image_path)
        inputs = self._dino_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self._torch.inference_mode():
            hidden = self._dino(**inputs).last_hidden_state
            feat = hidden[:, 0]  # TikZilla DinoScore default: CLS token.
            feat = self._F.normalize(feat, dim=-1)
        return feat.squeeze(0).float().cpu().numpy()

    # ------------------------------------------------------------------
    # CLIPImg -- TikZilla's SigLIP image-image score
    # ------------------------------------------------------------------
    def _load_siglip(self) -> None:
        if self._siglip is not None:
            return
        _check_local_model(self.siglip_model_path, "SigLIP")
        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except ImportError as exc:
            raise ImportError("CLIPImg scoring requires torch and transformers") from exc
        self._torch = torch
        dtype = _resolve_torch_dtype(torch, self.device, self.siglip_dtype)
        self._siglip = AutoModel.from_pretrained(self.siglip_model_path, torch_dtype=dtype).to(self.device).eval()
        self._siglip_processor = AutoProcessor.from_pretrained(self.siglip_model_path)

    @lru_cache(maxsize=4096)
    def _siglip_embedding(self, image_path: str) -> np.ndarray:
        self._load_siglip()
        image = _load_rgb(image_path)
        # Match TikZilla ClipScoreImg: no custom crop/pad here; let AutoProcessor
        # apply the model-native 384px preprocessing.
        inputs = self._siglip_processor(images=[image], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self._torch.inference_mode():
            emb = self._siglip.get_image_features(pixel_values=inputs["pixel_values"])
            # transformers >= 5.x returns BaseModelOutputWithPooling here; extract
            # the pooled image embedding (SigLIP [CLS] projection) explicitly.
            if hasattr(emb, "pooler_output"):
                emb = emb.pooler_output
            elif hasattr(emb, "image_embeds"):
                emb = emb.image_embeds
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.squeeze(0).float().cpu().numpy()

    # ------------------------------------------------------------------
    # DreamSim -- exact public TikZilla convention
    # ------------------------------------------------------------------
    def _load_dreamsim(self) -> None:
        if self._dreamsim is not None:
            return
        try:
            import torch
            from dreamsim import dreamsim
        except ImportError as exc:
            raise ImportError("DreamSim scoring requires the dreamsim package") from exc
        self._torch = torch
        dtype = _resolve_torch_dtype(torch, self.device, self.dreamsim_dtype)
        cache_dir = Path(self.dreamsim_cache_dir).expanduser()
        cache_dir.mkdir(parents=True, exist_ok=True)
        model, processor = dreamsim(
            dreamsim_type="ensemble",
            pretrained=True,
            normalize_embeds=True,
            device=self.device,
            cache_dir=str(cache_dir),
        )
        # TikZilla's public implementation explicitly casts every extractor and
        # projection before casting the wrapper model itself.
        if hasattr(model, "extractor_list"):
            for extractor in model.extractor_list:
                if hasattr(extractor, "model"):
                    extractor.model = extractor.model.to(dtype)
                if hasattr(extractor, "proj"):
                    extractor.proj = extractor.proj.to(dtype)
        self._dreamsim = model.to(self.device, dtype).eval()
        self._dreamsim_preprocess = processor
        self._dreamsim_torch_dtype = dtype

    @staticmethod
    def _dreamsim_pair(a: str | Path, b: str | Path) -> tuple[Image.Image, Image.Image]:
        ia, ib = _load_rgb(a), _load_rgb(b)
        max_dim = max(ia.width, ia.height, ib.width, ib.height)
        bg = (255, 255, 255)
        return (
            ImageOps.pad(ia, (max_dim, max_dim), color=bg, centering=(0.5, 0.5)),
            ImageOps.pad(ib, (max_dim, max_dim), color=bg, centering=(0.5, 0.5)),
        )

    def _dreamsim_similarity(self, a: str, b: str) -> float:
        self._load_dreamsim()
        img_a, img_b = self._dreamsim_pair(a, b)
        ta = self._dreamsim_preprocess(img_a).to(self.device, self._dreamsim_torch_dtype)
        tb = self._dreamsim_preprocess(img_b).to(self.device, self._dreamsim_torch_dtype)
        with self._torch.inference_mode():
            distance = self._dreamsim(ta, tb).item()
        return float(1.0 - distance)

    # ------------------------------------------------------------------
    # RSim -- exact DeTikZify ImageSim EMD backend
    # ------------------------------------------------------------------
    def _load_rsim(self) -> None:
        if self._rsim is not None:
            return
        if not self.rsim_model_path:
            raise ValueError(
                "RSim is enabled but rsim_model_path is not set. Pass the exact "
                "DeTikZify checkpoint used for the desired RSim protocol."
            )
        _check_local_model(self.rsim_model_path, "RSim/DeTikZify")
        try:
            import torch
            from detikzify.model import load as load_detikzify
            from detikzify.evaluate.imagesim import ImageSim
        except ImportError as exc:
            raise ImportError(
                "RSim requires the DeTikZify package (detikzify) and its dependencies, "
                "including POT/ot. Run RSim in a dedicated DeTikZify environment if needed."
            ) from exc
        self._torch = torch
        dtype = _resolve_torch_dtype(torch, self.device, self.rsim_dtype)
        model, processor = load_detikzify(
            model_name_or_path=self.rsim_model_path,
            device_map=self.device,
            torch_dtype=dtype,
        )
        self._rsim = ImageSim.from_detikzify(model=model, processor=processor, mode=self.rsim_mode)
        self._rsim_backend_label = f"detikzify:{self.rsim_mode}:{self.rsim_model_path}"
        # ImageSim keeps the vision submodule and processor; the language model
        # parent is no longer needed for scoring.  Releasing the parent reduces
        # memory without changing the vision encoder used by ImageSim.
        del model
        gc.collect()
        if self.device.startswith("cuda") and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    def _rsim_similarity(self, a: str, b: str) -> float:
        self._load_rsim()
        score = self._rsim.get_similarity(img1=str(a), img2=str(b))
        return float(score)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def prepare(self) -> None:
        """Eagerly initialize all enabled metric backends."""
        for name in self.enabled:
            if name == "dino":
                self._load_dino()
            elif name == "clipimg":
                self._load_siglip()
            elif name == "dreamsim":
                self._load_dreamsim()
            elif name == "rsim":
                self._load_rsim()
            else:  # pragma: no cover - parse_metric_names prevents this.
                raise ValueError(f"Unknown metric: {name}")

    def compare(self, target: str, candidate: str) -> dict[str, float]:
        target = str(Path(target).resolve())
        candidate = str(Path(candidate).resolve())
        if not Path(target).exists():
            raise FileNotFoundError(target)
        if not Path(candidate).exists():
            raise FileNotFoundError(candidate)
        scores: dict[str, float] = {}
        for name in self.enabled:
            if name == "dino":
                a = self._dino_embedding(target)
                b = self._dino_embedding(candidate)
                # Embeddings are already normalized, so dot == cosine.
                scores[name] = float(np.dot(a, b))
            elif name == "clipimg":
                a = self._siglip_embedding(target)
                b = self._siglip_embedding(candidate)
                scores[name] = float(np.dot(a, b))
            elif name == "dreamsim":
                scores[name] = self._dreamsim_similarity(target, candidate)
            elif name == "rsim":
                scores[name] = self._rsim_similarity(target, candidate)
        # Guard against silently propagating NaNs into JSONL/report statistics.
        for key, val in list(scores.items()):
            if not math.isfinite(float(val)):
                raise ValueError(f"Metric {key} produced non-finite score: {val}")
        return scores

    def compare_batch(self, pairs: Iterable[tuple[str, str]]) -> list[dict[str, float]]:
        return [self.compare(gt, pred) for gt, pred in pairs]

    def dump_protocol(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.protocol_record(), ensure_ascii=False, indent=2), encoding="utf-8")
        return out
