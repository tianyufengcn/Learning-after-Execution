from __future__ import annotations

import gc
import hashlib
import os
import sys
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageChops, ImageOps
from ot.lp import emd2
from safetensors.torch import load_file


def _trim_white(image: Image.Image) -> Image.Image:
    image = image.convert("RGB")
    bbox = ImageChops.difference(image, Image.new("RGB", image.size, "white")).getbbox()
    return image.crop(bbox) if bbox else image


def _square(image: Image.Image) -> Image.Image:
    image = _trim_white(image)
    side = max(image.size)
    return ImageOps.pad(image, (side, side), color="white", method=Image.Resampling.LANCZOS)


class RSimV2Scorer:
    """DeTikZify vision-encoder EMD similarity, following TikZilla's image reward.

    Improvements over the minimal implementation:

    - only the DeTikZify vision encoder remains resident;
    - GT features persist on disk and can be precomputed for the whole GRPO set;
    - generated-image features use an in-memory LRU cache;
    - EMD for a score batch can run on several CPU workers after the batched GPU
      vision forward finishes.
    """

    def __init__(
        self,
        model_path: str,
        cache_dir: str,
        detikzify_repo: str | None = None,
        batch_size: int = 16,
        device: str | None = None,
        mem_cache_size: int = 4096,
        emd_workers: int = 4,
    ) -> None:
        if detikzify_repo:
            sys.path.insert(0, str(Path(detikzify_repo).resolve()))
        try:
            from detikzify.model import load as load_detikzify
        except Exception as exc:
            raise ImportError(
                "Could not import detikzify. Install DeTikZify (`pip install -e /path/to/DeTikZify`) "
                "or set reward.detikzify_repo."
            ) from exc

        self.device = torch.device(device or (f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"))
        self.dtype = torch.bfloat16 if self.device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
        model_path_p = Path(model_path)
        vision_only_marker = model_path_p / "vision_only.json"
        if vision_only_marker.is_file():
            # Frozen vision-only RSim checkpoint exported from detikzify-v2-8b.
            # Loads only the Siglip vision encoder (~1.2 GiB) instead of the
            # full 8B model; RSim math/processor/EMD are unchanged.
            import json as _json
            from transformers import AutoImageProcessor, SiglipVisionModel
            from detikzify.model.configuration_detikzify import DetikzifyVisionConfig

            vc_dict = _json.loads((model_path_p / "vision_config.json").read_text(encoding="utf-8"))
            vcfg = DetikzifyVisionConfig.from_dict(dict(vc_dict))
            vision = SiglipVisionModel._from_config(vcfg)
            vision.load_state_dict(load_file(str(model_path_p / "vision_model.safetensors")))
            processor = AutoImageProcessor.from_pretrained(str(model_path_p))
            model = None
        else:
            model, processor = load_detikzify(model_path, torch_dtype=self.dtype)
            core = getattr(model, "model", model)
            vision = getattr(core, "vision_model", None)
            if vision is None:
                raise RuntimeError("DeTikZify model has no vision_model")
        self.vision = vision.to(self.device).eval().requires_grad_(False)
        self.processor = getattr(processor, "image_processor", processor)
        self.param_dtype = next(self.vision.parameters()).dtype
        self.batch_size = max(1, int(batch_size))
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.mem_cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self.mem_cache_size = max(1, int(mem_cache_size))
        self.emd_workers = max(1, int(emd_workers))

        # Release the 8B text decoder; only the trained vision encoder is needed.
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _feature_cache_path(self, path: str) -> Path:
        st = os.stat(path)
        key = f"{Path(path).resolve()}|{st.st_mtime_ns}|{st.st_size}"
        return self.cache_dir / (hashlib.sha1(key.encode()).hexdigest() + ".pt")

    def _remember(self, key: str, feat: torch.Tensor) -> None:
        self.mem_cache[key] = feat
        self.mem_cache.move_to_end(key)
        while len(self.mem_cache) > self.mem_cache_size:
            self.mem_cache.popitem(last=False)

    def _encode_pils(self, images: list[Image.Image]) -> list[torch.Tensor]:
        images = [_square(im.convert("RGB")) for im in images]
        encoding = self.processor(images=images, return_tensors="pt")
        pixel_values = encoding["pixel_values"]
        if pixel_values.ndim == 3:
            pixel_values = pixel_values.unsqueeze(0)
        pixel_values = pixel_values.to(self.device, dtype=self.param_dtype, non_blocking=True)
        with torch.inference_mode():
            output = self.vision(pixel_values=pixel_values)
            hidden = output.last_hidden_state.detach().to("cpu", dtype=torch.float32)
        return [hidden[i] for i in range(hidden.shape[0])]

    def _features(self, paths: list[str], persistent: bool) -> list[torch.Tensor | None]:
        result: list[torch.Tensor | None] = [None] * len(paths)
        missing_idx: list[int] = []
        missing_images: list[Image.Image] = []
        cache_paths: list[Path | None] = [None] * len(paths)

        for i, path in enumerate(paths):
            try:
                cp = self._feature_cache_path(path) if persistent else None
                cache_paths[i] = cp
                # Generated paths are content-addressed renderer files, so the path
                # itself is a stable cache key for the lifetime of a run.
                key = str(cp) if cp else str(Path(path).resolve())
                if key in self.mem_cache:
                    result[i] = self.mem_cache[key]
                    self.mem_cache.move_to_end(key)
                    continue
                if cp is not None and cp.exists():
                    feat = torch.load(cp, map_location="cpu")
                    result[i] = feat
                    self._remember(key, feat)
                    continue
                image = Image.open(path).convert("RGB")
                image.load()
                missing_idx.append(i)
                missing_images.append(image)
            except Exception:
                result[i] = None

        for start in range(0, len(missing_images), self.batch_size):
            imgs = missing_images[start : start + self.batch_size]
            idxs = missing_idx[start : start + self.batch_size]
            feats = self._encode_pils(imgs)
            for idx, feat in zip(idxs, feats, strict=True):
                result[idx] = feat
                cp = cache_paths[idx]
                key = str(cp) if cp else str(Path(paths[idx]).resolve())
                self._remember(key, feat)
                if cp is not None:
                    try:
                        tmp = cp.with_suffix(f".tmp.{os.getpid()}.pt")
                        torch.save(feat, tmp)
                        os.replace(tmp, cp)
                    except Exception:
                        try:
                            if 'tmp' in locals() and tmp.exists():
                                tmp.unlink()
                        except Exception:
                            pass
        return result

    @staticmethod
    def _emd_similarity(x: torch.Tensor, y: torch.Tensor) -> float:
        x = F.normalize(x.double(), p=2, dim=-1)
        y = F.normalize(y.double(), p=2, dim=-1)
        distances = (1.0 - x @ y.T).cpu().numpy()
        value = float(emd2(M=distances, a=list(), b=list()))
        score = 2.0 * np.tanh(-value) + 1.0
        return float(np.clip(score, 0.0, 1.0))

    def precompute_gt(self, gt_paths: list[str]) -> dict[str, int]:
        """Materialize persistent GT features before a long GRPO run."""
        unique = list(dict.fromkeys(str(Path(p).resolve()) for p in gt_paths if p and Path(p).exists()))
        feats = self._features(unique, persistent=True)
        ok = sum(x is not None for x in feats)
        return {"requested": len(gt_paths), "unique": len(unique), "encoded_or_cached": ok, "failed": len(unique) - ok}

    def gt_cache_hits(self, gt_paths: list[str]) -> list[bool]:
        """Logging-only: report whether each GT feature file already exists on disk."""
        out: list[bool] = []
        for path in gt_paths:
            try:
                out.append(bool(path) and self._feature_cache_path(path).exists())
            except Exception:
                out.append(False)
        return out

    def score_paths(self, generated_paths: list[str], gt_paths: list[str]) -> list[float]:
        if len(generated_paths) != len(gt_paths):
            raise ValueError("generated_paths and gt_paths must have equal length")
        if not generated_paths:
            return []

        # Generated vision forward and GT cache lookup/forward are performed in
        # sizeable batches by _features. GT paths normally hit disk cache after
        # scripts/precompute_gt_rsim.py has been run once.
        generated = self._features(generated_paths, persistent=False)
        gt = self._features(gt_paths, persistent=True)

        def one(pair: tuple[torch.Tensor | None, torch.Tensor | None]) -> float:
            x, y = pair
            if x is None or y is None:
                return 0.0
            try:
                return self._emd_similarity(x, y)
            except Exception:
                return 0.0

        pairs = list(zip(generated, gt, strict=True))
        if self.emd_workers <= 1 or len(pairs) <= 1:
            return [one(pair) for pair in pairs]
        with ThreadPoolExecutor(max_workers=min(self.emd_workers, len(pairs))) as pool:
            return list(pool.map(one, pairs))
