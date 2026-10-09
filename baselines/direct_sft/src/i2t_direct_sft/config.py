from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import os
import yaml

DEFAULT_PROMPT = r"""You are an expert Image-to-TikZ model and LaTeX/TikZ developer.
Study the provided target image carefully and reconstruct it as a high-quality
TikZ figure. Focus on the visible objects, geometry, topology, text and
mathematical labels, colors, line and arrow styles, relative sizes, alignment,
and overall layout. Choose the native TikZ, PGFPlots, circuitikz, tikz-cd, or
other LaTeX drawing constructs that best reproduce the image. Do not invent or
omit important visible content when it can be represented faithfully.

OUTPUT REQUIREMENTS

- Return only raw LaTeX source code. Do not output Markdown fences,
  explanations, review prose, or any text outside the document.
- Output exactly one complete standalone document. Start with
  \documentclass[tikz]{standalone}; only when a small margin is needed to
  prevent visible clipping, use border=1pt, border=2pt, or border=3pt in the
  same document class.
- Include every package and TikZ library required by the generated code in the
  preamble. The document must be self-contained, use native TikZ/PGF drawing,
  and compile successfully with pdflatex with shell escape disabled. Do not
  load external files or assets.
- Reconstruct the target image as faithfully as possible while preserving
  compilability and document completeness.
- Include exactly one \begin{document} and one matching \end{document}. The
  final non-whitespace content must be \end{document}; stop immediately after
  it."""


@dataclass
class PathConfig:
    # Local locations are supplied through I2T_MODEL / I2T_WORK_ROOT /
    # I2T_OUTPUT_ROOT (see load_config) or through the config file.
    model: str = ""
    work_root: str = ""
    output_root: str = ""


@dataclass
class DataConfig:
    source: str = "auto"
    cohort_size: int = 1000
    prompt: str = DEFAULT_PROMPT
    # TikZilla uses max_seq_length=2048 for text-only SFT. Our target TikZ is
    # likewise capped at 2048 Qwen tokens, but the multimodal sequence also
    # contains a fixed 448x448 image and the user prompt, so the total context
    # guard is 4096. Nothing is silently truncated.
    max_tikz_tokens: int = 2048
    model_max_length: int = 4096
    limit: int | None = None
    image_size: int = 448
    strict_image_size: bool = True
    # The formal I2T pipeline rasterizes/pads to exactly 448x448. For Qwen-VL,
    # 448 is divisible by 28, so setting min=max pins the processor budget to
    # the same physical resolution rather than allowing dynamic down/up-scaling.
    min_pixels: int | None = 448 * 448
    max_pixels: int | None = 448 * 448
    require_end_document: bool = True
    image_path_keys: list[str] = field(default_factory=lambda: [
        "gt_image_path", "image_path", "image_file", "render_path", "png_path"
    ])
    image_value_keys: list[str] = field(default_factory=lambda: ["image", "gt_image"])
    tikz_path_keys: list[str] = field(default_factory=lambda: [
        "reference_tikz_path", "reference_tex_path", "gt_tikz_path", "tikz_path", "code_path"
    ])
    tikz_value_keys: list[str] = field(default_factory=lambda: [
        "gt_tikz", "tikz", "tikz_code", "code", "reference_tikz", "reference"
    ])
    id_keys: list[str] = field(default_factory=lambda: ["sample_id", "id", "file_id"])


@dataclass
class ModelConfig:
    attention_backend: str = "auto"          # auto | flash_attention_2 | sdpa
    # Match TikZilla SFT's published source defaults where they transfer
    # cleanly to Qwen3-VL: r=256, alpha=512, dropout=0.05 and the same seven
    # attention/MLP projection families. Vision/merger remain frozen here.
    lora_r: int = 256
    lora_alpha: int = 512
    lora_dropout: float = 0.05
    lora_targets: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    ])
    exclude_module_substrings: list[str] = field(default_factory=lambda: [
        "visual", "vision", "merger", "deepstack"
    ])
    gradient_checkpointing: bool = True


@dataclass
class TrainConfig:
    seed: int = 20260818
    # TikZilla SFT source defaults: 5 epochs, 1e-4 LR, 0.03 warmup,
    # cosine schedule, max_grad_norm=0.3. We retain fused AdamW for A100 speed.
    epochs: float = 5.0
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    max_grad_norm: float = 0.3
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    micro_batch_size: int = 4
    global_batch_size: int = 32
    logging_steps: int = 5
    save_steps: int = 50
    save_total_limit: int = 3
    dataloader_num_workers: int = 6
    dataloader_prefetch_factor: int = 4
    backend: str = "ddp"                     # ddp | zero2
    torch_compile: bool = False
    max_steps: int = -1
    report_to: str = "none"


@dataclass
class ExperimentConfig:
    name: str = "direct_1k"
    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @property
    def output_dir(self) -> Path:
        return Path(self.paths.output_root) / self.name


def _merge_dataclass(obj: Any, values: dict[str, Any]) -> None:
    for key, value in values.items():
        if not hasattr(obj, key):
            raise KeyError(f"Unknown config key {type(obj).__name__}.{key}")
        setattr(obj, key, value)


def load_config(path: str | os.PathLike[str]) -> ExperimentConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    cfg = ExperimentConfig(name=raw.get("name", Path(path).stem))
    for section in ("paths", "data", "model", "train"):
        if section in raw:
            _merge_dataclass(getattr(cfg, section), raw[section] or {})
    if "name" in raw:
        cfg.name = raw["name"]

    # Environment overrides make the downloaded repo usable without editing files.
    cfg.paths.model = os.getenv("I2T_MODEL", cfg.paths.model)
    cfg.paths.work_root = os.getenv("I2T_WORK_ROOT", cfg.paths.work_root)
    cfg.paths.output_root = os.getenv("I2T_OUTPUT_ROOT", cfg.paths.output_root)
    cfg.data.source = os.getenv("I2T_DATA_SOURCE", cfg.data.source)
    cfg.data.prompt = os.getenv("I2T_PROMPT", cfg.data.prompt)
    cfg.train.micro_batch_size = int(os.getenv("I2T_MICRO_BATCH", cfg.train.micro_batch_size))
    cfg.train.global_batch_size = int(os.getenv("I2T_GLOBAL_BATCH", cfg.train.global_batch_size))
    cfg.train.max_steps = int(os.getenv("I2T_MAX_STEPS", cfg.train.max_steps))
    cfg.data.limit = int(os.environ["I2T_DATASET_LIMIT"]) if "I2T_DATASET_LIMIT" in os.environ else cfg.data.limit
    if "I2T_GRAD_CHECKPOINTING" in os.environ:
        cfg.model.gradient_checkpointing = os.environ["I2T_GRAD_CHECKPOINTING"].lower() in {"1", "true", "yes"}
    if "I2T_ATTENTION_BACKEND" in os.environ:
        cfg.model.attention_backend = os.environ["I2T_ATTENTION_BACKEND"]
    if "I2T_BACKEND" in os.environ:
        cfg.train.backend = os.environ["I2T_BACKEND"]

    # Fail with an actionable message instead of a confusing error later on.
    missing_locations = [
        name
        for name, value in (
            ("I2T_MODEL", cfg.paths.model),
            ("I2T_DATA_SOURCE", cfg.data.source),
            ("I2T_OUTPUT_ROOT", cfg.paths.output_root),
        )
        if not str(value).strip()
    ]
    if missing_locations:
        raise ValueError(
            "Direct-SFT configuration is missing required location(s): "
            + ", ".join(missing_locations)
            + " (set them in the environment, see env.example, or in the config file)."
        )
    return cfg
