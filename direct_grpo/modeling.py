from __future__ import annotations

from typing import Any

import torch


def _auto_vlm_class():
    try:
        from transformers import Qwen3VLForConditionalGeneration
        return Qwen3VLForConditionalGeneration
    except ImportError:
        try:
            from transformers import AutoModelForImageTextToText
            return AutoModelForImageTextToText
        except ImportError:
            from transformers import AutoModelForVision2Seq
            return AutoModelForVision2Seq


def load_policy_and_processor(cfg: dict[str, Any]):
    """Load the Direct-SFT LoRA checkpoint as the trainable GRPO policy.

    Processor image bounds are frozen to the same 448x448 budget as Direct-SFT
    rather than relying on a model default that may dynamically resize images.
    """
    from peft import PeftConfig, PeftModel
    from transformers import AutoProcessor

    adapter_path = cfg.get("adapter_path")
    base_model_path = cfg.get("base_model_path")
    if adapter_path:
        peft_cfg = PeftConfig.from_pretrained(adapter_path)
        base_model_path = base_model_path or peft_cfg.base_model_name_or_path
    if not base_model_path:
        raise ValueError("model.base_model_path is required when adapter_path does not identify a PEFT base model")

    dtype_name = cfg.get("dtype", "bfloat16")
    dtype = getattr(torch, dtype_name) if isinstance(dtype_name, str) else dtype_name
    model_kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": cfg.get("trust_remote_code", True),
    }
    if cfg.get("attn_implementation"):
        model_kwargs["attn_implementation"] = cfg["attn_implementation"]

    model_cls = _auto_vlm_class()
    model = model_cls.from_pretrained(base_model_path, **model_kwargs)
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)

    image_size = int(cfg.get("image_size", 448))
    pixels = image_size * image_size
    processor_path = cfg.get("processor_path") or base_model_path
    processor = AutoProcessor.from_pretrained(
        processor_path,
        trust_remote_code=cfg.get("trust_remote_code", True),
        min_pixels=int(cfg.get("min_pixels", pixels)),
        max_pixels=int(cfg.get("max_pixels", pixels)),
    )
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        # Generation uses left-padding inside current TRL GRPOTrainer. Setting a
        # valid pad token here avoids backend-dependent fallbacks.
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

    if hasattr(model, "config"):
        model.config.use_cache = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    if trainable == 0:
        raise RuntimeError("Policy has no trainable parameters. Expected a trainable Direct-SFT LoRA adapter.")
    print(f"[model] base={base_model_path}")
    print(f"[model] adapter={adapter_path}")
    print(f"[model] image_protocol={image_size}x{image_size} min_pixels=max_pixels={pixels}")
    print(f"[model] trainable={trainable:,} / total={total:,} ({100.0*trainable/total:.4f}%)")
    return model, processor
