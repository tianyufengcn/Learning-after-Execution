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
    from peft import PeftConfig, PeftModel
    from transformers import AutoProcessor

    adapter_path = cfg.get("adapter_path")
    base_model_path = cfg.get("base_model_path")
    if adapter_path:
        peft_cfg = PeftConfig.from_pretrained(adapter_path)
        base_model_path = base_model_path or peft_cfg.base_model_name_or_path
    if not base_model_path:
        raise ValueError("model.base_model_path is required")

    dtype_name = cfg.get("dtype", "bfloat16")
    dtype = getattr(torch, dtype_name) if isinstance(dtype_name, str) else dtype_name
    kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": cfg.get("trust_remote_code", True),
    }
    if cfg.get("attn_implementation"):
        kwargs["attn_implementation"] = cfg["attn_implementation"]

    model = _auto_vlm_class().from_pretrained(base_model_path, **kwargs)
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)

    image_size = int(cfg.get("image_size", 448))
    pixels = image_size * image_size
    processor = AutoProcessor.from_pretrained(
        cfg.get("processor_path") or base_model_path,
        trust_remote_code=cfg.get("trust_remote_code", True),
        min_pixels=int(cfg.get("min_pixels", pixels)),
        max_pixels=int(cfg.get("max_pixels", pixels)),
    )
    tok = getattr(processor, "tokenizer", None)
    if tok is not None:
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"

    if hasattr(model, "config"):
        model.config.use_cache = False
    if cfg.get("disable_dropout", True):
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
    if cfg.get("gradient_checkpointing", True):
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    if trainable == 0:
        raise RuntimeError("No trainable parameters. Expected a trainable LoRA adapter.")
    print(f"[model] base={base_model_path}")
    print(f"[model] adapter={adapter_path}")
    print(f"[model] trainable={trainable:,}/{total:,} ({100*trainable/total:.4f}%)")
    return model, processor
