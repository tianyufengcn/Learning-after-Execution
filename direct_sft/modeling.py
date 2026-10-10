from __future__ import annotations

import importlib.util
import logging
from typing import Iterable

import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoProcessor

from .config import ExperimentConfig

log=logging.getLogger(__name__)


def resolve_attention_backend(preferred: str) -> str:
    if preferred != "auto": return preferred
    return "flash_attention_2" if importlib.util.find_spec("flash_attn") else "sdpa"


def load_processor(cfg: ExperimentConfig):
    kwargs={}
    if cfg.data.min_pixels is not None: kwargs["min_pixels"]=cfg.data.min_pixels
    if cfg.data.max_pixels is not None: kwargs["max_pixels"]=cfg.data.max_pixels
    p=AutoProcessor.from_pretrained(cfg.paths.model, **kwargs)
    if getattr(p,"tokenizer",None) is not None:
        p.tokenizer.padding_side="right"
        if p.tokenizer.pad_token_id is None:
            p.tokenizer.pad_token=p.tokenizer.eos_token
    return p


def _load_qwen_model(model_path: str, attention_backend: str):
    kwargs=dict(torch_dtype=torch.bfloat16, attn_implementation=attention_backend)
    try:
        from transformers import Qwen3VLForConditionalGeneration
        return Qwen3VLForConditionalGeneration.from_pretrained(model_path, **kwargs)
    except ImportError:
        from transformers import AutoModelForImageTextToText
        return AutoModelForImageTextToText.from_pretrained(model_path, **kwargs)


def discover_language_lora_modules(model: nn.Module, target_suffixes: Iterable[str], excluded_substrings: Iterable[str]) -> list[str]:
    suffixes=tuple(target_suffixes); excluded=tuple(s.lower() for s in excluded_substrings)
    names=[]
    for name,module in model.named_modules():
        low=name.lower()
        if any(x in low for x in excluded): continue
        if not isinstance(module,nn.Linear): continue
        if name.endswith(suffixes): names.append(name)
    if not names:
        raise RuntimeError("No LoRA target modules discovered. Refusing to silently train zero parameters.")
    return sorted(names)


def build_model(cfg: ExperimentConfig):
    attention=resolve_attention_backend(cfg.model.attention_backend)
    log.info("attention_backend=%s",attention)
    model=_load_qwen_model(cfg.paths.model,attention)
    model.config.use_cache=False
    for p in model.parameters(): p.requires_grad=False

    targets=discover_language_lora_modules(model,cfg.model.lora_targets,cfg.model.exclude_module_substrings)
    lora=LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.model.lora_r,
        lora_alpha=cfg.model.lora_alpha,
        lora_dropout=cfg.model.lora_dropout,
        bias="none",
        target_modules=targets,
    )
    model=get_peft_model(model,lora)
    if cfg.model.gradient_checkpointing:
        if hasattr(model,"enable_input_require_grads"): model.enable_input_require_grads()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    total=sum(p.numel() for p in model.parameters())
    log.info("lora_target_count=%d trainable=%d total=%d trainable_pct=%.4f",len(targets),trainable,total,100*trainable/total)
    log.info("first_lora_targets=%s",targets[:30])
    return model,attention,targets,trainable,total
