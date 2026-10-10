from __future__ import annotations

import argparse
import json
import logging
import math
import os
from pathlib import Path
import subprocess
import sys

import torch
from transformers import Trainer, TrainingArguments, set_seed
from transformers.trainer_utils import get_last_checkpoint

from .callbacks import RuntimeCallback
from .collator import Qwen3VLI2TCollator
from .config import load_config
from .data import I2TTikZDataset, discover_data_source
from .modeling import build_model, load_processor
from .reporting import write_json, write_training_summary


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--config",required=True)
    p.add_argument("--source",default=None)
    p.add_argument("--output-dir",default=None)
    p.add_argument("--resume",default="auto",help="auto | none | /path/to/checkpoint")
    return p.parse_args()


def find_deepspeed_config() -> str:
    """Locate configs/deepspeed/zero2.json in the supported source layouts."""
    here = Path(__file__).resolve()
    candidates = [
        here.parents[1] / "configs" / "deepspeed" / "zero2.json",
        Path.cwd() / "configs" / "deepspeed" / "zero2.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return str(candidates[0])


def configure_fast_math():
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    torch.set_float32_matmul_precision("high")


def main():
    args=parse_args(); cfg=load_config(args.config)
    if args.source: cfg.data.source=args.source
    if args.output_dir: cfg.paths.output_root=str(Path(args.output_dir).parent); cfg.name=Path(args.output_dir).name
    configure_fast_math(); set_seed(cfg.train.seed)
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log=logging.getLogger("i2t_direct_sft")

    world=int(os.getenv("WORLD_SIZE","1")); rank=int(os.getenv("RANK","0")); local_rank=int(os.getenv("LOCAL_RANK","0"))
    source=discover_data_source(cfg.paths.work_root,cfg.data.cohort_size) if cfg.data.source=="auto" else Path(cfg.data.source)
    output=cfg.output_dir; output.mkdir(parents=True,exist_ok=True)
    if rank==0: log.info("data_source=%s cohort=%d output=%s",source,cfg.data.cohort_size,output)

    processor=load_processor(cfg)
    dataset=I2TTikZDataset(source,cfg.data)
    collator=Qwen3VLI2TCollator(
        processor=processor,
        prompt=cfg.data.prompt,
        model_max_length=cfg.data.model_max_length,
        max_tikz_tokens=cfg.data.max_tikz_tokens,
        image_size=cfg.data.image_size,
        strict_image_size=cfg.data.strict_image_size,
        require_end_document=cfg.data.require_end_document,
    )
    model,attention,targets,trainable,total=build_model(cfg)

    denom=max(1,world*cfg.train.micro_batch_size)
    grad_acc=max(1,math.ceil(cfg.train.global_batch_size/denom))
    effective=denom*grad_acc
    deepspeed=None
    if cfg.train.backend=="zero2":
        deepspeed=find_deepspeed_config()

    targs=TrainingArguments(
        output_dir=str(output/"checkpoints"),
        num_train_epochs=cfg.train.epochs,
        max_steps=cfg.train.max_steps,
        per_device_train_batch_size=cfg.train.micro_batch_size,
        gradient_accumulation_steps=grad_acc,
        learning_rate=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
        max_grad_norm=cfg.train.max_grad_norm,
        warmup_ratio=cfg.train.warmup_ratio,
        lr_scheduler_type=cfg.train.lr_scheduler_type,
        bf16=True, fp16=False, tf32=True,
        optim="adamw_torch_fused",
        logging_strategy="steps", logging_steps=cfg.train.logging_steps, logging_first_step=True,
        save_strategy="steps", save_steps=cfg.train.save_steps, save_total_limit=cfg.train.save_total_limit,
        report_to=[] if cfg.train.report_to=="none" else [cfg.train.report_to],
        remove_unused_columns=False,
        dataloader_num_workers=cfg.train.dataloader_num_workers,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=cfg.train.dataloader_num_workers>0,
        dataloader_prefetch_factor=cfg.train.dataloader_prefetch_factor if cfg.train.dataloader_num_workers>0 else None,
        ddp_find_unused_parameters=False,
        gradient_checkpointing=cfg.model.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant":False} if cfg.model.gradient_checkpointing else None,
        torch_compile=cfg.train.torch_compile,
        deepspeed=deepspeed,
        seed=cfg.train.seed,
        data_seed=cfg.train.seed,
    )
    callback=RuntimeCallback(str(output))
    trainer=Trainer(model=model,args=targs,train_dataset=dataset,data_collator=collator,callbacks=[callback])

    meta={
        "name":cfg.name,"source":str(source),"n_samples":len(dataset),"model":cfg.paths.model,
        "attention_backend":attention,"world_size":world,"micro_batch_per_gpu":cfg.train.micro_batch_size,
        "gradient_accumulation":grad_acc,"effective_global_batch":effective,"epochs":cfg.train.epochs,
        "learning_rate":cfg.train.learning_rate,"lora_r":cfg.model.lora_r,"lora_alpha":cfg.model.lora_alpha,
        "lora_target_count":len(targets),"trainable_params":trainable,"total_params":total,
        "trainable_pct":100*trainable/total,"gradient_checkpointing":cfg.model.gradient_checkpointing,
        "torch_compile":cfg.train.torch_compile,"backend":cfg.train.backend,"model_max_length":cfg.data.model_max_length,
        "max_tikz_tokens":cfg.data.max_tikz_tokens,
        "image_size":cfg.data.image_size,
        "strict_image_size":cfg.data.strict_image_size,
        "min_pixels":cfg.data.min_pixels,
        "max_pixels":cfg.data.max_pixels,
        "max_grad_norm":cfg.train.max_grad_norm,
        "prompt":cfg.data.prompt,
    }
    if rank==0:
        write_json(output/"run_config.json",meta)
        write_json(output/"lora_targets.json",targets)

    resume=False
    if args.resume.lower() == "auto":
        ckpt_dir=output/"checkpoints"
        if ckpt_dir.exists():
            last=get_last_checkpoint(str(ckpt_dir))
            resume=last if last else False
    elif args.resume.lower() in {"none","false","0","no"}:
        resume=False
    else:
        resume=args.resume
    if rank==0: log.info("resume_from_checkpoint=%s",resume)
    result=trainer.train(resume_from_checkpoint=resume)
    metrics=dict(result.metrics)
    if torch.cuda.is_available(): metrics["rank0_peak_allocated_gib"]=torch.cuda.max_memory_allocated()/2**30
    benchmark_only=os.getenv("I2T_BENCHMARK_ONLY","0").lower() in {"1","true","yes"}
    if rank==0:
        write_training_summary(output,meta,metrics)
    if benchmark_only:
        if rank==0: log.info("BENCHMARK_ONLY complete; adapter save skipped: %s",output)
        return
    trainer.save_model(str(output/"final_adapter"))
    if rank==0:
        processor.save_pretrained(str(output/"final_adapter"))
        (output/"DONE").write_text(json.dumps({"status":"ok","name":cfg.name,"metrics":metrics},indent=2)+"\n",encoding="utf-8")
        log.info("DONE %s",output)

if __name__=="__main__": main()
