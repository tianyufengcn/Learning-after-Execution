from __future__ import annotations

import json
import os
import time
from pathlib import Path
import torch
from transformers import TrainerCallback


class RuntimeCallback(TrainerCallback):
    def __init__(self, output_dir: str):
        self.output_dir=Path(output_dir); self.start=time.time(); self.last=time.time()
        self.rank=int(os.getenv("RANK","0")); self.local_rank=int(os.getenv("LOCAL_RANK","0"))
        self.path=self.output_dir / "runtime" / f"rank{self.rank}.jsonl"
        self.path.parent.mkdir(parents=True,exist_ok=True)

    def on_log(self,args,state,control,logs=None,**kwargs):
        now=time.time(); record={"time":now,"step":state.global_step,"rank":self.rank,"local_rank":self.local_rank,"dt_since_log":now-self.last}
        self.last=now
        if logs: record.update({k:(float(v) if isinstance(v,(int,float)) else v) for k,v in logs.items()})
        if torch.cuda.is_available():
            dev=torch.cuda.current_device()
            record.update({
                "gpu":dev,
                "mem_allocated_gib":torch.cuda.memory_allocated(dev)/2**30,
                "mem_reserved_gib":torch.cuda.memory_reserved(dev)/2**30,
                "max_mem_allocated_gib":torch.cuda.max_memory_allocated(dev)/2**30,
                "max_mem_reserved_gib":torch.cuda.max_memory_reserved(dev)/2**30,
            })
        with self.path.open("a",encoding="utf-8") as f: f.write(json.dumps(record,ensure_ascii=False)+"\n")
