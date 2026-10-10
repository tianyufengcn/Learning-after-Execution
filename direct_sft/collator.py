from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import torch

from .data import NormalizedExample, open_image

IGNORE_INDEX=-100


def make_labels(input_ids: torch.Tensor, prompt_len: int, attention_mask: torch.Tensor) -> torch.Tensor:
    labels=input_ids.clone()
    labels[:prompt_len]=IGNORE_INDEX
    labels[attention_mask == 0]=IGNORE_INDEX
    return labels


@dataclass
class Qwen3VLI2TCollator:
    processor: Any
    prompt: str
    model_max_length: int = 4096
    max_tikz_tokens: int = 2048
    image_size: int = 448
    strict_image_size: bool = True
    require_end_document: bool = True

    def _messages(self, image, target: str | None):
        user={"role":"user","content":[{"type":"image","image":image},{"type":"text","text":self.prompt}]}
        if target is None:
            return [user]
        return [user,{"role":"assistant","content":[{"type":"text","text":target}]}]

    def _encode_one(self, ex: NormalizedExample):
        image=open_image(ex.image)
        if self.strict_image_size and image.size != (self.image_size, self.image_size):
            raise ValueError(
                f"{ex.sample_id}: expected fixed {self.image_size}x{self.image_size} GT image, got {image.size}"
            )
        if self.require_end_document and not ex.tikz.rstrip().endswith(r"\end{document}"):
            raise ValueError(f"{ex.sample_id}: GT TikZ does not end with \\end{{document}}")
        target_ids=self.processor.tokenizer(ex.tikz, add_special_tokens=False).input_ids
        if len(target_ids) > self.max_tikz_tokens:
            raise ValueError(f"{ex.sample_id}: TikZ target has {len(target_ids)} tokens > {self.max_tikz_tokens}")

        full=self.processor.apply_chat_template(
            self._messages(image,ex.tikz), tokenize=True, add_generation_prompt=False,
            return_dict=True, return_tensors="pt"
        )
        prompt=self.processor.apply_chat_template(
            self._messages(image,None), tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt"
        )
        full.pop("token_type_ids",None); prompt.pop("token_type_ids",None)
        flen=int(full["attention_mask"][0].sum().item())
        plen=int(prompt["attention_mask"][0].sum().item())
        if flen > self.model_max_length:
            raise ValueError(f"{ex.sample_id}: multimodal length {flen} > model_max_length={self.model_max_length}; refusing silent truncation")
        if plen >= flen:
            raise ValueError(f"{ex.sample_id}: invalid prompt/full lengths {plen}/{flen}")
        if not torch.equal(full["input_ids"][0,:plen], prompt["input_ids"][0,:plen]):
            raise RuntimeError(f"{ex.sample_id}: prompt is not an exact prefix of full chat template; cannot construct safe labels")
        return ex.sample_id, full, plen

    def __call__(self, examples: list[NormalizedExample]):
        encoded=[self._encode_one(ex) for ex in examples]
        tokenizer=self.processor.tokenizer
        pad_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        max_len=max(int(item[1]["attention_mask"][0].sum().item()) for item in encoded)
        b=len(encoded)
        input_ids=torch.full((b,max_len),pad_id,dtype=torch.long)
        attention_mask=torch.zeros((b,max_len),dtype=torch.long)
        labels=torch.full((b,max_len),IGNORE_INDEX,dtype=torch.long)
        pixel_values=[]; image_grid=[]
        mm_token_type_ids=None
        sample_ids=[]
        extra_cat: dict[str,list[torch.Tensor]]={}

        for i,(sid,feat,plen) in enumerate(encoded):
            sample_ids.append(sid)
            n=int(feat["attention_mask"][0].sum().item())
            ids=feat["input_ids"][0,:n]
            mask=feat["attention_mask"][0,:n]
            input_ids[i,:n]=ids
            attention_mask[i,:n]=mask
            labels[i,:n]=make_labels(ids,plen,mask)
            if "mm_token_type_ids" in feat:
                if mm_token_type_ids is None:
                    mm_token_type_ids=torch.zeros((b,max_len),dtype=torch.long)
                mm_token_type_ids[i,:n]=feat["mm_token_type_ids"][0,:n]
            if "pixel_values" in feat: pixel_values.append(feat["pixel_values"])
            if "image_grid_thw" in feat: image_grid.append(feat["image_grid_thw"])
            for key,val in feat.items():
                if key in {"input_ids","attention_mask","pixel_values","image_grid_thw","token_type_ids","mm_token_type_ids"}: continue
                if torch.is_tensor(val) and val.ndim>0:
                    extra_cat.setdefault(key,[]).append(val)

        batch={"input_ids":input_ids,"attention_mask":attention_mask,"labels":labels}
        if mm_token_type_ids is not None:
            batch["mm_token_type_ids"]=mm_token_type_ids
        if pixel_values: batch["pixel_values"]=torch.cat(pixel_values,dim=0)
        if image_grid: batch["image_grid_thw"]=torch.cat(image_grid,dim=0)
        # Qwen image-only SFT normally needs no additional tensor keys; preserve compatible per-image tensors if present.
        for key,vals in extra_cat.items():
            try: batch[key]=torch.cat(vals,dim=0)
            except Exception: pass
        return batch
