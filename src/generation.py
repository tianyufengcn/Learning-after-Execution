from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import torch
from PIL import Image

from .prompt_contract import direct_messages, repair_messages, revision_messages

END_DOCUMENT = r"\end{document}"


def open_rgb(path: str) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.load()
    return image


@dataclass(frozen=True)
class PromptSpec:
    kind: str
    target_image_path: str
    current_image_path: str | None = None
    current_tikz: str | None = None
    failure_status: str | None = None

    def messages(self) -> list[dict[str, Any]]:
        target = open_rgb(self.target_image_path)
        if self.kind == "direct":
            return direct_messages(target)
        if self.kind == "revision":
            if not self.current_image_path or self.current_tikz is None:
                raise ValueError("revision PromptSpec requires current_image_path and current_tikz")
            return revision_messages(target, open_rgb(self.current_image_path), self.current_tikz)
        if self.kind == "repair":
            if self.current_tikz is None:
                raise ValueError("repair PromptSpec requires current_tikz")
            return repair_messages(target, self.current_tikz, self.failure_status or "render_failed")
        raise ValueError(f"Unsupported prompt kind: {self.kind}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "target_image_path": self.target_image_path,
            "current_image_path": self.current_image_path,
            "current_tikz": self.current_tikz,
            "failure_status": self.failure_status,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PromptSpec":
        return cls(**payload)


@dataclass
class PromptEncoding:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    extras: dict[str, torch.Tensor] = field(default_factory=dict)

    @property
    def prompt_len(self) -> int:
        return int(self.attention_mask.sum().item())


@dataclass
class GeneratedAction:
    spec: PromptSpec
    prompt: PromptEncoding
    completion_ids: torch.Tensor
    text: str
    hit_max_tokens: bool
    ends_with_document: bool
    stopped_on_end_document: bool = False
    generation_batch_elapsed_s: float = 0.0

    @property
    def completion_tokens(self) -> int:
        return int(self.completion_ids.numel())

    @property
    def full_sequence_tokens(self) -> int:
        return int(self.prompt.prompt_len + self.completion_tokens)


def encode_prompt(processor, spec: PromptSpec) -> PromptEncoding:
    encoded = processor.apply_chat_template(
        spec.messages(),
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    encoded.pop("token_type_ids", None)
    ids = encoded.pop("input_ids")[0].detach().cpu()
    mask = encoded.pop("attention_mask")[0].detach().cpu()
    extras = {k: v.detach().cpu() for k, v in encoded.items() if torch.is_tensor(v)}
    return PromptEncoding(ids, mask, extras)


# Backward-compatible private alias.
_encode_one = encode_prompt


def _pad_ids(rows: list[torch.Tensor], pad_id: int, side: str) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(int(x.numel()) for x in rows)
    ids = torch.full((len(rows), max_len), pad_id, dtype=torch.long)
    mask = torch.zeros((len(rows), max_len), dtype=torch.long)
    for i, row in enumerate(rows):
        n = int(row.numel())
        if side == "left":
            ids[i, max_len - n :] = row
            mask[i, max_len - n :] = 1
        else:
            ids[i, :n] = row
            mask[i, :n] = 1
    return ids, mask


def _is_prompt_seq_like(prompt_len: int, tensor: torch.Tensor) -> bool:
    if tensor.ndim == 1:
        return int(tensor.numel()) == int(prompt_len)
    return tensor.ndim == 2 and int(tensor.shape[0]) == 1 and int(tensor.shape[1]) == int(prompt_len)


def _collate_prompt_extra(pairs: list[tuple[int, torch.Tensor]], side: str = "left") -> torch.Tensor:
    """Collate one Qwen multimodal extra field.

    Qwen3-VL emits per-token fields such as ``mm_token_type_ids`` with shape
    ``(1, prompt_len)`` and per-image fields such as ``pixel_values`` and
    ``image_grid_thw``. The former must follow the same prompt padding as
    ``input_ids``; the latter are concatenated over images.
    """
    if not pairs:
        raise ValueError("Cannot collate an empty extra field")
    if all(_is_prompt_seq_like(plen, tensor) for plen, tensor in pairs):
        max_len = max(int(plen) for plen, _ in pairs)
        padded: list[torch.Tensor] = []
        for plen, tensor in pairs:
            view = tensor[0] if tensor.ndim == 2 else tensor
            pad_n = max_len - int(plen)
            pad = torch.zeros(pad_n, dtype=view.dtype)
            merged = torch.cat([pad, view], dim=0) if side == "left" else torch.cat([view, pad], dim=0)
            padded.append(merged)
        return torch.stack(padded, dim=0)
    return torch.cat([tensor for _, tensor in pairs], dim=0)


def collate_prompt_encodings(
    encodings: list[PromptEncoding], pad_id: int, side: str = "left"
) -> dict[str, torch.Tensor]:
    ids, mask = _pad_ids([e.input_ids for e in encodings], pad_id, side)
    out: dict[str, torch.Tensor] = {"input_ids": ids, "attention_mask": mask}
    all_keys = sorted(set().union(*(e.extras.keys() for e in encodings)))
    for key in all_keys:
        vals = [e.extras.get(key) for e in encodings]
        if any(v is None for v in vals):
            continue
        try:
            out[key] = _collate_prompt_extra(
                [(int(e.input_ids.numel()), v) for e, v in zip(encodings, vals, strict=True) if v is not None],
                side=side,
            )
        except Exception as exc:
            raise RuntimeError(f"Cannot collate multimodal field {key!r}") from exc
    return out


def _extend_prompt_seq_field(
    pairs: list[tuple[GeneratedAction, torch.Tensor]], max_full: int
) -> torch.Tensor:
    """Extend a prompt-length per-token field to the full prompt+completion sequence.

    Full sequences are left-padded. Completion tokens are plain text, so their
    multimodal token type is zero.
    """
    rows: list[torch.Tensor] = []
    for action, tensor in pairs:
        prompt_len = int(action.prompt.input_ids.numel())
        completion_len = int(action.completion_ids.numel())
        pad_left = max_full - (prompt_len + completion_len)
        view = tensor[0] if tensor.ndim == 2 else tensor
        row = torch.cat(
            [
                torch.zeros(pad_left, dtype=view.dtype),
                view,
                torch.zeros(completion_len, dtype=view.dtype),
            ],
            dim=0,
        )
        rows.append(row)
    return torch.stack(rows, dim=0)


def collate_action_inputs(
    actions: list[GeneratedAction], pad_id: int
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Build left-padded full sequences and completion-tail masks.

    The implementation supports heterogeneous revision/repair prompts. Qwen
    per-token multimodal fields are extended through the completion, while
    per-image fields are concatenated over the batch.
    """
    full_rows: list[torch.Tensor] = []
    completion_lengths: list[int] = []
    for action in actions:
        full_rows.append(torch.cat([action.prompt.input_ids, action.completion_ids], dim=0))
        completion_lengths.append(int(action.completion_ids.numel()))

    ids, attention = _pad_ids(full_rows, pad_id, side="left")
    max_full = int(ids.shape[1])
    max_completion = max(completion_lengths) if completion_lengths else 0
    token_mask = torch.zeros((len(actions), max_completion), dtype=torch.float32)
    for i, n in enumerate(completion_lengths):
        if n:
            token_mask[i, max_completion - n :] = 1.0

    out: dict[str, torch.Tensor] = {"input_ids": ids, "attention_mask": attention}
    all_keys = sorted(set().union(*(a.prompt.extras.keys() for a in actions)))
    for key in all_keys:
        vals = [a.prompt.extras.get(key) for a in actions]
        if any(v is None for v in vals):
            continue
        pairs = [(a, v) for a, v in zip(actions, vals, strict=True) if v is not None]
        if all(_is_prompt_seq_like(int(a.prompt.input_ids.numel()), v) for a, v in pairs):
            out[key] = _extend_prompt_seq_field(pairs, max_full)
        else:
            out[key] = torch.cat([v for _, v in pairs], dim=0)
    return out, token_mask


def _trim_completion(
    row: torch.Tensor,
    eos_id: int | None,
    pad_id: int | None,
    max_new_tokens: int,
) -> tuple[torch.Tensor, bool]:
    """Remove generation padding and detect true max-token truncation.

    Hugging Face generation can stop one row through a per-sequence stopping
    criterion while other rows in the static batch continue. Finished rows are
    then padded. We therefore have to consider *both* EOS and PAD even when an
    EOS id exists; otherwise a row that correctly stopped at ``\\end{document}``
    could be mis-recorded as a max-length completion simply because a longer
    neighbor kept the batch alive.
    """
    row = row.detach().cpu().long()
    raw_len = int(row.numel())
    stop = raw_len
    found_eos = False
    found_pad = False

    if eos_id is not None:
        eos_pos = (row == int(eos_id)).nonzero(as_tuple=False)
        if eos_pos.numel():
            stop = min(stop, int(eos_pos[0].item()) + 1)
            found_eos = True

    # If PAD and EOS are the same token, preserve the EOS token using the EOS
    # branch above. With distinct ids, PAD marks the first position after a row
    # was individually finished by EOS or StopStringCriteria.
    if pad_id is not None and (eos_id is None or int(pad_id) != int(eos_id)):
        pad_pos = (row == int(pad_id)).nonzero(as_tuple=False)
        if pad_pos.numel():
            pad_stop = int(pad_pos[0].item())
            if pad_stop < stop:
                stop = pad_stop
                found_pad = True

    trimmed = row[:stop]
    ended_early = found_eos or found_pad or stop < raw_len
    hit_max = (not ended_early) and int(trimmed.numel()) >= int(max_new_tokens)
    return trimmed, hit_max


def _truncate_to_first_end_document(
    ids: torch.Tensor, tokenizer
) -> tuple[torch.Tensor, str, bool]:
    """Keep the shortest generated-token prefix containing ``\\end{document}``.

    Native ``StopStringCriteria`` should already terminate the row at this
    point. This post-generation guard makes the semantic contract explicit and
    protects older/fallback generation paths. It never re-tokenizes generated
    text: the returned ids are an exact prefix of the tokens sampled by the
    policy, preserving policy-logprob correctness.
    """
    ids = ids.detach().cpu().long()
    text = tokenizer.decode(ids, skip_special_tokens=True)
    if END_DOCUMENT not in text:
        return ids, text, False

    lo, hi = 1, int(ids.numel())
    while lo < hi:
        mid = (lo + hi) // 2
        prefix_text = tokenizer.decode(ids[:mid], skip_special_tokens=True)
        if END_DOCUMENT in prefix_text:
            hi = mid
        else:
            lo = mid + 1
    clipped = ids[:lo]
    return clipped, tokenizer.decode(clipped, skip_special_tokens=True), True


def action_to_transport(action: GeneratedAction) -> dict[str, Any]:
    """Serialize an action without large image tensors for cross-rank transport."""
    return {
        "spec": action.spec.to_dict(),
        "completion_ids": action.completion_ids.detach().cpu().tolist(),
        "text": action.text,
        "hit_max_tokens": bool(action.hit_max_tokens),
        "ends_with_document": bool(action.ends_with_document),
        "stopped_on_end_document": bool(action.stopped_on_end_document),
        "prompt_len": int(action.prompt.prompt_len),
        "generation_batch_elapsed_s": float(action.generation_batch_elapsed_s),
    }


def action_from_transport(payload: dict[str, Any], processor) -> GeneratedAction:
    spec = PromptSpec.from_dict(payload["spec"])
    prompt = encode_prompt(processor, spec)
    return GeneratedAction(
        spec=spec,
        prompt=prompt,
        completion_ids=torch.tensor(payload["completion_ids"], dtype=torch.long),
        text=str(payload["text"]),
        hit_max_tokens=bool(payload.get("hit_max_tokens", False)),
        ends_with_document=bool(payload.get("ends_with_document", END_DOCUMENT in str(payload["text"]))),
        stopped_on_end_document=bool(payload.get("stopped_on_end_document", False)),
        generation_batch_elapsed_s=float(payload.get("generation_batch_elapsed_s", 0.0)),
    )


class QwenWorkflowGenerator:
    backend_name = "transformers"

    def __init__(self, model, processor, accelerator, cfg: dict[str, Any]) -> None:
        self.model = model
        self.processor = processor
        self.accelerator = accelerator
        self.temperature = float(cfg.get("temperature", 1.0))
        self.top_p = float(cfg.get("top_p", 0.9))
        self.repetition_penalty = float(cfg.get("repetition_penalty", 1.0))
        self.stage0_max = int(cfg.get("stage0_max_completion_length", 8192))
        self.stage1_max = int(cfg.get("stage1_max_completion_length", 8192))
        self.stage0_microbatch = int(cfg.get("stage0_generation_microbatch", 4))
        self.stage1_microbatch = int(cfg.get("stage1_generation_microbatch", 2))
        self.stop_on_end_document = bool(cfg.get("stop_on_end_document", True))

    @property
    def tokenizer(self):
        return self.processor.tokenizer

    def sync_policy(self, global_step: int) -> None:
        """Transformers backend shares learner weights, so synchronization is a no-op."""
        return None

    def _generate_chunk(self, specs: list[PromptSpec], max_new_tokens: int) -> list[GeneratedAction]:
        import time

        encodings = [encode_prompt(self.processor, spec) for spec in specs]
        pad_id = int(self.tokenizer.pad_token_id)
        batch = collate_prompt_encodings(encodings, pad_id, side="left")
        batch = {k: v.to(self.accelerator.device, non_blocking=True) for k, v in batch.items()}
        padded_prompt_len = int(batch["input_ids"].shape[1])
        policy = self.accelerator.unwrap_model(self.model)
        was_training = policy.training
        policy.eval()
        kwargs: dict[str, Any] = {
            **batch,
            "do_sample": True,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
            "max_new_tokens": max_new_tokens,
            "use_cache": True,
            "pad_token_id": pad_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if self.stop_on_end_document:
            # Native StopStringCriteria is tokenization-aware and returns a
            # per-sequence stopping mask, so a short program stops at its first
            # complete \end{document} even when another row keeps decoding.
            try:
                from transformers import StopStringCriteria, StoppingCriteriaList

                kwargs["stopping_criteria"] = StoppingCriteriaList(
                    [StopStringCriteria(self.tokenizer, [END_DOCUMENT])]
                )
            except ImportError:
                # Compatibility path for older Transformers. Formal server
                # preflight reports whether native stop-string support exists.
                kwargs["stop_strings"] = [END_DOCUMENT]
                kwargs["tokenizer"] = self.tokenizer

        started = time.monotonic()
        with torch.inference_mode():
            out = policy.generate(**kwargs)
        elapsed = time.monotonic() - started
        if was_training:
            policy.train()

        new = out[:, padded_prompt_len:]
        results: list[GeneratedAction] = []
        for spec, enc, row in zip(specs, encodings, new, strict=True):
            ids, hit_max = _trim_completion(row, self.tokenizer.eos_token_id, pad_id, max_new_tokens)
            ids, text, complete = _truncate_to_first_end_document(ids, self.tokenizer)
            # A semantically complete program is valid even when its closing
            # marker happens to land exactly on the configured token budget.
            if self.stop_on_end_document and complete:
                hit_max = False
            results.append(
                GeneratedAction(
                    spec=spec,
                    prompt=enc,
                    completion_ids=ids,
                    text=text,
                    hit_max_tokens=hit_max,
                    ends_with_document=complete,
                    stopped_on_end_document=self.stop_on_end_document and complete,
                    generation_batch_elapsed_s=float(elapsed),
                )
            )
        return results

    def generate_iter(self, specs: list[PromptSpec], stage: int) -> Iterable[list[GeneratedAction]]:
        if not specs:
            return
        max_new = self.stage0_max if stage == 0 else self.stage1_max
        micro = self.stage0_microbatch if stage == 0 else self.stage1_microbatch
        for start in range(0, len(specs), micro):
            yield self._generate_chunk(specs[start : start + micro], max_new)

    def generate(self, specs: list[PromptSpec], stage: int) -> list[GeneratedAction]:
        output: list[GeneratedAction] = []
        for chunk in self.generate_iter(specs, stage):
            output.extend(chunk)
        return output
