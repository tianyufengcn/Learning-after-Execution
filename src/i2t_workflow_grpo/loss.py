from __future__ import annotations

import torch


def selected_logps(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Exact selected-token log probabilities without materializing log_softmax."""
    selected = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1).float()
    log_z = torch.logsumexp(logits.float(), dim=-1)
    return selected - log_z


def forward_completion_logps(model, batch: dict[str, torch.Tensor], completion_mask: torch.Tensor) -> torch.Tensor:
    """Compute exact policy log-probs only for completion-tail positions."""
    max_completion = int(completion_mask.shape[1])
    if max_completion <= 0:
        return torch.zeros_like(completion_mask)

    keep = max_completion + 1
    kwargs = dict(batch)
    try:
        outputs = model(**kwargs, logits_to_keep=keep)
    except (TypeError, ValueError):
        outputs = model(**kwargs)
    logits = outputs.logits
    if logits.shape[1] < keep:
        raise RuntimeError(
            f"Model returned only {logits.shape[1]} logits positions, need at least {keep} for completion loss"
        )
    suffix_logits = logits[:, -keep:-1, :]
    targets = batch["input_ids"][:, -max_completion:]
    return selected_logps(suffix_logits, targets)


def clipped_grpo_token_loss(
    new_logps: torch.Tensor,
    old_logps: torch.Tensor,
    advantages: torch.Tensor,
    epsilon_low: float,
    epsilon_high: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Token-level Clip-Higher GRPO surrogate; returns loss and clipped mask."""
    log_ratio = new_logps - old_logps
    ratio = torch.exp(log_ratio)
    clipped = torch.clamp(ratio, 1.0 - epsilon_low, 1.0 + epsilon_high)
    adv = advantages[:, None]
    loss1 = ratio * adv
    loss2 = clipped * adv
    loss = -torch.minimum(loss1, loss2)
    was_clipped = (ratio < 1.0 - epsilon_low) | (ratio > 1.0 + epsilon_high)
    return loss, was_clipped
