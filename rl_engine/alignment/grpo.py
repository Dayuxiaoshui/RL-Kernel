# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kernel-Align Contributors

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import torch


@dataclass(frozen=True)
class GRPOConfig:
    """Configuration for the reference GRPO objective.

    ``global_token_mean`` matches the original issue #16 behavior and averages
    over all active completion tokens. ``sequence_mean`` first averages each
    active completion independently, then averages those per-sequence losses so
    long completions do not receive more weight than short completions.
    """

    clip_epsilon: float = 0.2
    kl_beta: float = 0.01
    advantage_eps: float = 1e-8
    loss_reduction: str = "global_token_mean"

    def __post_init__(self) -> None:
        if self.clip_epsilon < 0.0:
            raise ValueError("clip_epsilon must be non-negative")
        if self.kl_beta < 0.0:
            raise ValueError("kl_beta must be non-negative")
        if self.advantage_eps <= 0.0:
            raise ValueError("advantage_eps must be greater than zero")
        if self.loss_reduction not in {"global_token_mean", "sequence_mean"}:
            raise ValueError("loss_reduction must be 'global_token_mean' or 'sequence_mean'")


@dataclass(frozen=True)
class GRPOResult:
    """Tensors and scalar metrics produced by the GRPO objective."""

    loss: torch.Tensor
    policy_loss: torch.Tensor
    kl_loss: torch.Tensor
    advantages: torch.Tensor
    sequence_advantages: torch.Tensor
    ratio: torch.Tensor
    kl: torch.Tensor
    metrics: Mapping[str, Any]

    def with_metrics(self, extra: Mapping[str, Any]) -> GRPOResult:
        """Return a copy with additional scalar or string metrics."""

        return GRPOResult(
            loss=self.loss,
            policy_loss=self.policy_loss,
            kl_loss=self.kl_loss,
            advantages=self.advantages,
            sequence_advantages=self.sequence_advantages,
            ratio=self.ratio,
            kl=self.kl,
            metrics={**dict(self.metrics), **dict(extra)},
        )


def compute_group_relative_advantages(
    rewards: torch.Tensor,
    *,
    group_ids: Optional[torch.Tensor] = None,
    num_groups: Optional[int] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Normalize scalar rewards independently inside each prompt group."""

    if eps <= 0.0:
        raise ValueError("eps must be greater than zero")
    if rewards.ndim != 1:
        raise ValueError(f"rewards must be one-dimensional, got shape {tuple(rewards.shape)}")
    if rewards.numel() == 0:
        raise ValueError("rewards must contain at least one value")

    resolved_group_ids, resolved_num_groups = _resolve_group_ids(
        rewards,
        group_ids=group_ids,
        num_groups=num_groups,
    )
    rewards_fp32 = rewards.detach().to(dtype=torch.float32)
    advantages = torch.empty_like(rewards_fp32)

    for group_index in range(resolved_num_groups):
        mask = resolved_group_ids == group_index
        if not bool(mask.any().item()):
            raise ValueError(f"group {group_index} has no rewards")
        group_rewards = rewards_fp32[mask]
        mean = group_rewards.mean()
        std = group_rewards.std(unbiased=False)
        advantages[mask] = (group_rewards - mean) / std.clamp_min(float(eps))

    if not bool(torch.isfinite(advantages).all().item()):
        raise ValueError("computed group-relative advantages contain non-finite values")
    return advantages


def broadcast_sequence_advantages(
    sequence_advantages: torch.Tensor,
    completion_mask: torch.Tensor,
) -> torch.Tensor:
    """Broadcast per-sequence advantages onto active completion tokens only."""

    if sequence_advantages.ndim != 1:
        raise ValueError(
            "sequence_advantages must be one-dimensional, got shape "
            f"{tuple(sequence_advantages.shape)}"
        )
    if completion_mask.ndim != 2:
        raise ValueError(
            f"completion_mask must be two-dimensional, got shape {tuple(completion_mask.shape)}"
        )
    if sequence_advantages.shape[0] != completion_mask.shape[0]:
        raise ValueError(
            "sequence_advantages length must match completion_mask batch size, got "
            f"{sequence_advantages.shape[0]} and {completion_mask.shape[0]}"
        )

    mask = completion_mask.to(dtype=torch.bool, device=sequence_advantages.device)
    broadcast = sequence_advantages.float().unsqueeze(-1).expand_as(mask).clone()
    return broadcast.masked_fill(~mask, 0.0)


def compute_grpo_loss(
    current_logps: torch.Tensor,
    old_logps: torch.Tensor,
    ref_logps: torch.Tensor,
    rewards: torch.Tensor,
    completion_mask: torch.Tensor,
    *,
    group_ids: Optional[torch.Tensor] = None,
    num_groups: Optional[int] = None,
    config: Optional[GRPOConfig] = None,
) -> GRPOResult:
    """Compute the reference GRPO clipped policy objective."""

    cfg = config or GRPOConfig()
    _validate_loss_shapes(current_logps, old_logps, ref_logps, rewards, completion_mask)

    mask = completion_mask.to(dtype=torch.bool, device=current_logps.device)
    active_tokens = int(mask.sum().item())
    if active_tokens <= 0:
        raise ValueError("completion_mask must contain at least one active token")

    sequence_advantages = compute_group_relative_advantages(
        rewards.to(device=current_logps.device),
        group_ids=group_ids.to(device=current_logps.device) if group_ids is not None else None,
        num_groups=num_groups,
        eps=cfg.advantage_eps,
    )
    advantages = broadcast_sequence_advantages(sequence_advantages, mask)

    current_fp32 = current_logps.float()
    old_fp32 = old_logps.detach().to(device=current_logps.device, dtype=torch.float32)
    ref_fp32 = ref_logps.detach().to(device=current_logps.device, dtype=torch.float32)

    ratio = torch.exp(current_fp32 - old_fp32).masked_fill(~mask, 0.0)
    unclipped = ratio * advantages
    clipped_ratio = torch.clamp(
        ratio,
        1.0 - cfg.clip_epsilon,
        1.0 + cfg.clip_epsilon,
    )
    clipped = clipped_ratio * advantages
    token_policy_loss = -torch.minimum(unclipped, clipped).masked_fill(~mask, 0.0)

    diff = ref_fp32 - current_fp32
    kl = (torch.exp(diff) - diff - 1.0).masked_fill(~mask, 0.0)
    loss_terms = token_policy_loss + cfg.kl_beta * kl

    policy_loss = _reduce_loss_terms(token_policy_loss, mask, cfg.loss_reduction)
    kl_mean = _reduce_loss_terms(kl, mask, cfg.loss_reduction)
    kl_loss = cfg.kl_beta * kl_mean
    loss = _reduce_loss_terms(loss_terms, mask, cfg.loss_reduction)

    clipped_tokens = ((ratio < 1.0 - cfg.clip_epsilon) | (ratio > 1.0 + cfg.clip_epsilon)) & mask
    clip_fraction = clipped_tokens.to(dtype=torch.float32).sum() / float(active_tokens)

    active_advantages = advantages[mask]
    metrics = {
        "loss": _scalar(loss),
        "policy_loss": _scalar(policy_loss),
        "kl_loss": _scalar(kl_loss),
        "kl_mean": _scalar(kl_mean),
        "clip_fraction": _scalar(clip_fraction),
        "active_tokens": float(active_tokens),
        "reward_mean": _scalar(rewards.detach().float().mean()),
        "reward_std": _scalar(rewards.detach().float().std(unbiased=False)),
        "advantage_mean": _scalar(sequence_advantages.mean()),
        "advantage_std": _scalar(sequence_advantages.std(unbiased=False)),
        "token_advantage_mean": _scalar(active_advantages.mean()),
        "token_advantage_std": _scalar(active_advantages.std(unbiased=False)),
        "active_sequences": float(_active_sequence_count(mask)),
        "loss_reduction": cfg.loss_reduction,
    }

    return GRPOResult(
        loss=loss,
        policy_loss=policy_loss,
        kl_loss=kl_loss,
        advantages=advantages,
        sequence_advantages=sequence_advantages,
        ratio=ratio,
        kl=kl,
        metrics=metrics,
    )


def _resolve_group_ids(
    rewards: torch.Tensor,
    *,
    group_ids: Optional[torch.Tensor],
    num_groups: Optional[int],
) -> tuple[torch.Tensor, int]:
    if num_groups is not None and num_groups <= 0:
        raise ValueError("num_groups must be greater than zero")

    if group_ids is None:
        if num_groups is None:
            raise ValueError("group_ids or num_groups is required")
        if rewards.numel() % num_groups != 0:
            raise ValueError(
                "rewards length must be divisible by num_groups when group_ids are inferred"
            )
        group_size = rewards.numel() // num_groups
        inferred = torch.arange(num_groups, device=rewards.device, dtype=torch.long)
        return inferred.repeat_interleave(group_size), int(num_groups)

    if group_ids.ndim != 1:
        raise ValueError(f"group_ids must be one-dimensional, got shape {tuple(group_ids.shape)}")
    if group_ids.shape[0] != rewards.shape[0]:
        raise ValueError(
            f"group_ids length {group_ids.shape[0]} must match rewards length {rewards.shape[0]}"
        )

    resolved = group_ids.to(device=rewards.device, dtype=torch.long)
    if bool((resolved < 0).any().item()):
        raise ValueError("group_ids must be non-negative")
    inferred_num_groups = int(resolved.max().item()) + 1 if resolved.numel() else 0
    resolved_num_groups = int(num_groups) if num_groups is not None else inferred_num_groups
    if inferred_num_groups > resolved_num_groups:
        raise ValueError("group_ids contain an id greater than or equal to num_groups")
    for group_index in range(resolved_num_groups):
        if not bool((resolved == group_index).any().item()):
            raise ValueError(f"group {group_index} has no rewards")
    return resolved, resolved_num_groups


def _validate_loss_shapes(
    current_logps: torch.Tensor,
    old_logps: torch.Tensor,
    ref_logps: torch.Tensor,
    rewards: torch.Tensor,
    completion_mask: torch.Tensor,
) -> None:
    if current_logps.ndim != 2:
        raise ValueError(f"current_logps must be two-dimensional, got {tuple(current_logps.shape)}")
    if old_logps.shape != current_logps.shape:
        raise ValueError("old_logps shape must match current_logps shape")
    if ref_logps.shape != current_logps.shape:
        raise ValueError("ref_logps shape must match current_logps shape")
    if completion_mask.shape != current_logps.shape:
        raise ValueError("completion_mask shape must match current_logps shape")
    if rewards.ndim != 1:
        raise ValueError(f"rewards must be one-dimensional, got shape {tuple(rewards.shape)}")
    if rewards.shape[0] != current_logps.shape[0]:
        raise ValueError("rewards length must match current_logps batch size")


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return values.masked_fill(~mask, 0.0).sum() / mask.sum().clamp_min(1).to(dtype=torch.float32)


def _sequence_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked = values.masked_fill(~mask, 0.0)
    counts = mask.sum(dim=1)
    active_rows = counts > 0
    if not bool(active_rows.any().item()):
        return values.new_tensor(0.0, dtype=torch.float32)
    per_sequence = masked.sum(dim=1) / counts.clamp_min(1).to(dtype=torch.float32)
    return per_sequence[active_rows].mean()


def _reduce_loss_terms(values: torch.Tensor, mask: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "global_token_mean":
        return _masked_mean(values, mask)
    if mode == "sequence_mean":
        return _sequence_mean(values, mask)
    raise ValueError(f"unsupported loss reduction mode {mode!r}")


def _active_sequence_count(mask: torch.Tensor) -> int:
    return int((mask.sum(dim=1) > 0).sum().item())


def _scalar(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())
