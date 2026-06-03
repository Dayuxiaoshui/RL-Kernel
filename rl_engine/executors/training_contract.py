# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Protocol, Sequence

import torch

from rl_engine.alignment.grpo import (
    GRPOConfig,
    broadcast_sequence_advantages,
    compute_group_relative_advantages,
    compute_grpo_loss,
)
from rl_engine.testing import SyntheticRLKernelBatch, make_synthetic_rl_kernel_batch
from rl_engine.testing.reference_ops import selected_logprobs_reference

_OLD_LOGP_KEYS = (
    "old_logps",
    "old_logprobs",
    "old_selected_logps",
    "old_selected_logprobs",
    "old_policy_logps",
    "old_policy_logprobs",
)
_REF_LOGP_KEYS = (
    "ref_logps",
    "ref_logprobs",
    "reference_logps",
    "reference_logprobs",
    "ref_selected_logps",
    "ref_selected_logprobs",
    "reference_selected_logps",
    "reference_selected_logprobs",
)


@dataclass(frozen=True)
class RolloutStageResult:
    """Result consumed by training workers."""

    iteration: int
    weight_version: int
    payload: Any
    started_at: float
    finished_at: float
    metrics: Mapping[str, Any] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float:
        return self.finished_at - self.started_at


@dataclass(frozen=True)
class TrainingStageResult:
    """Result produced by training workers."""

    iteration: int
    consumed_weight_version: int
    published_weight_version: Optional[int]
    metrics: Mapping[str, Any]
    started_at: float
    finished_at: float

    @property
    def duration_seconds(self) -> float:
        return self.finished_at - self.started_at


class TrainingWorker(Protocol):
    def train(self, rollout: RolloutStageResult) -> TrainingStageResult: ...


class RewardProvider(Protocol):
    """Callable reward source for grouped rollout candidates."""

    def __call__(
        self,
        candidate_groups: Sequence[Sequence[Sequence[int]]],
        rollout: RolloutStageResult,
    ) -> Sequence[Sequence[float]]: ...


@dataclass(frozen=True)
class TorchRLTrainingConfig:
    """Config shared by local and DeepSpeed training workers."""

    num_prompts: int = 1
    samples_per_prompt: int = 2
    prompt_len: int = 4
    completion_len: int = 8
    vocab_size: int = 64
    hidden_dim: int = 32
    valid_density: float = 0.75
    lr: float = 1e-3
    device: str = "cpu"
    dtype: torch.dtype = torch.float32
    seed: int = 0
    min_completion_len: int = 1
    clip_epsilon: float = 0.2
    kl_beta: float = 0.01
    advantage_eps: float = 1e-8
    loss_reduction: str = "global_token_mean"
    require_payload_rewards: bool = False
    require_payload_logps: bool = False
    reward_provider: Optional[RewardProvider] = None


class RolloutBatchMixin:
    config: TorchRLTrainingConfig
    device: torch.device

    def _batch_from_rollout_or_synthetic(
        self,
        rollout: RolloutStageResult,
    ) -> tuple[SyntheticRLKernelBatch, dict[str, Any]]:
        candidate_groups = extract_rollout_candidate_groups(rollout.payload)
        if candidate_groups:
            reward_groups, reward_source = _resolve_reward_groups(
                candidate_groups,
                rollout,
                self.config.reward_provider,
            )
            has_rewards = reward_groups is not None
            if self.config.require_payload_rewards and not has_rewards:
                raise ValueError(
                    "require_payload_rewards=True requires one scalar reward for every "
                    "rollout candidate"
                )
            old_logp_groups = extract_rollout_logp_groups(
                rollout.payload,
                keys=_OLD_LOGP_KEYS,
            )
            ref_logp_groups = extract_rollout_logp_groups(
                rollout.payload,
                keys=_REF_LOGP_KEYS,
            )
            has_payload_logps = _logp_groups_match(
                candidate_groups,
                old_logp_groups,
            ) and _logp_groups_match(candidate_groups, ref_logp_groups)
            if self.config.require_payload_logps and not has_payload_logps:
                raise ValueError(
                    "require_payload_logps=True requires old/ref selected logprobs "
                    "for every rollout candidate token"
                )
            batch = self._batch_from_candidate_groups(
                candidate_groups,
                rollout,
                reward_groups=reward_groups if has_rewards else None,
                reward_source=reward_source if has_rewards else "token_id_proxy",
                old_logp_groups=old_logp_groups if has_payload_logps else None,
                ref_logp_groups=ref_logp_groups if has_payload_logps else None,
            )
            token_groups = [tokens for group in candidate_groups for tokens in group]
            return batch, {
                "training_data_source": "rollout_payload",
                "reward_source": reward_source if has_rewards else "token_id_proxy",
                "logprob_source": "payload_logps" if has_payload_logps else "smoke_logps",
                "rollout_prompt_groups": len(candidate_groups),
                "rollout_sequences": len(token_groups),
                "rollout_tokens": sum(len(group) for group in token_groups),
            }

        if self.config.require_payload_rewards or self.config.require_payload_logps:
            raise ValueError(
                "strict production inputs require rollout payload candidate groups; "
                "strict mode requires rollout payload candidate groups before training"
            )

        seed = self.config.seed + int(rollout.iteration)
        batch = make_synthetic_rl_kernel_batch(
            num_prompts=self.config.num_prompts,
            samples_per_prompt=self.config.samples_per_prompt,
            prompt_len=self.config.prompt_len,
            completion_len=self.config.completion_len,
            vocab_size=self.config.vocab_size,
            valid_density=self.config.valid_density,
            dtype=self.config.dtype,
            device=self.device,
            seed=seed,
        )
        return batch, {
            "training_data_source": "synthetic_fallback",
            "rollout_sequences": 0,
            "rollout_tokens": 0,
            "reward_source": "synthetic_rewards",
            "logprob_source": "synthetic_logps",
        }

    def _batch_from_token_groups(
        self,
        token_groups: Sequence[Sequence[int]],
        rollout: RolloutStageResult,
    ) -> SyntheticRLKernelBatch:
        return self._batch_from_candidate_groups(
            [[group] for group in token_groups],
            rollout,
            reward_groups=None,
            reward_source="token_id_proxy",
        )

    def _batch_from_candidate_groups(
        self,
        candidate_groups: Sequence[Sequence[Sequence[int]]],
        rollout: RolloutStageResult,
        *,
        reward_groups: Optional[Sequence[Sequence[float]]] = None,
        reward_source: str = "token_id_proxy",
        old_logp_groups: Optional[Sequence[Sequence[Sequence[float]]]] = None,
        ref_logp_groups: Optional[Sequence[Sequence[Sequence[float]]]] = None,
    ) -> SyntheticRLKernelBatch:
        flat_token_groups = [tokens for group in candidate_groups for tokens in group]
        completion_len = max(
            self.config.min_completion_len,
            min(self.config.completion_len, max(len(group) for group in flat_token_groups)),
        )
        batch_size = len(flat_token_groups)
        token_ids = torch.zeros(
            (batch_size, completion_len),
            device=self.device,
            dtype=torch.long,
        )
        completion_mask = torch.zeros(
            (batch_size, completion_len),
            device=self.device,
            dtype=torch.bool,
        )
        flat_rewards: list[float] = []
        flat_group_ids: list[int] = []
        row = 0
        for group_index, group in enumerate(candidate_groups):
            for candidate_index, candidate_tokens in enumerate(group):
                clipped = [
                    int(token) % self.config.vocab_size
                    for token in candidate_tokens[:completion_len]
                ]
                if clipped:
                    values = torch.tensor(clipped, device=self.device, dtype=torch.long)
                    token_ids[row, : values.numel()] = values
                    completion_mask[row, : values.numel()] = True
                flat_rewards.append(
                    _candidate_reward_value(
                        candidate_tokens,
                        reward_groups,
                        group_index,
                        candidate_index,
                        vocab_size=self.config.vocab_size,
                    )
                )
                flat_group_ids.append(group_index)
                row += 1

        if not bool(completion_mask.any().item()):
            completion_mask[:, :1] = True

        prompt_tokens = torch.zeros(
            (batch_size, self.config.prompt_len),
            device=self.device,
            dtype=torch.long,
        )
        input_ids = torch.cat([prompt_tokens, token_ids], dim=1)
        prompt_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        if self.config.prompt_len:
            prompt_mask[:, : self.config.prompt_len] = True
        attention_mask = torch.cat(
            [
                prompt_mask[:, : self.config.prompt_len],
                completion_mask,
            ],
            dim=1,
        )

        rewards = torch.tensor(flat_rewards, device=self.device, dtype=self.config.dtype)
        group_ids = torch.tensor(flat_group_ids, device=self.device, dtype=torch.long)
        sequence_advantages = compute_group_relative_advantages(
            rewards,
            group_ids=group_ids,
            num_groups=len(candidate_groups),
            eps=self.config.advantage_eps,
        )
        advantages = broadcast_sequence_advantages(sequence_advantages, completion_mask).to(
            dtype=self.config.dtype
        )
        old_logps = torch.zeros(
            (batch_size, completion_len),
            device=self.device,
            dtype=self.config.dtype,
        )
        ref_logps = torch.zeros(
            (batch_size, completion_len),
            device=self.device,
            dtype=self.config.dtype,
        )
        if old_logp_groups is not None and ref_logp_groups is not None:
            row = 0
            for group_index, group in enumerate(candidate_groups):
                for candidate_index, candidate_tokens in enumerate(group):
                    token_count = min(len(candidate_tokens), completion_len)
                    if token_count:
                        old_values = torch.tensor(
                            old_logp_groups[group_index][candidate_index][:token_count],
                            device=self.device,
                            dtype=self.config.dtype,
                        )
                        ref_values = torch.tensor(
                            ref_logp_groups[group_index][candidate_index][:token_count],
                            device=self.device,
                            dtype=self.config.dtype,
                        )
                        old_logps[row, :token_count] = old_values
                        ref_logps[row, :token_count] = ref_values
                    row += 1
        valid_indices = completion_mask.reshape(-1).nonzero(as_tuple=False).squeeze(-1)
        metadata: dict[str, Any] = {
            "num_prompts": len(candidate_groups),
            "samples_per_prompt": max(len(group) for group in candidate_groups),
            "batch_size": batch_size,
            "prompt_len": self.config.prompt_len,
            "completion_len": completion_len,
            "total_seq_len": self.config.prompt_len + completion_len,
            "vocab_size": self.config.vocab_size,
            "valid_density": float(completion_mask.float().mean().item()),
            "valid_tokens": int(completion_mask.sum().item()),
            "dtype": self.config.dtype,
            "device": str(self.device),
            "seed": self.config.seed + int(rollout.iteration),
            "source": "rollout_payload",
            "group_ids": flat_group_ids,
            "reward_source": reward_source if reward_groups is not None else "token_id_proxy",
            "logprob_source": (
                "payload_logps"
                if old_logp_groups is not None and ref_logp_groups is not None
                else "smoke_logps"
            ),
        }
        return SyntheticRLKernelBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            prompt_mask=prompt_mask,
            completion_mask=completion_mask,
            token_ids=token_ids,
            rewards=rewards,
            advantages=advantages,
            old_logps=old_logps,
            ref_logps=ref_logps,
            valid_indices=valid_indices,
            metadata=metadata,
        )


def make_rollout_result(
    *,
    iteration: int,
    weight_version: int,
    payload: Any,
    metrics: Optional[Mapping[str, Any]] = None,
) -> RolloutStageResult:
    now = time.perf_counter()
    return RolloutStageResult(
        iteration=iteration,
        weight_version=weight_version,
        payload=payload,
        started_at=now,
        finished_at=time.perf_counter(),
        metrics=dict(metrics or {}),
    )


def extract_rollout_token_groups(payload: Any) -> list[list[int]]:
    """Extract generated token ids from RL-Kernel/vLLM-style rollout payloads."""

    return [tokens for group in extract_rollout_candidate_groups(payload) for tokens in group]


def extract_rollout_candidate_groups(payload: Any) -> list[list[list[int]]]:
    """Extract generated token ids while preserving prompt-level candidate groups."""

    if not isinstance(payload, Mapping):
        return []

    normalized_outputs = payload.get("normalized_outputs")
    if isinstance(normalized_outputs, Sequence) and not isinstance(
        normalized_outputs, (str, bytes)
    ):
        groups = _candidate_groups_from_grouped_outputs(normalized_outputs)
        if groups:
            return groups

    outputs = payload.get("outputs")
    if isinstance(outputs, Sequence) and not isinstance(outputs, (str, bytes)):
        return _candidate_groups_from_grouped_outputs(outputs)
    return []


def extract_rollout_reward_groups(payload: Any) -> list[list[float]]:
    """Extract scalar reward groups from rollout payloads when they are present."""

    if not isinstance(payload, Mapping):
        return []

    normalized_outputs = payload.get("normalized_outputs")
    if isinstance(normalized_outputs, Sequence) and not isinstance(
        normalized_outputs, (str, bytes)
    ):
        groups = _reward_groups_from_grouped_outputs(normalized_outputs)
        if groups:
            return groups

    outputs = payload.get("outputs")
    if isinstance(outputs, Sequence) and not isinstance(outputs, (str, bytes)):
        return _reward_groups_from_grouped_outputs(outputs)
    return []


def extract_rollout_logp_groups(payload: Any, *, keys: Sequence[str]) -> list[list[list[float]]]:
    """Extract selected token logprobs from grouped rollout payloads."""

    if not isinstance(payload, Mapping):
        return []

    normalized_outputs = payload.get("normalized_outputs")
    if isinstance(normalized_outputs, Sequence) and not isinstance(
        normalized_outputs, (str, bytes)
    ):
        groups = _logp_groups_from_grouped_outputs(normalized_outputs, keys=keys)
        if groups:
            return groups

    outputs = payload.get("outputs")
    if isinstance(outputs, Sequence) and not isinstance(outputs, (str, bytes)):
        return _logp_groups_from_grouped_outputs(outputs, keys=keys)
    return []


def compute_training_grpo_objective(
    logits: torch.Tensor,
    batch: SyntheticRLKernelBatch,
    config: TorchRLTrainingConfig,
):
    current_logps = selected_logprobs_reference(
        logits,
        batch.token_ids,
        mask=batch.completion_mask,
        output_dtype=torch.float32,
    )
    if batch.metadata.get("logprob_source") == "payload_logps":
        old_logps = batch.old_logps.detach().to(device=current_logps.device, dtype=torch.float32)
        ref_logps = batch.ref_logps.detach().to(device=current_logps.device, dtype=torch.float32)
        logprob_source = "payload_logps"
    else:
        old_logps = current_logps.detach() - 0.01
        ref_logps = current_logps.detach() - 0.02
        logprob_source = str(batch.metadata.get("logprob_source", "smoke_logps"))

    return compute_grpo_loss(
        current_logps=current_logps,
        old_logps=old_logps,
        ref_logps=ref_logps,
        rewards=batch.rewards,
        completion_mask=batch.completion_mask,
        group_ids=batch_group_ids(batch),
        num_groups=int(batch.metadata.get("num_prompts", 1)),
        config=GRPOConfig(
            clip_epsilon=config.clip_epsilon,
            kl_beta=config.kl_beta,
            advantage_eps=config.advantage_eps,
            loss_reduction=config.loss_reduction,
        ),
    ).with_metrics({"logprob_source": logprob_source})


def batch_group_ids(batch: SyntheticRLKernelBatch) -> Optional[torch.Tensor]:
    group_ids = batch.metadata.get("group_ids")
    if group_ids is None:
        return None
    if isinstance(group_ids, torch.Tensor):
        return group_ids.to(device=batch.rewards.device, dtype=torch.long)
    return torch.tensor(group_ids, device=batch.rewards.device, dtype=torch.long)


def _candidate_groups_from_grouped_outputs(grouped_outputs: Sequence[Any]) -> list[list[list[int]]]:
    groups: list[list[list[int]]] = []
    for group in grouped_outputs:
        if isinstance(group, Sequence) and not isinstance(group, (str, bytes, Mapping)):
            candidates = group
        else:
            candidates = [group]
        candidate_group = []
        for candidate in candidates:
            token_ids = _candidate_token_ids(candidate)
            if token_ids is not None:
                candidate_group.append(token_ids)
        if candidate_group:
            groups.append(candidate_group)
    return groups


def _reward_groups_from_grouped_outputs(grouped_outputs: Sequence[Any]) -> list[list[float]]:
    groups: list[list[float]] = []
    for group in grouped_outputs:
        if isinstance(group, Sequence) and not isinstance(group, (str, bytes, Mapping)):
            candidates = group
        else:
            candidates = [group]
        reward_group = []
        for candidate in candidates:
            reward = _candidate_reward(candidate)
            if reward is not None:
                reward_group.append(reward)
        if reward_group:
            groups.append(reward_group)
    return groups


def _logp_groups_from_grouped_outputs(
    grouped_outputs: Sequence[Any],
    *,
    keys: Sequence[str],
) -> list[list[list[float]]]:
    groups: list[list[list[float]]] = []
    for group in grouped_outputs:
        if isinstance(group, Sequence) and not isinstance(group, (str, bytes, Mapping)):
            candidates = group
        else:
            candidates = [group]
        logp_group = []
        for candidate in candidates:
            logps = _candidate_logps(candidate, keys=keys)
            if logps is not None:
                logp_group.append(logps)
        if logp_group:
            groups.append(logp_group)
    return groups


def _candidate_token_ids(candidate: Any) -> Optional[list[int]]:
    if candidate is None:
        return None
    if isinstance(candidate, Mapping):
        nested_outputs = candidate.get("outputs")
        if isinstance(nested_outputs, Sequence) and not isinstance(nested_outputs, (str, bytes)):
            for nested in nested_outputs:
                token_ids = _candidate_token_ids(nested)
                if token_ids is not None:
                    return token_ids
        if "token_ids" in candidate:
            return _copy_int_list(candidate.get("token_ids"))
        return None

    value = getattr(candidate, "token_ids", None)
    if value is not None:
        return _copy_int_list(value)
    nested_outputs = getattr(candidate, "outputs", None)
    if isinstance(nested_outputs, Sequence) and not isinstance(nested_outputs, (str, bytes)):
        for nested in nested_outputs:
            token_ids = _candidate_token_ids(nested)
            if token_ids is not None:
                return token_ids
    return None


def _candidate_logps(candidate: Any, *, keys: Sequence[str]) -> Optional[list[float]]:
    if candidate is None:
        return None
    if isinstance(candidate, Mapping):
        nested_outputs = candidate.get("outputs")
        if isinstance(nested_outputs, Sequence) and not isinstance(nested_outputs, (str, bytes)):
            for nested in nested_outputs:
                logps = _candidate_logps(nested, keys=keys)
                if logps is not None:
                    return logps
        for key in keys:
            if key in candidate:
                return _copy_float_list(candidate[key])
        return None

    nested_outputs = getattr(candidate, "outputs", None)
    if isinstance(nested_outputs, Sequence) and not isinstance(nested_outputs, (str, bytes)):
        for nested in nested_outputs:
            logps = _candidate_logps(nested, keys=keys)
            if logps is not None:
                return logps
    for key in keys:
        if hasattr(candidate, key):
            return _copy_float_list(getattr(candidate, key))
    return None


def _candidate_reward(candidate: Any) -> Optional[float]:
    if candidate is None:
        return None
    if isinstance(candidate, Mapping):
        nested_outputs = candidate.get("outputs")
        if isinstance(nested_outputs, Sequence) and not isinstance(nested_outputs, (str, bytes)):
            for nested in nested_outputs:
                reward = _candidate_reward(nested)
                if reward is not None:
                    return reward
        for key in ("reward", "score", "scalar_reward", "reward_score"):
            if key in candidate:
                return _safe_float(candidate[key])
        return None

    nested_outputs = getattr(candidate, "outputs", None)
    if isinstance(nested_outputs, Sequence) and not isinstance(nested_outputs, (str, bytes)):
        for nested in nested_outputs:
            reward = _candidate_reward(nested)
            if reward is not None:
                return reward
    for attr in ("reward", "score", "scalar_reward", "reward_score"):
        if hasattr(candidate, attr):
            return _safe_float(getattr(candidate, attr))
    return None


def _resolve_reward_groups(
    candidate_groups: Sequence[Sequence[Sequence[int]]],
    rollout: RolloutStageResult,
    reward_provider: Optional[RewardProvider],
) -> tuple[Optional[Sequence[Sequence[float]]], str]:
    payload_reward_groups = extract_rollout_reward_groups(rollout.payload)
    if _reward_groups_match(candidate_groups, payload_reward_groups):
        return payload_reward_groups, "payload_rewards"

    if reward_provider is None:
        return None, "token_id_proxy"

    provided = reward_provider(candidate_groups, rollout)
    if not _reward_groups_match(candidate_groups, provided):
        raise ValueError(
            "reward_provider must return one scalar reward for every rollout candidate"
        )
    return provided, "provider_rewards"


def _safe_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        flat = value.detach().cpu().reshape(-1)
        if flat.numel() != 1:
            raise ValueError("rollout reward tensors must contain exactly one value")
        return float(flat[0].item())
    return float(value)


def _reward_groups_match(
    candidate_groups: Sequence[Sequence[Sequence[int]]],
    reward_groups: Sequence[Sequence[float]],
) -> bool:
    if len(candidate_groups) != len(reward_groups):
        return False
    return all(
        len(candidates) == len(rewards)
        for candidates, rewards in zip(candidate_groups, reward_groups, strict=False)
    )


def _logp_groups_match(
    candidate_groups: Sequence[Sequence[Sequence[int]]],
    logp_groups: Sequence[Sequence[Sequence[float]]],
) -> bool:
    if len(candidate_groups) != len(logp_groups):
        return False
    for candidate_group, logp_group in zip(candidate_groups, logp_groups, strict=False):
        if len(candidate_group) != len(logp_group):
            return False
        for token_ids, logps in zip(candidate_group, logp_group, strict=False):
            if len(token_ids) != len(logps):
                return False
    return True


def _candidate_reward_value(
    token_ids: Sequence[int],
    reward_groups: Optional[Sequence[Sequence[float]]],
    group_index: int,
    candidate_index: int,
    *,
    vocab_size: int,
) -> float:
    if reward_groups is not None:
        return float(reward_groups[group_index][candidate_index])
    if not token_ids:
        return 0.0
    clipped = [int(token) % vocab_size for token in token_ids]
    return float(sum(clipped)) / float(max(len(clipped), 1) * max(vocab_size - 1, 1))


def _copy_int_list(value: Any) -> list[int]:
    if isinstance(value, torch.Tensor):
        return [int(item) for item in value.detach().cpu().reshape(-1).tolist()]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [int(item) for item in value]
    return []


def _copy_float_list(value: Any) -> list[float]:
    if isinstance(value, torch.Tensor):
        return [float(item) for item in value.detach().cpu().reshape(-1).tolist()]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [float(item) for item in value]
    return []
