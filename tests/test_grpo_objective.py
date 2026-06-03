# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kernel-Align Contributors

from __future__ import annotations

import pytest
import torch

from rl_engine.alignment.grpo import (
    GRPOConfig,
    broadcast_sequence_advantages,
    compute_group_relative_advantages,
    compute_grpo_loss,
)


def test_group_relative_advantages_normalize_each_prompt_group_independently():
    rewards = torch.tensor([1.0, 3.0, 100.0, 104.0])
    group_ids = torch.tensor([0, 0, 1, 1])

    advantages = compute_group_relative_advantages(rewards, group_ids=group_ids)

    assert torch.allclose(advantages, torch.tensor([-1.0, 1.0, -1.0, 1.0]))
    assert torch.allclose(advantages[group_ids == 0].mean(), torch.tensor(0.0))
    assert torch.allclose(advantages[group_ids == 1].mean(), torch.tensor(0.0))


def test_group_relative_advantages_can_infer_contiguous_groups():
    rewards = torch.tensor([1.0, 3.0, 10.0, 14.0])

    advantages = compute_group_relative_advantages(rewards, num_groups=2)

    assert torch.allclose(advantages, torch.tensor([-1.0, 1.0, -1.0, 1.0]))


def test_equal_reward_and_single_sample_groups_are_finite_and_centered():
    rewards = torch.tensor([5.0, 5.0, 2.0])
    group_ids = torch.tensor([0, 0, 1])

    advantages = compute_group_relative_advantages(rewards, group_ids=group_ids, num_groups=2)

    assert torch.isfinite(advantages).all()
    assert torch.equal(advantages, torch.zeros_like(advantages))


def test_broadcast_sequence_advantages_masks_inactive_completion_tokens():
    sequence_advantages = torch.tensor([2.0, -1.0])
    completion_mask = torch.tensor([[True, False, True], [False, True, True]])

    token_advantages = broadcast_sequence_advantages(sequence_advantages, completion_mask)

    assert torch.equal(
        token_advantages,
        torch.tensor([[2.0, 0.0, 2.0], [0.0, -1.0, -1.0]]),
    )


def test_grpo_loss_matches_direct_pytorch_reference_and_masks_inactive_tokens():
    current = torch.tensor(
        [
            [0.10, 0.20, 0.30],
            [0.00, -0.10, -0.20],
            [0.40, 0.30, 0.20],
            [-0.30, -0.20, -0.10],
        ],
        dtype=torch.float32,
    )
    old = current.detach() - 0.05
    ref = current.detach() - 0.02
    rewards = torch.tensor([1.0, 3.0, 10.0, 14.0])
    group_ids = torch.tensor([0, 0, 1, 1])
    mask = torch.tensor(
        [
            [True, True, False],
            [True, False, False],
            [False, True, True],
            [True, True, True],
        ]
    )
    config = GRPOConfig(clip_epsilon=0.2, kl_beta=0.03)

    result = compute_grpo_loss(
        current,
        old,
        ref,
        rewards,
        mask,
        group_ids=group_ids,
        config=config,
    )

    sequence_advantages = torch.tensor([-1.0, 1.0, -1.0, 1.0])
    advantages = sequence_advantages.unsqueeze(-1).expand_as(current).masked_fill(~mask, 0.0)
    ratio = torch.exp(current - old).masked_fill(~mask, 0.0)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 0.8, 1.2) * advantages
    policy_loss = -torch.minimum(unclipped, clipped).masked_fill(~mask, 0.0)
    diff = ref - current
    kl = (torch.exp(diff) - diff - 1.0).masked_fill(~mask, 0.0)
    expected = (policy_loss + config.kl_beta * kl).sum() / mask.sum()

    assert torch.allclose(result.sequence_advantages, sequence_advantages)
    assert torch.equal(result.advantages[~mask], torch.zeros_like(result.advantages[~mask]))
    assert torch.allclose(result.ratio[~mask], torch.zeros_like(result.ratio[~mask]))
    assert torch.allclose(result.kl[~mask], torch.zeros_like(result.kl[~mask]))
    assert torch.allclose(result.loss, expected)
    assert result.metrics["active_tokens"] == float(mask.sum().item())
    assert result.metrics["loss_reduction"] == "global_token_mean"


def test_grpo_loss_sequence_mean_is_explicit_and_differs_on_ragged_completions():
    current = torch.zeros((2, 3), dtype=torch.float32)
    old = torch.zeros_like(current)
    ref = torch.zeros_like(current)
    rewards = torch.tensor([1.0, 3.0])
    mask = torch.tensor(
        [
            [True, False, False],
            [True, True, True],
        ]
    )

    global_result = compute_grpo_loss(
        current,
        old,
        ref,
        rewards,
        mask,
        num_groups=1,
        config=GRPOConfig(kl_beta=0.0),
    )
    sequence_result = compute_grpo_loss(
        current,
        old,
        ref,
        rewards,
        mask,
        num_groups=1,
        config=GRPOConfig(kl_beta=0.0, loss_reduction="sequence_mean"),
    )

    assert torch.allclose(global_result.loss, torch.tensor(-0.5))
    assert torch.allclose(global_result.policy_loss, torch.tensor(-0.5))
    assert torch.allclose(sequence_result.loss, torch.tensor(0.0))
    assert torch.allclose(sequence_result.policy_loss, torch.tensor(0.0))
    assert global_result.metrics["loss_reduction"] == "global_token_mean"
    assert sequence_result.metrics["loss_reduction"] == "sequence_mean"
    assert global_result.metrics["active_sequences"] == 2.0
    assert sequence_result.metrics["active_tokens"] == 4.0


def test_grpo_config_rejects_unknown_loss_reduction():
    with pytest.raises(ValueError, match="loss_reduction"):
        GRPOConfig(loss_reduction="batch_mean")


def test_grpo_loss_only_backpropagates_through_current_logps():
    current = torch.tensor(
        [[0.10, 0.20], [0.05, -0.15], [0.30, 0.40], [-0.20, -0.10]],
        requires_grad=True,
    )
    old = torch.zeros_like(current, requires_grad=True)
    ref = torch.zeros_like(current, requires_grad=True)
    rewards = torch.tensor([1.0, 3.0, 2.0, 6.0], requires_grad=True)
    mask = torch.ones_like(current, dtype=torch.bool)

    result = compute_grpo_loss(current, old, ref, rewards, mask, num_groups=2)
    result.loss.backward()

    assert current.grad is not None
    assert current.grad.abs().sum() > 0
    assert old.grad is None
    assert ref.grad is None
    assert rewards.grad is None


def test_grpo_loss_rejects_shape_mismatches_and_empty_masks():
    current = torch.zeros(2, 3)
    old = torch.zeros(2, 3)
    ref = torch.zeros(2, 3)
    rewards = torch.ones(2)
    mask = torch.ones(2, 3, dtype=torch.bool)

    with pytest.raises(ValueError, match="old_logps shape"):
        compute_grpo_loss(current, torch.zeros(2, 2), ref, rewards, mask, num_groups=1)

    with pytest.raises(ValueError, match="completion_mask must contain"):
        compute_grpo_loss(current, old, ref, rewards, torch.zeros_like(mask), num_groups=1)

    with pytest.raises(ValueError, match="divisible by num_groups"):
        compute_grpo_loss(current, old, ref, rewards, mask, num_groups=3)


def test_group_relative_advantages_reject_invalid_group_ids():
    rewards = torch.ones(3)

    with pytest.raises(ValueError, match="non-negative"):
        compute_group_relative_advantages(rewards, group_ids=torch.tensor([0, -1, 1]))

    with pytest.raises(ValueError, match="has no rewards"):
        compute_group_relative_advantages(
            rewards,
            group_ids=torch.tensor([0, 0, 2]),
            num_groups=3,
        )
