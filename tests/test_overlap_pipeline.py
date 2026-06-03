# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest
import torch

from rl_engine.executors.overlap_pipeline import (
    IterationSpec,
    ManifestWeightHandoff,
    OverlapPipeline,
    PipelineConfig,
    PipelineExecutionError,
    RolloutExecutorWorker,
    RolloutStageResult,
    TorchRLTrainingConfig,
    TorchRLTrainingWorker,
    TrainingStageResult,
    extract_rollout_candidate_groups,
    extract_rollout_token_groups,
)
from rl_engine.executors.training_contract import (
    compute_training_grpo_objective,
    extract_rollout_logp_groups,
)


class RecordingRolloutWorker:
    def __init__(self, *, delay: float = 0.0, fail_iteration: int | None = None):
        self.delay = delay
        self.fail_iteration = fail_iteration
        self.started: dict[int, threading.Event] = {}
        self.finished: dict[int, threading.Event] = {}
        self.calls: list[int] = []
        self._lock = threading.Lock()
        self._in_flight = 0
        self.max_in_flight = 0

    def rollout(self, spec: IterationSpec) -> RolloutStageResult:
        with self._lock:
            self.calls.append(spec.iteration)
            self._in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self._in_flight)
            started = self.started.setdefault(spec.iteration, threading.Event())
            finished = self.finished.setdefault(spec.iteration, threading.Event())

        started_at = time.perf_counter()
        started.set()
        try:
            if self.fail_iteration == spec.iteration:
                raise RuntimeError(f"rollout boom {spec.iteration}")
            if self.delay:
                time.sleep(self.delay)
            return RolloutStageResult(
                iteration=spec.iteration,
                weight_version=int(spec.weight_version),
                payload={"prompts": list(spec.prompts)},
                started_at=started_at,
                finished_at=time.perf_counter(),
                metrics={"worker": "recording"},
            )
        finally:
            finished.set()
            with self._lock:
                self._in_flight -= 1


class RecordingTrainingWorker:
    def __init__(
        self,
        *,
        delay: float = 0.0,
        block_iteration: int | None = None,
        fail_iteration: int | None = None,
    ):
        self.delay = delay
        self.block_iteration = block_iteration
        self.fail_iteration = fail_iteration
        self.allow_finish = threading.Event()
        self.started: dict[int, threading.Event] = {}
        self.finished: dict[int, threading.Event] = {}
        self.calls: list[int] = []
        self.consumed_versions: list[int] = []

    def train(self, rollout: RolloutStageResult) -> TrainingStageResult:
        self.calls.append(rollout.iteration)
        self.consumed_versions.append(rollout.weight_version)
        started = self.started.setdefault(rollout.iteration, threading.Event())
        finished = self.finished.setdefault(rollout.iteration, threading.Event())
        started_at = time.perf_counter()
        started.set()
        try:
            if self.fail_iteration == rollout.iteration:
                raise RuntimeError(f"training boom {rollout.iteration}")
            if self.block_iteration == rollout.iteration:
                assert self.allow_finish.wait(timeout=5.0)
            if self.delay:
                time.sleep(self.delay)
            return TrainingStageResult(
                iteration=rollout.iteration,
                consumed_weight_version=rollout.weight_version,
                published_weight_version=rollout.weight_version + 1,
                metrics={"worker": "recording"},
                started_at=started_at,
                finished_at=time.perf_counter(),
            )
        finally:
            finished.set()


def _specs(count: int) -> list[IterationSpec]:
    return [
        IterationSpec(iteration=index, weight_version=index, prompts=[f"prompt-{index}"])
        for index in range(count)
    ]


def _unversioned_specs(count: int) -> list[IterationSpec]:
    return [IterationSpec(iteration=index, prompts=[f"prompt-{index}"]) for index in range(count)]


def test_pipeline_overlaps_next_rollout_with_current_training():
    rollout = RecordingRolloutWorker(delay=0.01)
    training = RecordingTrainingWorker(block_iteration=0)
    pipeline = OverlapPipeline(
        rollout,
        training,
        PipelineConfig(max_prefetch=1, rollout_workers=1, training_workers=1),
    )

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(pipeline.run, _specs(3))

        assert training.started.setdefault(0, threading.Event()).wait(timeout=5.0)
        assert rollout.started.setdefault(1, threading.Event()).wait(timeout=5.0)
        assert not training.finished.setdefault(0, threading.Event()).is_set()

        training.allow_finish.set()
        results = future.result(timeout=5.0)

    assert [result.iteration for result in results] == [0, 1, 2]
    assert [result.consumed_weight_version for result in results] == [0, 0, 1]
    assert [result.published_weight_version for result in results] == [1, 1, 2]

    summary = pipeline.timeline_summary()
    assert summary.overlap_seconds > 0.0
    assert summary.overlap_ratio > 0.0


def test_pipeline_backpressure_limits_rollout_prefetch():
    rollout = RecordingRolloutWorker(delay=0.02)
    training = RecordingTrainingWorker(delay=0.02)
    pipeline = OverlapPipeline(
        rollout,
        training,
        PipelineConfig(max_prefetch=1, rollout_workers=4, training_workers=1),
    )

    pipeline.run(_specs(4))

    assert rollout.max_in_flight == 1
    assert pipeline.max_queue_depth <= 1
    assert training.calls == [0, 1, 2, 3]


def test_pipeline_defaults_to_current_published_weight_version():
    rollout = RecordingRolloutWorker(delay=0.01)
    training = RecordingTrainingWorker(delay=0.01)
    pipeline = OverlapPipeline(
        rollout,
        training,
        PipelineConfig(max_prefetch=1, initial_weight_version=5),
    )

    results = pipeline.run(_unversioned_specs(3))

    assert [result.consumed_weight_version for result in results] == [5, 5, 6]
    assert [result.published_weight_version for result in results] == [6, 6, 7]
    assert pipeline.timeline_summary().final_published_weight_version == 7


def test_torch_training_worker_publishes_monotonic_versions_under_stale_rollout():
    worker = TorchRLTrainingWorker(
        TorchRLTrainingConfig(
            num_prompts=1,
            samples_per_prompt=1,
            prompt_len=1,
            completion_len=2,
            vocab_size=16,
            hidden_dim=8,
            valid_density=1.0,
            seed=3,
        )
    )
    rollout_a = RolloutStageResult(
        iteration=0,
        weight_version=5,
        payload={"normalized_outputs": [[{"token_ids": [1, 2]}]]},
        started_at=time.perf_counter(),
        finished_at=time.perf_counter(),
    )
    rollout_b = RolloutStageResult(
        iteration=1,
        weight_version=5,
        payload={"normalized_outputs": [[{"token_ids": [3, 4]}]]},
        started_at=time.perf_counter(),
        finished_at=time.perf_counter(),
    )

    first = worker.train(rollout_a)
    second = worker.train(rollout_b)

    assert first.published_weight_version == 6
    assert second.published_weight_version == 7


class ManifestAwareRolloutWorker(RecordingRolloutWorker):
    def __init__(self, *, delay: float = 0.0):
        super().__init__(delay=delay)
        self.installed_versions: list[int] = []
        self.installed_transports: list[str] = []
        self.released_update_ids: list[str] = []

    def install_weight_manifest(self, manifest):
        self.installed_versions.append(manifest.weight_version)
        self.installed_transports.append(manifest.transport)

    def release_weight_manifest(self, update_id):
        self.released_update_ids.append(update_id)


class ManifestPublishingTrainingWorker(RecordingTrainingWorker):
    def __init__(self, *, delay: float = 0.0):
        super().__init__(delay=delay)
        self.published_versions: list[int] = []
        self.released_update_ids: list[str] = []
        self._latest_published_weight_version = -1
        self._bridges = {}

    def publish_weights(self, *, weight_version, metadata=None):
        import torch

        from rl_engine.executors.bridge import LocalTensorCopyBridge

        version = max(self._latest_published_weight_version + 1, int(weight_version))
        self._latest_published_weight_version = version
        self.published_versions.append(version)
        model = torch.nn.Linear(2, 2)
        bridge = LocalTensorCopyBridge(source_worker="test-training")
        manifest = bridge.publish(model, weight_version=version, metadata=metadata)
        self._bridges[manifest.update_id] = bridge
        return manifest

    def release_weights(self, update_id):
        self.released_update_ids.append(update_id)
        self._bridges.pop(update_id).release(update_id)


def test_pipeline_manifest_weight_handoff_installs_complete_published_updates():
    rollout = ManifestAwareRolloutWorker(delay=0.01)
    training = ManifestPublishingTrainingWorker(delay=0.01)
    pipeline = OverlapPipeline(
        rollout,
        training,
        PipelineConfig(max_prefetch=1, initial_weight_version=0),
        weight_handoff=ManifestWeightHandoff(),
    )

    results = pipeline.run(_unversioned_specs(3))
    summary = pipeline.timeline_summary()

    assert [result.consumed_weight_version for result in results] == [0, 0, 1]
    assert [result.published_weight_version for result in results] == [1, 1, 2]
    assert training.published_versions == [1, 2, 3]
    assert rollout.installed_versions == [1, 2, 3]
    assert rollout.installed_transports == ["local-clone", "local-clone", "local-clone"]
    assert len(summary.weight_handoffs) == 3
    assert summary.weight_handoffs[-1]["weight_version"] == 3
    assert len(rollout.released_update_ids) == 3
    assert len(training.released_update_ids) == 3


def test_pipeline_surfaces_rollout_failure():
    rollout = RecordingRolloutWorker(fail_iteration=0)
    training = RecordingTrainingWorker()
    pipeline = OverlapPipeline(rollout, training)

    with pytest.raises(PipelineExecutionError) as exc_info:
        pipeline.run(_specs(2))

    assert exc_info.value.stage == "rollout"
    assert exc_info.value.iteration == 0
    assert training.calls == []


def test_pipeline_surfaces_training_failure_without_publishing_next_result():
    rollout = RecordingRolloutWorker()
    training = RecordingTrainingWorker(fail_iteration=0)
    pipeline = OverlapPipeline(rollout, training)

    with pytest.raises(PipelineExecutionError) as exc_info:
        pipeline.run(_specs(2))

    assert exc_info.value.stage == "training"
    assert exc_info.value.iteration == 0
    assert pipeline.training_results == []


class FakeRolloutExecutor:
    def __init__(self):
        self.calls = []

    def generate_candidates(self, prompts, *, num_generations=None, sampling_params=None):
        self.calls.append((prompts, num_generations, sampling_params))
        return {
            "backend": "fake-vllm",
            "num_prompts": len(prompts),
            "num_generations": num_generations,
        }


def test_rollout_executor_worker_uses_generate_candidates():
    executor = FakeRolloutExecutor()
    worker = RolloutExecutorWorker(
        executor,
        num_generations=2,
        sampling_params={"max_tokens": 8},
    )
    result = worker.rollout(IterationSpec(iteration=3, weight_version=7, prompts=["a", "b"]))

    assert executor.calls == [(["a", "b"], 2, {"max_tokens": 8})]
    assert result.iteration == 3
    assert result.weight_version == 7
    assert result.payload["backend"] == "fake-vllm"
    assert result.metrics["num_prompts"] == 2


def test_torch_rl_training_worker_runs_real_optimizer_step():
    worker = TorchRLTrainingWorker(
        TorchRLTrainingConfig(
            num_prompts=1,
            samples_per_prompt=2,
            prompt_len=2,
            completion_len=3,
            vocab_size=16,
            hidden_dim=8,
            valid_density=1.0,
            seed=5,
        )
    )
    rollout = RolloutStageResult(
        iteration=2,
        weight_version=9,
        payload={
            "normalized_outputs": [
                [
                    {
                        "token_ids": [3, 4, 5],
                        "text": "abc",
                    }
                ],
                [
                    {
                        "token_ids": [6, 7, 8],
                        "text": "def",
                    }
                ],
            ]
        },
        started_at=time.perf_counter(),
        finished_at=time.perf_counter(),
    )

    result = worker.train(rollout)

    assert result.iteration == 2
    assert result.consumed_weight_version == 9
    assert result.published_weight_version == 10
    assert result.metrics["training_backend"] == "torch"
    assert result.metrics["training_data_source"] == "rollout_payload"
    assert result.metrics["rollout_sequences"] == 2
    assert result.metrics["rollout_tokens"] == 6
    assert result.metrics["reward_source"] == "token_id_proxy"
    assert math.isfinite(result.metrics["loss"])
    assert result.metrics["active_tokens"] == 6
    assert result.metrics["objective"] == "grpo"


def test_torch_rl_training_worker_uses_payload_rewards_for_grpo_groups():
    worker = TorchRLTrainingWorker(
        TorchRLTrainingConfig(
            num_prompts=1,
            samples_per_prompt=2,
            prompt_len=1,
            completion_len=2,
            vocab_size=16,
            hidden_dim=8,
            valid_density=1.0,
            seed=7,
        )
    )
    rollout = RolloutStageResult(
        iteration=0,
        weight_version=4,
        payload={
            "normalized_outputs": [
                [
                    {"token_ids": [1, 2], "reward": 1.0},
                    {"token_ids": [3, 4], "reward": 3.0},
                ]
            ]
        },
        started_at=time.perf_counter(),
        finished_at=time.perf_counter(),
    )

    result = worker.train(rollout)

    assert result.metrics["training_data_source"] == "rollout_payload"
    assert result.metrics["reward_source"] == "payload_rewards"
    assert result.metrics["rollout_prompt_groups"] == 1
    assert result.metrics["rollout_sequences"] == 2
    assert result.metrics["advantage_mean"] == pytest.approx(0.0, abs=1e-6)
    assert result.metrics["advantage_std"] == pytest.approx(1.0, abs=1e-6)


def test_training_grpo_uses_payload_old_and_reference_logps_as_fixed_inputs():
    config = TorchRLTrainingConfig(
        num_prompts=1,
        samples_per_prompt=2,
        prompt_len=1,
        completion_len=2,
        vocab_size=16,
        hidden_dim=8,
        valid_density=1.0,
        seed=11,
        require_payload_rewards=True,
        require_payload_logps=True,
        kl_beta=0.4,
    )
    worker = TorchRLTrainingWorker(config)
    rollout = RolloutStageResult(
        iteration=0,
        weight_version=4,
        payload={
            "normalized_outputs": [
                [
                    {
                        "token_ids": [1, 2],
                        "reward": 1.0,
                        "old_logps": [-0.20, -0.30],
                        "reference_logprobs": [-0.40, -0.50],
                    },
                    {
                        "token_ids": [3, 4],
                        "reward": 3.0,
                        "old_selected_logprobs": [-1.20, -1.00],
                        "ref_selected_logps": [-0.80, -0.90],
                    },
                ]
            ]
        },
        started_at=time.perf_counter(),
        finished_at=time.perf_counter(),
    )

    batch, metrics = worker._batch_from_rollout_or_synthetic(rollout)
    logits = torch.zeros((2, 2, config.vocab_size), requires_grad=True)
    batch.old_logps.requires_grad_()
    batch.ref_logps.requires_grad_()

    payload_result = compute_training_grpo_objective(logits, batch, config)
    smoke_batch = replace(batch, metadata={**batch.metadata, "logprob_source": "smoke_logps"})
    smoke_result = compute_training_grpo_objective(logits.detach().clone(), smoke_batch, config)
    payload_result.loss.backward()

    assert metrics["reward_source"] == "payload_rewards"
    assert metrics["logprob_source"] == "payload_logps"
    assert batch.metadata["logprob_source"] == "payload_logps"
    assert torch.allclose(batch.old_logps, torch.tensor([[-0.20, -0.30], [-1.20, -1.00]]))
    assert torch.allclose(batch.ref_logps, torch.tensor([[-0.40, -0.50], [-0.80, -0.90]]))
    assert payload_result.metrics["logprob_source"] == "payload_logps"
    assert not torch.allclose(payload_result.loss.detach(), smoke_result.loss.detach())
    assert logits.grad is not None
    assert logits.grad.abs().sum() > 0
    assert batch.old_logps.grad is None
    assert batch.ref_logps.grad is None


def test_torch_rl_training_worker_strict_logps_reject_missing_or_mismatched_payload():
    worker = TorchRLTrainingWorker(
        TorchRLTrainingConfig(
            num_prompts=1,
            samples_per_prompt=2,
            prompt_len=1,
            completion_len=2,
            vocab_size=16,
            hidden_dim=8,
            valid_density=1.0,
            seed=12,
            require_payload_logps=True,
        )
    )
    missing_ref = RolloutStageResult(
        iteration=0,
        weight_version=4,
        payload={
            "normalized_outputs": [
                [
                    {"token_ids": [1], "old_logps": [-0.1]},
                    {"token_ids": [2], "old_logps": [-0.2]},
                ]
            ]
        },
        started_at=time.perf_counter(),
        finished_at=time.perf_counter(),
    )
    mismatched_lengths = RolloutStageResult(
        iteration=0,
        weight_version=4,
        payload={
            "normalized_outputs": [
                [
                    {
                        "token_ids": [1, 2],
                        "old_logps": [-0.1],
                        "ref_logps": [-0.2, -0.3],
                    },
                    {
                        "token_ids": [3],
                        "old_logps": [-0.4],
                        "ref_logps": [-0.5],
                    },
                ]
            ]
        },
        started_at=time.perf_counter(),
        finished_at=time.perf_counter(),
    )

    with pytest.raises(ValueError, match="old/ref selected logprobs"):
        worker.train(missing_ref)
    with pytest.raises(ValueError, match="old/ref selected logprobs"):
        worker.train(mismatched_lengths)


def test_torch_rl_training_worker_uses_reward_provider_when_payload_rewards_are_absent():
    calls = []

    def reward_provider(candidate_groups, rollout):
        calls.append((candidate_groups, rollout.iteration))
        return [[1.0, 5.0]]

    worker = TorchRLTrainingWorker(
        TorchRLTrainingConfig(
            num_prompts=1,
            samples_per_prompt=2,
            prompt_len=1,
            completion_len=2,
            vocab_size=16,
            hidden_dim=8,
            valid_density=1.0,
            seed=13,
            reward_provider=reward_provider,
            require_payload_rewards=True,
        )
    )
    rollout = RolloutStageResult(
        iteration=7,
        weight_version=4,
        payload={"normalized_outputs": [[{"token_ids": [1, 2]}, {"token_ids": [3, 4]}]]},
        started_at=time.perf_counter(),
        finished_at=time.perf_counter(),
    )

    batch, metrics = worker._batch_from_rollout_or_synthetic(rollout)

    assert calls == [([[[1, 2], [3, 4]]], 7)]
    assert metrics["reward_source"] == "provider_rewards"
    assert batch.metadata["reward_source"] == "provider_rewards"
    assert torch.allclose(batch.rewards, torch.tensor([1.0, 5.0]))


def test_torch_rl_training_worker_rejects_malformed_reward_provider_output():
    def reward_provider(candidate_groups, rollout):
        del candidate_groups, rollout
        return [[1.0]]

    worker = TorchRLTrainingWorker(
        TorchRLTrainingConfig(
            num_prompts=1,
            samples_per_prompt=2,
            prompt_len=1,
            completion_len=2,
            vocab_size=16,
            hidden_dim=8,
            valid_density=1.0,
            seed=14,
            reward_provider=reward_provider,
        )
    )
    rollout = RolloutStageResult(
        iteration=0,
        weight_version=4,
        payload={"normalized_outputs": [[{"token_ids": [1]}, {"token_ids": [2]}]]},
        started_at=time.perf_counter(),
        finished_at=time.perf_counter(),
    )

    with pytest.raises(ValueError, match="reward_provider must return one scalar reward"):
        worker.train(rollout)


def test_torch_rl_training_worker_preserves_empty_completions_with_payload_rewards():
    worker = TorchRLTrainingWorker(
        TorchRLTrainingConfig(
            num_prompts=1,
            samples_per_prompt=2,
            prompt_len=1,
            completion_len=2,
            vocab_size=16,
            hidden_dim=8,
            valid_density=1.0,
            seed=8,
            require_payload_rewards=True,
        )
    )
    rollout = RolloutStageResult(
        iteration=0,
        weight_version=4,
        payload={
            "normalized_outputs": [
                [
                    {"token_ids": [], "reward": 0.0},
                    {"token_ids": [5], "reward": 2.0},
                ]
            ]
        },
        started_at=time.perf_counter(),
        finished_at=time.perf_counter(),
    )

    result = worker.train(rollout)

    assert result.metrics["training_data_source"] == "rollout_payload"
    assert result.metrics["reward_source"] == "payload_rewards"
    assert result.metrics["rollout_prompt_groups"] == 1
    assert result.metrics["rollout_sequences"] == 2
    assert result.metrics["rollout_tokens"] == 1
    assert result.metrics["active_tokens"] == 1
    assert result.metrics["advantage_mean"] == pytest.approx(0.0, abs=1e-6)
    assert result.metrics["advantage_std"] == pytest.approx(1.0, abs=1e-6)


def test_torch_rl_training_worker_accepts_empty_completion_payload_logps():
    worker = TorchRLTrainingWorker(
        TorchRLTrainingConfig(
            num_prompts=1,
            samples_per_prompt=2,
            prompt_len=1,
            completion_len=2,
            vocab_size=16,
            hidden_dim=8,
            valid_density=1.0,
            seed=15,
            require_payload_rewards=True,
            require_payload_logps=True,
        )
    )
    rollout = RolloutStageResult(
        iteration=0,
        weight_version=4,
        payload={
            "normalized_outputs": [
                [
                    {"token_ids": [], "reward": 0.0, "old_logps": [], "ref_logps": []},
                    {
                        "token_ids": [5],
                        "reward": 2.0,
                        "old_logps": [-0.4],
                        "ref_logps": [-0.6],
                    },
                ]
            ]
        },
        started_at=time.perf_counter(),
        finished_at=time.perf_counter(),
    )

    result = worker.train(rollout)

    assert result.metrics["reward_source"] == "payload_rewards"
    assert result.metrics["logprob_source"] == "payload_logps"
    assert result.metrics["rollout_tokens"] == 1
    assert result.metrics["active_tokens"] == 1


def test_torch_rl_training_worker_strict_rewards_reject_missing_rewards():
    worker = TorchRLTrainingWorker(
        TorchRLTrainingConfig(
            num_prompts=1,
            samples_per_prompt=2,
            prompt_len=1,
            completion_len=2,
            vocab_size=16,
            hidden_dim=8,
            valid_density=1.0,
            seed=9,
            require_payload_rewards=True,
        )
    )
    rollout = RolloutStageResult(
        iteration=0,
        weight_version=4,
        payload={
            "normalized_outputs": [
                [
                    {"token_ids": [1], "reward": 1.0},
                    {"token_ids": [2]},
                ]
            ]
        },
        started_at=time.perf_counter(),
        finished_at=time.perf_counter(),
    )

    with pytest.raises(ValueError, match="requires one scalar reward"):
        worker.train(rollout)


def test_torch_rl_training_worker_strict_rewards_reject_synthetic_fallback():
    worker = TorchRLTrainingWorker(
        TorchRLTrainingConfig(
            num_prompts=1,
            samples_per_prompt=2,
            prompt_len=1,
            completion_len=2,
            vocab_size=16,
            hidden_dim=8,
            valid_density=1.0,
            seed=10,
            require_payload_rewards=True,
        )
    )
    rollout = RolloutStageResult(
        iteration=0,
        weight_version=4,
        payload={"prompts": ["no candidates"]},
        started_at=time.perf_counter(),
        finished_at=time.perf_counter(),
    )

    with pytest.raises(ValueError, match="requires rollout payload candidate groups"):
        worker.train(rollout)


def test_extract_rollout_token_groups_from_normalized_payload():
    payload = {
        "normalized_outputs": [
            [{"token_ids": [1, 2]}, {"token_ids": [3]}],
            [{"outputs": [{"token_ids": [4, 5, 6]}]}],
        ]
    }

    assert extract_rollout_token_groups(payload) == [[1, 2], [3], [4, 5, 6]]


def test_extract_rollout_candidate_groups_preserves_prompt_boundaries():
    payload = {
        "normalized_outputs": [
            [{"token_ids": [1, 2]}, {"token_ids": [3]}],
            [{"outputs": [{"token_ids": [4, 5, 6]}]}],
        ]
    }

    assert extract_rollout_candidate_groups(payload) == [[[1, 2], [3]], [[4, 5, 6]]]


def test_extract_rollout_candidate_groups_preserves_empty_completions():
    payload = {
        "normalized_outputs": [
            [{"token_ids": []}, {"token_ids": [3]}],
            [{"outputs": [{"token_ids": []}]}],
        ]
    }

    assert extract_rollout_candidate_groups(payload) == [[[], [3]], [[]]]
    assert extract_rollout_token_groups(payload) == [[], [3], []]


def test_extract_rollout_logp_groups_supports_nested_aliases_and_empty_vectors():
    payload = {
        "outputs": [
            [
                {
                    "outputs": [
                        {
                            "token_ids": [1, 2],
                            "old_policy_logprobs": [-0.1, -0.2],
                            "reference_selected_logprobs": [-0.3, -0.4],
                        }
                    ]
                },
                {"token_ids": [], "old_logps": [], "ref_logps": []},
            ]
        ]
    }

    assert extract_rollout_logp_groups(payload, keys=("old_logps", "old_policy_logprobs")) == [
        [[-0.1, -0.2], []]
    ]
    assert extract_rollout_logp_groups(
        payload, keys=("ref_logps", "reference_selected_logprobs")
    ) == [[[-0.3, -0.4], []]]
