# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kernel-Align Contributors

from __future__ import annotations

import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence

import torch

from rl_engine.alignment.grpo import (
    GRPOConfig,
    broadcast_sequence_advantages,
    compute_group_relative_advantages,
    compute_grpo_loss,
)
from rl_engine.executors.bridge import WeightPublisher, WeightUpdateManifest, make_weight_bridge
from rl_engine.executors.rollout import RolloutExecutor
from rl_engine.executors.stateless_executor import (
    StatelessForwardExecutor,
    StatelessForwardInputs,
    StatelessForwardResult,
)
from rl_engine.testing import (
    SyntheticRLKernelBatch,
    make_synthetic_rl_kernel_batch,
    selected_logprobs_reference,
)


@dataclass(frozen=True)
class PipelineConfig:
    """Local overlap scheduler configuration."""

    max_prefetch: int = 1
    stop_on_error: bool = True
    rollout_workers: int = 1
    training_workers: int = 1
    initial_weight_version: int = 0
    weight_version_policy: str = "published"

    def __post_init__(self) -> None:
        if self.max_prefetch < 1:
            raise ValueError("max_prefetch must be >= 1")
        if self.rollout_workers < 1:
            raise ValueError("rollout_workers must be >= 1")
        if self.training_workers < 1:
            raise ValueError("training_workers must be >= 1")
        if self.weight_version_policy not in {"published", "spec"}:
            raise ValueError("weight_version_policy must be 'published' or 'spec'")


@dataclass(frozen=True)
class IterationSpec:
    """One scheduled rollout/training iteration."""

    iteration: int
    weight_version: Optional[int] = None
    prompts: Sequence[Any] = field(default_factory=list)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RolloutStageResult:
    """Result produced by a rollout worker."""

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
    """Result produced by a training worker."""

    iteration: int
    consumed_weight_version: int
    published_weight_version: Optional[int]
    metrics: Mapping[str, Any]
    started_at: float
    finished_at: float

    @property
    def duration_seconds(self) -> float:
        return self.finished_at - self.started_at


@dataclass(frozen=True)
class WeightHandoffRecord:
    """One published and installed weight manifest handoff."""

    iteration: int
    weight_version: int
    update_id: str
    transport: str
    tensor_count: int
    total_nbytes: int
    published_at: float
    installed_at: Optional[float] = None


@dataclass(frozen=True)
class PipelineTimelineSummary:
    """Serializable summary of one pipeline run."""

    started_at: float
    finished_at: float
    elapsed_seconds: float
    sequential_estimate_seconds: float
    overlap_seconds: float
    overlap_ratio: float
    max_queue_depth: int
    rollout_results: list[dict[str, Any]]
    training_results: list[dict[str, Any]]
    weight_handoffs: list[dict[str, Any]]
    final_published_weight_version: int


class RolloutWorker(Protocol):
    def rollout(self, spec: IterationSpec) -> RolloutStageResult: ...


class TrainingWorker(Protocol):
    def train(self, rollout: RolloutStageResult) -> TrainingStageResult: ...


class PipelineExecutionError(RuntimeError):
    """Raised when a rollout or training stage fails."""

    def __init__(self, stage: str, iteration: int, cause: BaseException):
        super().__init__(f"{stage} failed for iteration {iteration}: {cause}")
        self.stage = stage
        self.iteration = iteration
        self.cause = cause


class ManifestWeightHandoff:
    """
    Publish training weights and install them on rollout workers at safe boundaries.

    Publication happens after a training step finishes. Installation is deferred
    until there is no active rollout future, so a vLLM worker is not hot-updated
    while it may still be generating with the previous version.
    """

    def __init__(self, *, release_on_shutdown: bool = True):
        self.release_on_shutdown = bool(release_on_shutdown)
        self.records: list[WeightHandoffRecord] = []
        self._pending: list[tuple[WeightUpdateManifest, WeightHandoffRecord]] = []
        self._installed_update_ids: list[str] = []

    def publish(
        self,
        training_worker: TrainingWorker,
        result: TrainingStageResult,
    ) -> Optional[WeightHandoffRecord]:
        if result.published_weight_version is None:
            return None
        self._release_pending(training_worker)
        publish_weights = getattr(training_worker, "publish_weights", None)
        if not callable(publish_weights):
            raise RuntimeError(
                "manifest weight handoff requires the training worker to expose "
                "publish_weights(weight_version=..., metadata=...)"
            )

        manifest = publish_weights(
            weight_version=result.published_weight_version,
            metadata={
                "issue": "18",
                "pipeline_iteration": result.iteration,
                "consumed_weight_version": result.consumed_weight_version,
            },
        )
        record = WeightHandoffRecord(
            iteration=result.iteration,
            weight_version=manifest.weight_version,
            update_id=manifest.update_id,
            transport=manifest.transport,
            tensor_count=manifest.tensor_count,
            total_nbytes=manifest.total_nbytes,
            published_at=time.perf_counter(),
        )
        self._pending.append((manifest, record))
        return record

    def _release_pending(self, training_worker: TrainingWorker) -> None:
        if not self._pending:
            return
        release_training = getattr(training_worker, "release_weights", None)
        if callable(release_training):
            for manifest, _record in self._pending:
                release_training(manifest.update_id)
        self._pending.clear()

    def pending_iteration(self) -> int:
        if not self._pending:
            return -1
        return int(self._pending[-1][1].iteration)

    def install_latest(
        self,
        rollout_worker: RolloutWorker,
        training_worker: Optional[TrainingWorker] = None,
    ) -> Optional[WeightHandoffRecord]:
        if not self._pending:
            return None
        install = _resolve_manifest_install(rollout_worker)
        latest_manifest, latest_record = self._pending[-1]
        installed_at = time.perf_counter()
        install(latest_manifest)
        installed_record = replace(latest_record, installed_at=installed_at)
        self.records.append(installed_record)
        self._installed_update_ids.append(latest_manifest.update_id)
        self._pending.clear()
        return installed_record

    def release_all(
        self,
        training_worker: TrainingWorker,
        rollout_worker: RolloutWorker,
    ) -> None:
        if not self.release_on_shutdown:
            return

        release_rollout = getattr(rollout_worker, "release_weight_manifest", None)
        if callable(release_rollout):
            for update_id in reversed(self._installed_update_ids):
                release_rollout(update_id)

        release_training = getattr(training_worker, "release_weights", None)
        if callable(release_training):
            update_ids = [record.update_id for record in self.records]
            update_ids.extend(manifest.update_id for manifest, _record in self._pending)
            for update_id in reversed(update_ids):
                release_training(update_id)
        self._pending.clear()
        self._installed_update_ids.clear()


class OverlapPipeline:
    """Coordinate rollout and training stages with bounded prefetch overlap."""

    def __init__(
        self,
        rollout_worker: RolloutWorker,
        training_worker: TrainingWorker,
        config: Optional[PipelineConfig] = None,
        *,
        weight_handoff: Optional[ManifestWeightHandoff] = None,
    ):
        self.rollout_worker = rollout_worker
        self.training_worker = training_worker
        self.config = config or PipelineConfig()
        self.weight_handoff = weight_handoff
        self.rollout_results: list[RolloutStageResult] = []
        self.training_results: list[TrainingStageResult] = []
        self.weight_handoffs: list[WeightHandoffRecord] = []
        self.max_queue_depth = 0
        self.started_at = 0.0
        self.finished_at = 0.0
        self.final_published_weight_version = self.config.initial_weight_version

    def run(self, iterations: Sequence[IterationSpec]) -> list[TrainingStageResult]:
        specs = list(iterations)
        if not specs:
            return []

        self.rollout_results = []
        self.training_results = []
        self.weight_handoffs = []
        self.max_queue_depth = 0
        self.started_at = time.perf_counter()
        self.finished_at = self.started_at
        self.final_published_weight_version = self.config.initial_weight_version

        rollout_futures: dict[Future[RolloutStageResult], int] = {}
        training_futures: dict[Future[TrainingStageResult], int] = {}
        buffered_rollouts: dict[int, RolloutStageResult] = {}
        submitted_specs: dict[int, IterationSpec] = {}
        next_rollout_index = 0
        next_training_index = 0
        current_published_weight_version = self.config.initial_weight_version
        current_rollout_weight_version = self.config.initial_weight_version

        def rollout_backlog() -> int:
            return len(rollout_futures) + len(buffered_rollouts)

        def rollout_spec_for(index: int) -> IterationSpec:
            base_spec = specs[index]
            if self.config.weight_version_policy == "spec":
                if base_spec.weight_version is None:
                    raise ValueError("IterationSpec.weight_version is required for spec policy")
                return base_spec
            visible_version = (
                current_rollout_weight_version
                if self.weight_handoff is not None
                else current_published_weight_version
            )
            return replace(base_spec, weight_version=visible_version)

        def install_pending_weights_if_idle() -> None:
            nonlocal current_rollout_weight_version
            if self.weight_handoff is None or rollout_futures:
                return
            try:
                installed = self.weight_handoff.install_latest(
                    self.rollout_worker,
                    self.training_worker,
                )
            except BaseException as exc:
                raise PipelineExecutionError(
                    "weight_handoff",
                    self.weight_handoff.pending_iteration(),
                    exc,
                ) from exc
            if installed is None:
                return
            current_rollout_weight_version = installed.weight_version
            self.weight_handoffs.append(installed)

        def submit_rollouts(executor: ThreadPoolExecutor) -> int:
            nonlocal next_rollout_index
            submitted = 0
            while next_rollout_index < len(specs) and rollout_backlog() < self.config.max_prefetch:
                index = next_rollout_index
                rollout_spec = rollout_spec_for(index)
                submitted_specs[index] = rollout_spec
                future = executor.submit(self.rollout_worker.rollout, rollout_spec)
                rollout_futures[future] = index
                next_rollout_index += 1
                submitted += 1
            self.max_queue_depth = max(self.max_queue_depth, rollout_backlog())
            return submitted

        def submit_training(executor: ThreadPoolExecutor) -> int:
            nonlocal next_training_index
            submitted = 0
            while (
                next_training_index < len(specs)
                and next_training_index in buffered_rollouts
                and len(training_futures) < self.config.training_workers
            ):
                rollout = buffered_rollouts.pop(next_training_index)
                future = executor.submit(self.training_worker.train, rollout)
                training_futures[future] = next_training_index
                next_training_index += 1
                submitted += 1
            self.max_queue_depth = max(self.max_queue_depth, rollout_backlog())
            return submitted

        try:
            with (
                ThreadPoolExecutor(
                    max_workers=self.config.rollout_workers,
                    thread_name_prefix="kernel-align-rollout",
                ) as rollout_executor,
                ThreadPoolExecutor(
                    max_workers=self.config.training_workers,
                    thread_name_prefix="kernel-align-training",
                ) as training_executor,
            ):
                submit_rollouts(rollout_executor)

                while len(self.training_results) < len(specs):
                    install_pending_weights_if_idle()
                    submit_training(training_executor)
                    submit_rollouts(rollout_executor)

                    active_futures: set[Future[Any]] = set(rollout_futures) | set(training_futures)
                    if not active_futures:
                        break

                    done, _ = wait(active_futures, return_when=FIRST_COMPLETED)
                    for future in done:
                        if future in rollout_futures:
                            index = rollout_futures.pop(future)
                            rollout_spec = submitted_specs[index]
                            try:
                                result = future.result()
                            except BaseException as exc:
                                self._cancel_pending(rollout_futures, training_futures)
                                raise PipelineExecutionError(
                                    "rollout", rollout_spec.iteration, exc
                                ) from exc
                            self._validate_rollout_result(rollout_spec, result)
                            buffered_rollouts[index] = result
                            self.rollout_results.append(result)
                            self.max_queue_depth = max(self.max_queue_depth, rollout_backlog())
                        elif future in training_futures:
                            index = training_futures.pop(future)
                            rollout_spec = submitted_specs[index]
                            try:
                                result = future.result()
                            except BaseException as exc:
                                self._cancel_pending(rollout_futures, training_futures)
                                raise PipelineExecutionError(
                                    "training", rollout_spec.iteration, exc
                                ) from exc
                            try:
                                self._validate_training_result(rollout_spec, result)
                                if result.published_weight_version is not None:
                                    if self.weight_handoff is not None:
                                        self.weight_handoff.publish(self.training_worker, result)
                                    current_published_weight_version = max(
                                        current_published_weight_version,
                                        result.published_weight_version,
                                    )
                                self.training_results.append(result)
                            except BaseException as exc:
                                self._cancel_pending(rollout_futures, training_futures)
                                raise PipelineExecutionError(
                                    "weight_handoff", rollout_spec.iteration, exc
                                ) from exc

                return list(self.training_results)
        finally:
            try:
                install_pending_weights_if_idle()
            finally:
                if self.weight_handoff is not None:
                    self.weight_handoff.release_all(self.training_worker, self.rollout_worker)
            self.final_published_weight_version = current_published_weight_version
            self.finished_at = time.perf_counter()

    def timeline_summary(self) -> PipelineTimelineSummary:
        finished_at = self.finished_at or time.perf_counter()
        rollout_rows = [_stage_to_dict(result) for result in self.rollout_results]
        training_rows = [_stage_to_dict(result) for result in self.training_results]
        sequential = sum(result.duration_seconds for result in self.rollout_results) + sum(
            result.duration_seconds for result in self.training_results
        )
        overlap = compute_stage_overlap_seconds(self.rollout_results, self.training_results)
        rollout_total = sum(result.duration_seconds for result in self.rollout_results)
        denominator = max(rollout_total, 1e-12)
        return PipelineTimelineSummary(
            started_at=self.started_at,
            finished_at=finished_at,
            elapsed_seconds=finished_at - self.started_at,
            sequential_estimate_seconds=sequential,
            overlap_seconds=overlap,
            overlap_ratio=overlap / denominator,
            max_queue_depth=self.max_queue_depth,
            rollout_results=rollout_rows,
            training_results=training_rows,
            weight_handoffs=[asdict(record) for record in self.weight_handoffs],
            final_published_weight_version=self.final_published_weight_version,
        )

    @staticmethod
    def _cancel_pending(
        rollout_futures: Mapping[Future[RolloutStageResult], int],
        training_futures: Mapping[Future[TrainingStageResult], int],
    ) -> None:
        for future in list(rollout_futures) + list(training_futures):
            future.cancel()

    @staticmethod
    def _validate_rollout_result(spec: IterationSpec, result: RolloutStageResult) -> None:
        if result.iteration != spec.iteration:
            raise ValueError(
                f"rollout result iteration {result.iteration} does not match spec "
                f"{spec.iteration}"
            )
        if result.weight_version != spec.weight_version:
            raise ValueError(
                f"rollout result weight_version {result.weight_version} does not match spec "
                f"{spec.weight_version}"
            )

    @staticmethod
    def _validate_training_result(spec: IterationSpec, result: TrainingStageResult) -> None:
        if result.iteration != spec.iteration:
            raise ValueError(
                f"training result iteration {result.iteration} does not match spec "
                f"{spec.iteration}"
            )


class RolloutExecutorWorker:
    """Production-facing rollout adapter over RolloutExecutor.generate_candidates."""

    def __init__(
        self,
        executor: RolloutExecutor,
        *,
        num_generations: Optional[int] = None,
        sampling_params: Optional[Mapping[str, Any]] = None,
    ):
        self.executor = executor
        self.num_generations = num_generations
        self.sampling_params = dict(sampling_params or {})

    def rollout(self, spec: IterationSpec) -> RolloutStageResult:
        started_at = time.perf_counter()
        payload = self.executor.generate_candidates(
            spec.prompts,
            num_generations=self.num_generations,
            sampling_params=self.sampling_params,
        )
        finished_at = time.perf_counter()
        return RolloutStageResult(
            iteration=spec.iteration,
            weight_version=_require_weight_version(spec),
            payload=payload,
            started_at=started_at,
            finished_at=finished_at,
            metrics={
                "num_prompts": len(spec.prompts),
                "backend": payload.get("backend") if isinstance(payload, Mapping) else None,
            },
        )

    def install_weight_manifest(self, manifest: WeightUpdateManifest) -> Mapping[str, torch.Tensor]:
        return self.executor.update_weights(manifest)

    def release_weight_manifest(self, update_id: str) -> None:
        release_update = getattr(self.executor, "release_weight_update", None)
        if callable(release_update):
            release_update(update_id)
        elif getattr(self.executor, "active_weight_update_id", None) == update_id:
            self.executor.release_weights()


class StatelessScoringWorker:
    """Attach no-cache reference/reward scores to a completed rollout payload."""

    def __init__(
        self,
        executor: StatelessForwardExecutor,
        collate_inputs: Callable[[RolloutStageResult], StatelessForwardInputs],
    ):
        self.executor = executor
        self.collate_inputs = collate_inputs

    def score(self, rollout: RolloutStageResult) -> RolloutStageResult:
        started_at = time.perf_counter()
        inputs = self.collate_inputs(rollout)
        result = self.executor.score(inputs)
        finished_at = time.perf_counter()
        return RolloutStageResult(
            iteration=rollout.iteration,
            weight_version=rollout.weight_version,
            payload=attach_stateless_scores_to_payload(rollout.payload, result, inputs=inputs),
            started_at=started_at,
            finished_at=finished_at,
            metrics={
                **dict(rollout.metrics),
                "scoring_backend": "stateless",
                "scoring_mode": result.metrics["mode"],
                "scoring_elapsed_ms": result.metrics["elapsed_ms"],
                "scoring_active_completion_tokens": result.metrics["active_completion_tokens"],
                "scoring_zero_kv_cache": result.metrics["zero_kv_cache"],
                "scoring_attention_backend": result.metrics["attention_backend"],
                "scoring_kv_cache_output_mb": result.metrics["kv_cache_output_mb"],
            },
        )


def attach_stateless_scores_to_payload(
    payload: Any,
    result: StatelessForwardResult,
    *,
    inputs: Optional[StatelessForwardInputs] = None,
) -> dict[str, Any]:
    """Return a rollout payload augmented with stateless scoring outputs."""

    scored_payload = dict(payload) if isinstance(payload, Mapping) else {"raw_payload": payload}
    scored_payload["stateless_scores"] = {
        "reference_logps": result.reference_logps,
        "rewards": result.rewards,
        "token_scores": result.token_scores,
        "metrics": dict(result.metrics),
    }
    if result.rewards is not None:
        _attach_reward_tensor_to_grouped_candidates(scored_payload, result.rewards)
    if result.reference_logps is not None and inputs is not None:
        _attach_reference_logps_to_grouped_candidates(
            scored_payload,
            result.reference_logps,
            inputs.completion_mask,
        )
    return scored_payload


def build_stateless_inputs_from_rollout_payload(
    payload: Any,
    *,
    prompt_len: int = 1,
    prompt_token_id: int = 0,
    max_completion_len: Optional[int] = None,
    device: torch.device | str = "cpu",
) -> StatelessForwardInputs:
    """Build dense no-cache scoring inputs from grouped rollout token payloads."""

    if prompt_len < 0:
        raise ValueError("prompt_len must be non-negative")
    if max_completion_len is not None and max_completion_len <= 0:
        raise ValueError("max_completion_len must be greater than zero")

    candidate_groups = extract_rollout_candidate_groups(payload)
    flat_token_groups = [tokens for group in candidate_groups for tokens in group]
    if not flat_token_groups:
        raise ValueError("rollout payload does not contain candidate token ids")

    completion_len = max(len(tokens) for tokens in flat_token_groups)
    if max_completion_len is not None:
        completion_len = min(completion_len, max_completion_len)
    completion_len = max(1, completion_len)
    seq_len = prompt_len + completion_len
    resolved_device = torch.device(device)

    input_ids = torch.full(
        (len(flat_token_groups), seq_len),
        int(prompt_token_id),
        device=resolved_device,
        dtype=torch.long,
    )
    attention_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    completion_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    if prompt_len:
        attention_mask[:, :prompt_len] = True

    for row, token_ids in enumerate(flat_token_groups):
        clipped = [int(token) for token in token_ids[:completion_len]]
        if clipped:
            values = torch.tensor(clipped, device=resolved_device, dtype=torch.long)
            start = prompt_len
            end = prompt_len + values.numel()
            input_ids[row, start:end] = values
            attention_mask[row, start:end] = True
            completion_mask[row, start:end] = True

    if not bool(completion_mask.any().item()):
        raise ValueError("rollout payload does not contain active completion tokens")
    return StatelessForwardInputs(
        input_ids=input_ids,
        attention_mask=attention_mask,
        completion_mask=completion_mask,
    )


def _attach_reward_tensor_to_grouped_candidates(
    payload: dict[str, Any],
    rewards: torch.Tensor,
) -> None:
    reward_values = [float(value) for value in rewards.detach().cpu().reshape(-1).tolist()]
    if not reward_values:
        return

    for key in ("normalized_outputs", "outputs"):
        grouped_outputs = payload.get(key)
        if not isinstance(grouped_outputs, Sequence) or isinstance(grouped_outputs, (str, bytes)):
            continue
        updated_outputs, used = _attach_rewards_to_grouped_outputs(
            grouped_outputs,
            reward_values,
        )
        if used == 0:
            continue
        if used != len(reward_values):
            raise ValueError(
                "stateless reward count must match rollout candidate count, got "
                f"{len(reward_values)} rewards for {used} candidates"
            )
        payload[key] = updated_outputs
        return


def _attach_reference_logps_to_grouped_candidates(
    payload: dict[str, Any],
    reference_logps: torch.Tensor,
    completion_mask: torch.Tensor,
) -> None:
    if reference_logps.shape != completion_mask.shape:
        raise ValueError("reference_logps shape must match completion_mask shape")

    mask = completion_mask.detach().to(dtype=torch.bool, device=reference_logps.device)
    logp_rows = [
        [float(value) for value in reference_logps[row][mask[row]].detach().cpu().tolist()]
        for row in range(reference_logps.shape[0])
    ]
    if not logp_rows:
        return

    for key in ("normalized_outputs", "outputs"):
        grouped_outputs = payload.get(key)
        if not isinstance(grouped_outputs, Sequence) or isinstance(grouped_outputs, (str, bytes)):
            continue
        updated_outputs, used = _attach_reference_logps_to_grouped_outputs(
            grouped_outputs,
            logp_rows,
        )
        if used == 0:
            continue
        if used != len(logp_rows):
            raise ValueError(
                "stateless reference_logps row count must match rollout candidate count, got "
                f"{len(logp_rows)} rows for {used} candidates"
            )
        payload[key] = updated_outputs
        return


def _attach_rewards_to_grouped_outputs(
    grouped_outputs: Sequence[Any],
    rewards: Sequence[float],
) -> tuple[list[Any], int]:
    updated_groups: list[Any] = []
    reward_index = 0

    for group in grouped_outputs:
        if isinstance(group, Sequence) and not isinstance(group, (str, bytes, Mapping)):
            updated_candidates = []
            for candidate in group:
                updated_candidate, consumed = _attach_reward_to_candidate(
                    candidate,
                    rewards,
                    reward_index,
                )
                reward_index += consumed
                updated_candidates.append(updated_candidate)
            updated_groups.append(updated_candidates)
        else:
            updated_candidate, consumed = _attach_reward_to_candidate(
                group,
                rewards,
                reward_index,
            )
            reward_index += consumed
            updated_groups.append(updated_candidate)

    return updated_groups, reward_index


def _attach_reward_to_candidate(
    candidate: Any,
    rewards: Sequence[float],
    reward_index: int,
) -> tuple[Any, int]:
    token_ids = _candidate_token_ids(candidate)
    if not token_ids:
        return candidate, 0
    if reward_index >= len(rewards):
        raise ValueError(
            "stateless reward count must match rollout candidate count, got fewer rewards "
            "than candidates"
        )

    reward = float(rewards[reward_index])
    if isinstance(candidate, Mapping):
        updated = dict(candidate)
        updated["reward"] = reward
        updated["reward_source"] = "stateless_executor"
        return updated, 1

    try:
        updated = dict(vars(candidate))
    except TypeError:
        updated = {"candidate": candidate}
    updated["reward"] = reward
    updated["reward_source"] = "stateless_executor"
    return updated, 1


def _attach_reference_logps_to_grouped_outputs(
    grouped_outputs: Sequence[Any],
    logp_rows: Sequence[Sequence[float]],
) -> tuple[list[Any], int]:
    updated_groups: list[Any] = []
    row_index = 0

    for group in grouped_outputs:
        if isinstance(group, Sequence) and not isinstance(group, (str, bytes, Mapping)):
            updated_candidates = []
            for candidate in group:
                updated_candidate, consumed = _attach_reference_logps_to_candidate(
                    candidate,
                    logp_rows,
                    row_index,
                )
                row_index += consumed
                updated_candidates.append(updated_candidate)
            updated_groups.append(updated_candidates)
        else:
            updated_candidate, consumed = _attach_reference_logps_to_candidate(
                group,
                logp_rows,
                row_index,
            )
            row_index += consumed
            updated_groups.append(updated_candidate)

    return updated_groups, row_index


def _attach_reference_logps_to_candidate(
    candidate: Any,
    logp_rows: Sequence[Sequence[float]],
    row_index: int,
) -> tuple[Any, int]:
    token_ids = _candidate_token_ids(candidate)
    if not token_ids:
        return candidate, 0
    if row_index >= len(logp_rows):
        raise ValueError(
            "stateless reference_logps row count must match rollout candidate count, got "
            "fewer rows than candidates"
        )

    values = [float(value) for value in logp_rows[row_index]]
    if isinstance(candidate, Mapping):
        updated = dict(candidate)
        updated["reference_logps"] = values
        updated["reference_logp_source"] = "stateless_executor"
        return updated, 1

    try:
        updated = dict(vars(candidate))
    except TypeError:
        updated = {"candidate": candidate}
    updated["reference_logps"] = values
    updated["reference_logp_source"] = "stateless_executor"
    return updated, 1


@dataclass(frozen=True)
class TorchRLTrainingConfig:
    """Config for a real PyTorch RL-style smoke training step."""

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


class TorchRLTrainingWorker:
    """DeepSpeed-style local training adapter using a real PyTorch step."""

    def __init__(
        self,
        config: Optional[TorchRLTrainingConfig] = None,
        *,
        weight_bridge: Optional[WeightPublisher] = None,
        weight_transport: str = "local-clone",
    ):
        self.config = config or TorchRLTrainingConfig()
        self.weight_bridge = weight_bridge or make_weight_bridge(
            weight_transport,
            source_worker="torch-training",
            source_rank=0,
        )
        self.device = torch.device(self.config.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA training requested but torch.cuda.is_available() is false")

        torch.manual_seed(self.config.seed)
        self.model = torch.nn.Sequential(
            torch.nn.Embedding(self.config.vocab_size, self.config.hidden_dim),
            torch.nn.Linear(self.config.hidden_dim, self.config.vocab_size),
        ).to(device=self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.lr)
        self._latest_published_weight_version = -1

    def train(self, rollout: RolloutStageResult) -> TrainingStageResult:
        started_at = time.perf_counter()
        batch, payload_metrics = self._batch_from_rollout_or_synthetic(rollout)

        logits = self.model(batch.token_ids.long())
        objective = self._compute_grpo_objective(logits, batch)
        loss = objective.loss

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        finished_at = time.perf_counter()
        published = self._next_published_weight_version(rollout.weight_version)
        return TrainingStageResult(
            iteration=rollout.iteration,
            consumed_weight_version=rollout.weight_version,
            published_weight_version=published,
            metrics={
                "loss": float(loss.detach().cpu().item()),
                "active_tokens": int(batch.completion_mask.sum().item()),
                "payload_type": type(rollout.payload).__name__,
                "training_backend": "torch",
                "objective": "grpo",
                **objective.metrics,
                **payload_metrics,
            },
            started_at=started_at,
            finished_at=finished_at,
        )

    def _compute_grpo_objective(self, logits: torch.Tensor, batch: SyntheticRLKernelBatch):
        current_logps = selected_logprobs_reference(
            logits,
            batch.token_ids,
            mask=batch.completion_mask,
            output_dtype=torch.float32,
        )
        old_logps = current_logps.detach() - 0.01
        ref_logps = _objective_reference_logps(current_logps, batch)
        return compute_grpo_loss(
            current_logps=current_logps,
            old_logps=old_logps,
            ref_logps=ref_logps,
            rewards=batch.rewards,
            completion_mask=batch.completion_mask,
            group_ids=_batch_group_ids(batch),
            num_groups=int(batch.metadata.get("num_prompts", 1)),
            config=GRPOConfig(
                clip_epsilon=self.config.clip_epsilon,
                kl_beta=self.config.kl_beta,
                advantage_eps=self.config.advantage_eps,
            ),
        )

    def publish_weights(
        self,
        *,
        weight_version: int,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> WeightUpdateManifest:
        return self.weight_bridge.publish(
            self.model,
            weight_version=weight_version,
            metadata=metadata,
        )

    def release_weights(self, update_id: str) -> None:
        self.weight_bridge.release(update_id)

    def _next_published_weight_version(self, consumed_weight_version: int) -> int:
        published = max(
            self._latest_published_weight_version + 1,
            int(consumed_weight_version) + 1,
        )
        self._latest_published_weight_version = published
        return published

    def _batch_from_rollout_or_synthetic(
        self,
        rollout: RolloutStageResult,
    ) -> tuple[SyntheticRLKernelBatch, dict[str, Any]]:
        candidate_groups = extract_rollout_candidate_groups(rollout.payload)
        if candidate_groups:
            reward_groups = extract_rollout_reward_groups(rollout.payload)
            reference_logp_groups = extract_rollout_reference_logp_groups(rollout.payload)
            has_payload_rewards = _reward_groups_match(candidate_groups, reward_groups)
            has_payload_reference_logps = _reference_logp_groups_match(
                candidate_groups,
                reference_logp_groups,
            )
            batch = self._batch_from_candidate_groups(
                candidate_groups,
                rollout,
                reward_groups=reward_groups if has_payload_rewards else None,
                reference_logp_groups=(
                    reference_logp_groups if has_payload_reference_logps else None
                ),
            )
            token_groups = [tokens for group in candidate_groups for tokens in group]
            return batch, {
                "training_data_source": "rollout_payload",
                "reward_source": "payload_rewards" if has_payload_rewards else "token_id_proxy",
                "reference_logp_source": (
                    "payload_reference_logps"
                    if has_payload_reference_logps
                    else "synthetic_current_offset"
                ),
                "rollout_prompt_groups": len(candidate_groups),
                "rollout_sequences": len(token_groups),
                "rollout_tokens": sum(len(group) for group in token_groups),
            }

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
            "reference_logp_source": "synthetic_current_offset",
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
        )

    def _batch_from_candidate_groups(
        self,
        candidate_groups: Sequence[Sequence[Sequence[int]]],
        rollout: RolloutStageResult,
        *,
        reward_groups: Optional[Sequence[Sequence[float]]] = None,
        reference_logp_groups: Optional[Sequence[Sequence[Sequence[float]]]] = None,
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
        flat_reference_logps: list[list[float]] = []
        row = 0
        for group_index, group in enumerate(candidate_groups):
            for candidate_index, candidate_tokens in enumerate(group):
                clipped = [
                    int(token) % self.config.vocab_size
                    for token in candidate_tokens[:completion_len]
                ]
                if clipped:
                    token_tensor = torch.tensor(
                        clipped,
                        device=self.device,
                        dtype=torch.long,
                    )
                    token_ids[row, : token_tensor.numel()] = token_tensor
                    completion_mask[row, : token_tensor.numel()] = True
                flat_rewards.append(
                    _candidate_reward_value(
                        candidate_tokens,
                        reward_groups,
                        group_index,
                        candidate_index,
                        vocab_size=self.config.vocab_size,
                    )
                )
                if reference_logp_groups is not None:
                    flat_reference_logps.append(
                        _candidate_reference_logps(
                            reference_logp_groups,
                            group_index,
                            candidate_index,
                            completion_len=completion_len,
                        )
                    )
                flat_group_ids.append(group_index)
                row += 1

        if not bool(completion_mask.any().item()):
            completion_mask[:, :1] = True

        rewards = torch.tensor(
            flat_rewards,
            device=self.device,
            dtype=self.config.dtype,
        )
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
        if reference_logp_groups is not None:
            for row_index, reference_values in enumerate(flat_reference_logps):
                if reference_values:
                    value_tensor = torch.tensor(
                        reference_values,
                        device=self.device,
                        dtype=self.config.dtype,
                    )
                    ref_logps[row_index, : value_tensor.numel()] = value_tensor
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
            "reward_source": "payload_rewards" if reward_groups is not None else "token_id_proxy",
            "reference_logp_source": (
                "payload_reference_logps"
                if reference_logp_groups is not None
                else "synthetic_current_offset"
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


def compute_stage_overlap_seconds(
    rollouts: Sequence[RolloutStageResult],
    trainings: Sequence[TrainingStageResult],
) -> float:
    """Compute pairwise rollout/training interval overlap in seconds."""

    overlap = 0.0
    for rollout in rollouts:
        for training in trainings:
            if rollout.iteration == training.iteration:
                continue
            start = max(rollout.started_at, training.started_at)
            end = min(rollout.finished_at, training.finished_at)
            overlap += max(0.0, end - start)
    return overlap


def extract_rollout_token_groups(payload: Any) -> list[list[int]]:
    """Extract generated token ids from Kernel-Align/vLLM-style rollout payloads."""

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


def extract_rollout_reference_logp_groups(payload: Any) -> list[list[list[float]]]:
    """Extract per-candidate reference logprobs from rollout payloads when present."""

    if not isinstance(payload, Mapping):
        return []

    normalized_outputs = payload.get("normalized_outputs")
    if isinstance(normalized_outputs, Sequence) and not isinstance(
        normalized_outputs, (str, bytes)
    ):
        groups = _reference_logp_groups_from_grouped_outputs(normalized_outputs)
        if groups:
            return groups

    outputs = payload.get("outputs")
    if isinstance(outputs, Sequence) and not isinstance(outputs, (str, bytes)):
        return _reference_logp_groups_from_grouped_outputs(outputs)
    return []


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
            if token_ids:
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


def _reference_logp_groups_from_grouped_outputs(
    grouped_outputs: Sequence[Any],
) -> list[list[list[float]]]:
    groups: list[list[list[float]]] = []
    for group in grouped_outputs:
        if isinstance(group, Sequence) and not isinstance(group, (str, bytes, Mapping)):
            candidates = group
        else:
            candidates = [group]
        reference_group = []
        for candidate in candidates:
            reference_logps = _candidate_reference_logp_values(candidate)
            if reference_logps:
                reference_group.append(reference_logps)
        if reference_group:
            groups.append(reference_group)
    return groups


def _candidate_token_ids(candidate: Any) -> list[int]:
    if candidate is None:
        return []
    if isinstance(candidate, Mapping):
        nested_outputs = candidate.get("outputs")
        if isinstance(nested_outputs, Sequence) and not isinstance(nested_outputs, (str, bytes)):
            for nested in nested_outputs:
                token_ids = _candidate_token_ids(nested)
                if token_ids:
                    return token_ids
        value = candidate.get("token_ids")
        return _copy_int_list(value)

    value = getattr(candidate, "token_ids", None)
    if value is not None:
        return _copy_int_list(value)
    nested_outputs = getattr(candidate, "outputs", None)
    if isinstance(nested_outputs, Sequence) and not isinstance(nested_outputs, (str, bytes)):
        for nested in nested_outputs:
            token_ids = _candidate_token_ids(nested)
            if token_ids:
                return token_ids
    return []


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


def _candidate_reference_logp_values(candidate: Any) -> list[float]:
    if candidate is None:
        return []
    if isinstance(candidate, Mapping):
        nested_outputs = candidate.get("outputs")
        if isinstance(nested_outputs, Sequence) and not isinstance(nested_outputs, (str, bytes)):
            for nested in nested_outputs:
                reference_logps = _candidate_reference_logp_values(nested)
                if reference_logps:
                    return reference_logps
        for key in ("reference_logps", "ref_logps", "reference_logprobs", "ref_logprobs"):
            if key in candidate:
                return _copy_float_list(candidate[key])
        return []

    nested_outputs = getattr(candidate, "outputs", None)
    if isinstance(nested_outputs, Sequence) and not isinstance(nested_outputs, (str, bytes)):
        for nested in nested_outputs:
            reference_logps = _candidate_reference_logp_values(nested)
            if reference_logps:
                return reference_logps
    for attr in ("reference_logps", "ref_logps", "reference_logprobs", "ref_logprobs"):
        if hasattr(candidate, attr):
            return _copy_float_list(getattr(candidate, attr))
    return []


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


def _reference_logp_groups_match(
    candidate_groups: Sequence[Sequence[Sequence[int]]],
    reference_logp_groups: Sequence[Sequence[Sequence[float]]],
) -> bool:
    if len(candidate_groups) != len(reference_logp_groups):
        return False
    for candidates, reference_group in zip(
        candidate_groups,
        reference_logp_groups,
        strict=False,
    ):
        if len(candidates) != len(reference_group):
            return False
        for token_ids, reference_logps in zip(candidates, reference_group, strict=False):
            if len(reference_logps) < len(token_ids):
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


def _candidate_reference_logps(
    reference_logp_groups: Sequence[Sequence[Sequence[float]]],
    group_index: int,
    candidate_index: int,
    *,
    completion_len: int,
) -> list[float]:
    values = [
        float(value)
        for value in reference_logp_groups[group_index][candidate_index][:completion_len]
    ]
    if len(values) > completion_len:
        return values[:completion_len]
    return values


def _require_weight_version(spec: IterationSpec) -> int:
    if spec.weight_version is None:
        raise ValueError("IterationSpec.weight_version is required for rollout execution")
    return int(spec.weight_version)


def _batch_group_ids(batch: SyntheticRLKernelBatch) -> Optional[torch.Tensor]:
    group_ids = batch.metadata.get("group_ids")
    if group_ids is None:
        return None
    if isinstance(group_ids, torch.Tensor):
        return group_ids.to(device=batch.rewards.device, dtype=torch.long)
    return torch.tensor(group_ids, device=batch.rewards.device, dtype=torch.long)


def _objective_reference_logps(
    current_logps: torch.Tensor,
    batch: SyntheticRLKernelBatch,
) -> torch.Tensor:
    if batch.metadata.get("reference_logp_source") != "payload_reference_logps":
        return current_logps.detach() - 0.02
    if batch.ref_logps.shape != current_logps.shape:
        raise ValueError("payload reference logps shape must match current logps shape")
    return batch.ref_logps.detach().to(device=current_logps.device, dtype=torch.float32)


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


def _resolve_manifest_install(rollout_worker: RolloutWorker):
    install = getattr(rollout_worker, "install_weight_manifest", None)
    if callable(install):
        return install
    update_weights = getattr(rollout_worker, "update_weights", None)
    if callable(update_weights):
        return update_weights
    executor = getattr(rollout_worker, "executor", None)
    update_weights = getattr(executor, "update_weights", None)
    if callable(update_weights):
        return update_weights
    raise RuntimeError(
        "manifest weight handoff requires the rollout worker to expose "
        "install_weight_manifest(manifest) or update_weights(manifest)"
    )


def timeline_summary_to_dict(summary: PipelineTimelineSummary) -> dict[str, Any]:
    return asdict(summary)


def _stage_to_dict(stage: RolloutStageResult | TrainingStageResult) -> dict[str, Any]:
    row = asdict(stage)
    row.pop("payload", None)
    row["duration_seconds"] = stage.duration_seconds
    return row
