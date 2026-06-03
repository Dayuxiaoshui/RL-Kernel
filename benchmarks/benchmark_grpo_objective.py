# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kernel-Align Contributors

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl_engine.alignment.grpo import GRPOConfig, compute_grpo_loss  # noqa: E402
from rl_engine.testing import (  # noqa: E402
    make_synthetic_rl_kernel_batch,
    selected_logprobs_reference,
    summarize_kernel_drift,
)

CSV_COLUMNS = [
    "timestamp",
    "candidate",
    "mode",
    "stage",
    "num_prompts",
    "samples_per_prompt",
    "completion_len",
    "vocab_size",
    "device",
    "dtype",
    "advantage_max_abs_error",
    "loss_max_abs_error",
    "active_tokens",
    "status",
    "notes",
]


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _environment() -> str:
    parts = [f"torch={torch.__version__}", f"cuda_available={torch.cuda.is_available()}"]
    if torch.cuda.is_available():
        parts.append(f"cuda={torch.version.cuda}")
        parts.append(f"gpu={torch.cuda.get_device_name(0)}")
    return ";".join(parts)


def _append_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow({column: row.get(column, "") for column in CSV_COLUMNS})


def _write_json(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _make_inputs(args: argparse.Namespace, *, device: torch.device, dtype: torch.dtype):
    batch = make_synthetic_rl_kernel_batch(
        num_prompts=args.num_prompts,
        samples_per_prompt=args.samples_per_prompt,
        prompt_len=args.prompt_len,
        completion_len=args.completion_len,
        vocab_size=args.vocab_size,
        valid_density=args.valid_density,
        dtype=dtype,
        device=device,
        seed=args.seed,
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + 1009)
    logits = torch.randn(
        batch.batch_size,
        batch.completion_len,
        args.vocab_size,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    current_logps = selected_logprobs_reference(
        logits,
        batch.token_ids,
        mask=batch.completion_mask,
        output_dtype=torch.float32,
    )
    old_logps = current_logps.detach() - 0.01
    ref_logps = current_logps.detach() - 0.02
    return batch, logits, current_logps, old_logps, ref_logps


def _run_reference(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    batch, _logits, current_logps, old_logps, ref_logps = _make_inputs(
        args,
        device=device,
        dtype=dtype,
    )
    config = GRPOConfig(
        clip_epsilon=args.clip_epsilon,
        kl_beta=args.kl_beta,
        advantage_eps=args.advantage_eps,
        loss_reduction=args.loss_reduction,
    )

    started = time.perf_counter()
    result = compute_grpo_loss(
        current_logps,
        old_logps,
        ref_logps,
        batch.rewards,
        batch.completion_mask,
        num_groups=args.num_prompts,
        config=config,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    return {
        "timestamp": _timestamp(),
        "candidate": args.candidate,
        "mode": "reference",
        "stage": "evaluation",
        "num_prompts": args.num_prompts,
        "samples_per_prompt": args.samples_per_prompt,
        "completion_len": args.completion_len,
        "vocab_size": args.vocab_size,
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "advantage_max_abs_error": "0.000000",
        "loss_max_abs_error": "0.000000",
        "active_tokens": int(result.metrics["active_tokens"]),
        "status": "pass",
        "notes": (
            f"elapsed_ms={elapsed_ms:.4f};loss={result.metrics['loss']:.8f};"
            f"kl_mean={result.metrics['kl_mean']:.8f};"
            f"clip_fraction={result.metrics['clip_fraction']:.8f};"
            f"loss_reduction={result.metrics['loss_reduction']};{_environment()}"
        ),
    }


def _run_fused_logp(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        return _blocked_row(args, "fused-logp", "CUDA is not available")

    from rl_engine.kernels.registry import kernel_registry

    device = torch.device("cuda")
    dtype = getattr(torch, args.cuda_dtype)
    batch, logits, reference_logps, old_logps, ref_logps = _make_inputs(
        args,
        device=device,
        dtype=dtype,
    )
    op = kernel_registry.get_op("logp")
    if op.__class__.__name__ != "FusedLogpGenericOp":
        return _blocked_row(
            args,
            "fused-logp",
            f"fused logp backend is unavailable, got {op.__class__.__name__}",
        )

    config = GRPOConfig(
        clip_epsilon=args.clip_epsilon,
        kl_beta=args.kl_beta,
        advantage_eps=args.advantage_eps,
        loss_reduction=args.loss_reduction,
    )
    started = time.perf_counter()
    candidate_logps = op(logits, batch.token_ids).float()
    candidate_logps = candidate_logps.masked_fill(~batch.completion_mask, 0.0)
    reference_result = compute_grpo_loss(
        reference_logps,
        old_logps,
        ref_logps,
        batch.rewards,
        batch.completion_mask,
        num_groups=args.num_prompts,
        config=config,
    )
    candidate_result = compute_grpo_loss(
        candidate_logps,
        old_logps,
        ref_logps,
        batch.rewards,
        batch.completion_mask,
        num_groups=args.num_prompts,
        config=config,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    logp_drift = summarize_kernel_drift(
        candidate_logps,
        reference_logps,
        batch.completion_mask,
    )
    loss_drift = abs(
        float(candidate_result.loss.detach().cpu().item())
        - float(reference_result.loss.detach().cpu().item())
    )
    advantage_drift = summarize_kernel_drift(
        candidate_result.advantages,
        reference_result.advantages,
        batch.completion_mask,
    )

    return {
        "timestamp": _timestamp(),
        "candidate": args.candidate,
        "mode": "fused-logp",
        "stage": "evaluation",
        "num_prompts": args.num_prompts,
        "samples_per_prompt": args.samples_per_prompt,
        "completion_len": args.completion_len,
        "vocab_size": args.vocab_size,
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "advantage_max_abs_error": f"{advantage_drift['max_abs_error']:.8f}",
        "loss_max_abs_error": f"{loss_drift:.8f}",
        "active_tokens": int(candidate_result.metrics["active_tokens"]),
        "status": "pass",
        "notes": (
            f"elapsed_ms={elapsed_ms:.4f};"
            f"logp_max_abs_error={logp_drift['max_abs_error']:.8f};"
            f"loss={candidate_result.metrics['loss']:.8f};"
            f"loss_reduction={candidate_result.metrics['loss_reduction']};{_environment()}"
        ),
    }


def _blocked_row(args: argparse.Namespace, mode: str, reason: str) -> dict[str, Any]:
    return {
        "timestamp": _timestamp(),
        "candidate": args.candidate,
        "mode": mode,
        "stage": "evaluation",
        "num_prompts": args.num_prompts,
        "samples_per_prompt": args.samples_per_prompt,
        "completion_len": args.completion_len,
        "vocab_size": args.vocab_size,
        "device": args.device,
        "dtype": args.dtype,
        "status": "blocked",
        "notes": f"{reason};{_environment()}",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GRPO objective benchmark")
    parser.add_argument("--mode", choices=["reference", "fused-logp"], default="reference")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--num-prompts", type=int, default=4)
    parser.add_argument("--samples-per-prompt", type=int, default=4)
    parser.add_argument("--prompt-len", type=int, default=8)
    parser.add_argument("--completion-len", type=int, default=16)
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--valid-density", type=float, default=0.85)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--kl-beta", type=float, default=0.01)
    parser.add_argument("--advantage-eps", type=float, default=1e-8)
    parser.add_argument(
        "--loss-reduction",
        choices=["global_token_mean", "sequence_mean"],
        default="global_token_mean",
    )
    parser.add_argument("--seed", type=int, default=16)
    parser.add_argument("--candidate", default="issue-16-candidate-1")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--cuda-dtype", default="float16")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "task-workspace/issues/issue_16/benchmark.csv",
    )
    parser.add_argument("--json-output", type=Path, default=None)
    args = parser.parse_args()

    if args.smoke:
        args.num_prompts = min(args.num_prompts, 2)
        args.samples_per_prompt = min(args.samples_per_prompt, 3)
        args.prompt_len = min(args.prompt_len, 4)
        args.completion_len = min(args.completion_len, 8)
        args.vocab_size = min(args.vocab_size, 128)
    return args


def main() -> None:
    args = parse_args()
    if args.mode == "reference":
        row = _run_reference(args)
    else:
        row = _run_fused_logp(args)
    _append_row(args.output, row)
    _write_json(args.json_output, row)
    print(json.dumps(row, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
