# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

from typing import Optional

import torch

from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE
from rl_engine.utils.logger import logger

# Backward token-chunk target (mirrors the Triton op): keep peak backward memory
# at ~chunk*V instead of N*V.
_BWD_CHUNK_ELEMS = 1 << 24

# Hidden-dim slice the SM90 forward streams per TMA load; D must be a multiple of
# it (mirrors `constexpr int BK` in csrc/cuda/fused_linear_logp_sm90.cu).
_SM90_BK = 32


def _sm90_supported(hidden: torch.Tensor, lm_head_weight: torch.Tensor) -> bool:
    """Whether the bf16 TMA+MMA forward can run these inputs directly."""
    return (
        hidden.is_cuda
        and hidden.dtype == torch.bfloat16
        and lm_head_weight.dtype == torch.bfloat16
        and hidden.size(-1) % _SM90_BK == 0
    )


def _fallback_op():
    """Portable op for inputs the SM90 forward cannot take (fp32/fp16, or a hidden
    dim not divisible by the kernel's K slice). Prefers Triton, else native."""
    try:
        from rl_engine.kernels.ops.triton.loss.linear_logp import TritonLinearLogpOp

        return TritonLinearLogpOp()
    except Exception:  # pragma: no cover - Triton missing
        from rl_engine.kernels.ops.pytorch.loss.linear_logp import NativeLinearLogpOp

        return NativeLinearLogpOp()


class _FusedLinearLogpSM90Function(torch.autograd.Function):
    """SM90 TMA+WGMMA fused forward + Liger-style chunked backward.

    The forward calls the compiled ``_C.fused_linear_logp_sm90`` kernel (logits
    never materialized). The backward reuses the deterministic chunked cuBLAS
    path so gradients flow into ``hidden``, ``lm_head_weight`` and ``bias``.
    """

    @staticmethod
    def forward(ctx, hidden, lm_head_weight, bias, target_ids):
        hidden_2d = hidden.reshape(-1, hidden.size(-1)).contiguous()
        weight = lm_head_weight.contiguous()
        target_1d = (
            target_ids.reshape(-1).to(device=hidden_2d.device, dtype=torch.int32).contiguous()
        )
        logp, _lse = _C.fused_linear_logp_sm90(hidden_2d, weight, target_1d, bias)

        ctx.save_for_backward(hidden_2d, weight, bias if bias is not None else hidden_2d, target_1d)
        ctx.has_bias = bias is not None
        ctx.lead_shape = hidden.shape[:-1]
        ctx.hidden_dtype = hidden.dtype
        ctx.weight_dtype = lm_head_weight.dtype
        ctx.bias_dtype = bias.dtype if bias is not None else None
        return logp.reshape(hidden.shape[:-1])

    @staticmethod
    def backward(ctx, grad_logp):
        hidden_2d, weight, bias_t, target_1d = ctx.saved_tensors
        n, d = hidden_2d.shape
        v = weight.shape[0]
        dt = weight.dtype
        g = grad_logp.reshape(-1).to(torch.float32)

        grad_h = torch.empty_like(hidden_2d, dtype=torch.float32)
        grad_w = torch.zeros(v, d, device=weight.device, dtype=torch.float32)
        grad_b = torch.zeros(v, device=weight.device, dtype=torch.float32) if ctx.has_bias else None

        chunk = max(1, min(n, _BWD_CHUNK_ELEMS // v))
        for i0 in range(0, n, chunk):
            i1 = min(i0 + chunk, n)
            x = hidden_2d[i0:i1]
            logits = torch.matmul(x, weight.t())
            if ctx.has_bias:
                logits = logits + bias_t
            dz = torch.softmax(logits.float(), dim=-1).neg_()
            rows = torch.arange(i1 - i0, device=dz.device)
            dz[rows, target_1d[i0:i1].long()] += 1.0
            dz *= g[i0:i1].unsqueeze(1)

            dz_dt = dz.to(dt)
            grad_h[i0:i1] = torch.matmul(dz_dt, weight).float()
            grad_w += torch.matmul(dz_dt.t(), x).float()
            if ctx.has_bias:
                grad_b += dz.sum(0)

        grad_hidden = grad_h.to(ctx.hidden_dtype).reshape(ctx.lead_shape + (d,))
        grad_weight = grad_w.to(ctx.weight_dtype)
        grad_bias = grad_b.to(ctx.bias_dtype) if ctx.has_bias else None
        return grad_hidden, grad_weight, grad_bias, None


class FusedLinearLogpSM90Op:
    """SM90 (Hopper) TMA+WGMMA fused linear log-prob.

    Computes ``log_softmax(hidden @ W^T + b)[target]`` without materializing the
    ``[N, V]`` logits. Requires the extension built with ``KERNEL_ALIGN_FORCE_SM90=1``
    on an SM90 device; bfloat16 hidden/weight only.
    """

    def __init__(self) -> None:
        if not _EXT_AVAILABLE or not hasattr(_C, "fused_linear_logp_sm90"):
            raise RuntimeError(
                "fused_linear_logp_sm90 is not compiled into the extension. Rebuild with "
                "KERNEL_ALIGN_FORCE_SM90=1 on an SM90 (Hopper) device: 'pip install -e .'"
            )
        logger.info("Successfully linked to precompiled _C.fused_linear_logp_sm90 kernel.")

    def __call__(
        self,
        hidden: torch.Tensor,
        lm_head_weight: torch.Tensor,
        target_ids: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.apply(hidden, lm_head_weight, target_ids, bias)

    def apply(
        self,
        hidden: torch.Tensor,
        lm_head_weight: torch.Tensor,
        target_ids: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if lm_head_weight.size(-1) != hidden.size(-1):
            raise ValueError(
                f"hidden dim {hidden.size(-1)} must match lm_head_weight dim "
                f"{lm_head_weight.size(-1)}"
            )
        # The compiled forward is bf16-only and needs D % BK == 0. For anything
        # else (fp32/fp16 references, awkward hidden dims, CPU tensors) fall back
        # to a portable backend so the op stays a drop-in for any input.
        if not _sm90_supported(hidden, lm_head_weight):
            return _fallback_op()(hidden, lm_head_weight, target_ids, bias)
        return _FusedLinearLogpSM90Function.apply(hidden, lm_head_weight, bias, target_ids)
