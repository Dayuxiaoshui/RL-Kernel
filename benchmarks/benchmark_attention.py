# File: benchmarks/benchmark_attention.py
import pandas as pd
import torch
import triton
from tabulate import tabulate

from rl_engine.kernels.ops.cuda.attention.prefix_shared_attn import PrefixSharedAttentionOp


def run_benchmark():
    bs = 1
    G = 64
    len_q = 512
    dim = 128

    len_kvs = [1024, 2048, 4096, 8192, 16384]

    print("Benchmarking GRPO Prefix-Shared Attention")
    print(f"Fixed Shapes: Batch={bs}, Group(Response)={G}, Query_Len={len_q}, Head_Dim={dim}\n")

    prefix_shared_sdpa = PrefixSharedAttentionOp()
    results = []

    for len_kv in len_kvs:
        q = torch.randn(bs, G, len_q, dim, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(bs, len_kv, dim, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(bs, len_kv, dim, dtype=torch.bfloat16, device="cuda")

        k_exp = k.unsqueeze(1).expand(-1, G, -1, -1).reshape(bs * G, 1, len_kv, dim).contiguous()
        v_exp = v.unsqueeze(1).expand(-1, G, -1, -1).reshape(bs * G, 1, len_kv, dim).contiguous()
        q_res = q.view(bs * G, 1, len_q, dim)

        for _ in range(5):
            _ = torch.nn.functional.scaled_dot_product_attention(q_res, k_exp, v_exp)
            _ = prefix_shared_sdpa(q, k, v)

        native_ms = triton.testing.do_bench(
            lambda: torch.nn.functional.scaled_dot_product_attention(q_res, k_exp, v_exp),
            return_mode="median",
        )

        custom_ms = triton.testing.do_bench(
            lambda: prefix_shared_sdpa(q, k, v), return_mode="median"
        )

        speedup = native_ms / custom_ms
        reduction = (native_ms - custom_ms) / native_ms * 100

        flops = 4 * bs * G * len_q * len_kv * dim
        native_tflops = (flops / 1e12) / (native_ms / 1000)
        custom_tflops = (flops / 1e12) / (custom_ms / 1000)

        results.append(
            {
                "Prompt Len": len_kv,
                "Native (ms)": f"{native_ms:.3f}",
                "RL-Kernel (ms)": f"{custom_ms:.3f}",
                "Native TFLOPS": f"{native_tflops:.1f}",
                "RL-Kernel TFLOPS": f"{custom_tflops:.1f}",
                "Speedup": f"{speedup:.2f}x",
                "Time Saved": f"{reduction:.1f}%",
            }
        )

    df = pd.DataFrame(results)
    print(tabulate(df, headers="keys", tablefmt="pretty", stralign="center", showindex=False))


if __name__ == "__main__":
    run_benchmark()
