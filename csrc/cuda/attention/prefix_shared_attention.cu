// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors

#include <cuda_bf16.h>
#include <cstdint>
#include <float.h>
#include <iostream>

using nv_bfloat16 = __nv_bfloat16;
using nv_bfloat162 = __nv_bfloat162;

#define CUDA_CHECK(x)                                                                 \
  {                                                                                   \
    auto error = x;                                                                   \
    if (error != cudaSuccess) {                                                       \
      std::cerr << "CUDA error - L" << __LINE__ << ": " << cudaGetErrorString(error) << std::endl; \
      exit(1);                                                                        \
    }                                                                                 \
  }

inline constexpr int WARP_SIZE = 32;

__device__ __host__ constexpr int cdiv(int a, int b) { return (a + b - 1) / b; }

// NOTE: stride in bytes
template <int STRIDE>
__device__ inline uint32_t swizzle(uint32_t index) {
  if constexpr (STRIDE == 16) return index;
  uint32_t row_idx = (index / STRIDE) % 8;
  uint32_t bits_to_xor = row_idx / max(64 / STRIDE, 1);
  return index ^ (bits_to_xor << 4);
}

template <int HEIGHT, int WIDTH, int TB_SIZE>
__device__ inline void global_to_shared_swizzle(uint32_t dst, const nv_bfloat16 *src, int src_stride, int tid) {
  constexpr int num_elems = 16 / sizeof(nv_bfloat16);
  constexpr int num_iters = HEIGHT * WIDTH / (TB_SIZE * num_elems);

  for (int iter = 0; iter < num_iters; iter++) {
    const int idx = (iter * TB_SIZE + tid) * num_elems;
    const int row = idx / WIDTH;
    const int col = idx % WIDTH;

    const uint32_t dst_addr = swizzle<WIDTH * sizeof(nv_bfloat16)>(dst + (row * WIDTH + col) * sizeof(nv_bfloat16));
    const nv_bfloat16 *src_addr = src + (row * src_stride + col);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst_addr), "l"(src_addr));
  }
}

__device__ inline void ldmatrix_x4(uint32_t regs[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
              : "=r"(regs[0]), "=r"(regs[1]), "=r"(regs[2]), "=r"(regs[3])
              : "r"(addr));
}

__device__ inline void ldmatrix_x4_trans(uint32_t regs[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
              : "=r"(regs[0]), "=r"(regs[1]), "=r"(regs[2]), "=r"(regs[3])
              : "r"(addr));
}

__device__ inline void mma_m16n8k16(uint32_t A[4], uint32_t B[2], float D[4]) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
              "{%0, %1, %2, %3}, "
              "{%4, %5, %6, %7}, "
              "{%8, %9}, "
              "{%10, %11, %12, %13};"
              : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
              : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
                "r"(B[0]), "r"(B[1]),
                "f"(D[0]), "f"(D[1]), "f"(D[2]), "f"(D[3]));
}

template <typename T, typename... Args>
void launch_kernel(T *kernel, dim3 grid, int block_size, int smem_size, Args... args) {
  if (smem_size > 48000) {
    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
  }
  kernel<<<grid, block_size, smem_size>>>(args...);
  CUDA_CHECK(cudaGetLastError());
}

// Prefix-Shared Fused Attention Kernel

template<int BLOCK_Q, int BLOCK_KV, int DIM, int NUM_WARPS>
__launch_bounds__(NUM_WARPS * WARP_SIZE)
__global__
void prefix_shared_attention_kernel(
  const nv_bfloat16 *Q,  // [bs, G, len_q, DIM]
  const nv_bfloat16 *K,  // [bs, len_kv, DIM]
  const nv_bfloat16 *V,  // [bs, len_kv, DIM]
  nv_bfloat16 *O,        // [bs, G, len_q, DIM]
  int bs, int G, int len_q, int len_kv) {

  constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE;
  const int tid = threadIdx.x;
  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE;

  const int q_block_id = blockIdx.x;
  const int g_id = blockIdx.y;
  const int bs_id = blockIdx.z;

  Q += (bs_id * G * len_q + g_id * len_q + q_block_id * BLOCK_Q) * DIM;
  O += (bs_id * G * len_q + g_id * len_q + q_block_id * BLOCK_Q) * DIM;

  K += bs_id * len_kv * DIM;
  V += bs_id * len_kv * DIM;

  extern __shared__ nv_bfloat16 smem[];
  const uint32_t Q_smem = __cvta_generic_to_shared(smem);
  const uint32_t K_smem = Q_smem;
  const uint32_t V_smem = K_smem + 2 * BLOCK_KV * DIM * sizeof(nv_bfloat16);

  constexpr int WARP_Q = BLOCK_Q / NUM_WARPS;
  constexpr int MMA_M = 16;
  constexpr int MMA_N = 8;
  constexpr int MMA_K = 16;

  uint32_t Q_rmem[WARP_Q / MMA_M][DIM / MMA_K][4];
  uint32_t K_rmem[BLOCK_KV / MMA_N][DIM / MMA_K][2];
  uint32_t P_rmem[WARP_Q / MMA_M][BLOCK_KV / MMA_K][4];
  uint32_t V_rmem[BLOCK_KV / MMA_K][DIM / MMA_N][2];

  float O_rmem[WARP_Q / MMA_M][DIM / MMA_N][4] = {};
  const float softmax_scale = rsqrtf(static_cast<float>(DIM));
  float rowmax[WARP_Q / MMA_M][2];
  float rowsumexp[WARP_Q / MMA_M][2] = {};

  for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++) {
    rowmax[mma_id_q][0] = -FLT_MAX;
    rowmax[mma_id_q][1] = -FLT_MAX;
  }

  uint32_t Q_smem_thread, K_smem_thread, V_smem_thread;
  {
    const int row_off = warp_id * WARP_Q + (lane_id % 16);
    const int col_off = lane_id / 16 * 8;
    Q_smem_thread = swizzle<DIM * sizeof(nv_bfloat16)>(Q_smem + (row_off * DIM + col_off) * sizeof(nv_bfloat16));
  }
  {
    const int row_off = lane_id % 8;
    const int col_off = lane_id / 8 * 8;
    K_smem_thread = swizzle<DIM * sizeof(nv_bfloat16)>(K_smem + (row_off * DIM + col_off) * sizeof(nv_bfloat16));
  }
  {
    const int row_off = lane_id % 16;
    const int col_off = lane_id / 16 * 8;
    V_smem_thread = swizzle<DIM * sizeof(nv_bfloat16)>(V_smem + (row_off * DIM + col_off) * sizeof(nv_bfloat16));
  }

  global_to_shared_swizzle<BLOCK_Q, DIM, TB_SIZE>(Q_smem, Q, DIM, tid);
  asm volatile("cp.async.commit_group;");
  asm volatile("cp.async.wait_all;");
  __syncthreads();

  for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
    for (int mma_id_d = 0; mma_id_d < DIM / MMA_K; mma_id_d++) {
      uint32_t addr = Q_smem_thread;
      addr += mma_id_q * MMA_M * DIM * sizeof(nv_bfloat16);
      addr ^= mma_id_d * MMA_K * sizeof(nv_bfloat16);
      ldmatrix_x4(Q_rmem[mma_id_q][mma_id_d], addr);
    }
  __syncthreads();

  const int num_kv_iter = cdiv(len_kv, BLOCK_KV);

  auto load_K = [&](int kv_id) {
    if (kv_id < num_kv_iter) {
      const uint32_t dst = K_smem + (kv_id % 2) * (BLOCK_KV * DIM * sizeof(nv_bfloat16));
      global_to_shared_swizzle<BLOCK_KV, DIM, TB_SIZE>(dst, K, DIM, tid);
      K += BLOCK_KV * DIM;
    }
    asm volatile("cp.async.commit_group;");
  };

  auto load_V = [&](int kv_id) {
    if (kv_id < num_kv_iter) {
        const uint32_t dst = V_smem;
        global_to_shared_swizzle<BLOCK_KV, DIM, TB_SIZE>(dst, V, DIM, tid);
        V += BLOCK_KV * DIM;
    }
    asm volatile("cp.async.commit_group;");
  };

  load_K(0);

  for (int kv_id = 0; kv_id < num_kv_iter; kv_id++) {
    float S_rmem[WARP_Q / MMA_M][BLOCK_KV / MMA_N][4] = {};

    __syncthreads();
    load_V(kv_id);

    asm volatile("cp.async.wait_group 1;");
    __syncthreads();

    for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++)
      for (int mma_id_d = 0; mma_id_d < DIM / MMA_K; mma_id_d += 2) {
        uint32_t addr = K_smem_thread + (kv_id % 2) * (BLOCK_KV * DIM * sizeof(nv_bfloat16));
        addr += mma_id_kv * MMA_N * DIM * sizeof(nv_bfloat16);
        addr ^= mma_id_d * MMA_K * sizeof(nv_bfloat16);
        ldmatrix_x4(K_rmem[mma_id_kv][mma_id_d], addr);
      }

    for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
      for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++)
        for (int mma_id_d = 0; mma_id_d < DIM / MMA_K; mma_id_d++)
          mma_m16n8k16(Q_rmem[mma_id_q][mma_id_d], K_rmem[mma_id_kv][mma_id_d], S_rmem[mma_id_q][mma_id_kv]);

    load_K(kv_id + 1);

    for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++) {
      for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++)
        for (int reg_id = 0; reg_id < 4; reg_id++)
          S_rmem[mma_id_q][mma_id_kv][reg_id] *= softmax_scale;

      float this_rowmax[2];
      for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
        float *regs = S_rmem[mma_id_q][mma_id_kv];
        if (mma_id_kv == 0) {
          this_rowmax[0] = max(regs[0], regs[1]);
          this_rowmax[1] = max(regs[2], regs[3]);
        } else {
          this_rowmax[0] = max(this_rowmax[0], max(regs[0], regs[1]));
          this_rowmax[1] = max(this_rowmax[1], max(regs[2], regs[3]));
        }
      }

      this_rowmax[0] = max(this_rowmax[0], __shfl_xor_sync(0xFFFF'FFFF, this_rowmax[0], 1));
      this_rowmax[0] = max(this_rowmax[0], __shfl_xor_sync(0xFFFF'FFFF, this_rowmax[0], 2));
      this_rowmax[1] = max(this_rowmax[1], __shfl_xor_sync(0xFFFF'FFFF, this_rowmax[1], 1));
      this_rowmax[1] = max(this_rowmax[1], __shfl_xor_sync(0xFFFF'FFFF, this_rowmax[1], 2));

      this_rowmax[0] = max(this_rowmax[0], rowmax[mma_id_q][0]);
      this_rowmax[1] = max(this_rowmax[1], rowmax[mma_id_q][1]);

      float rescale[2];
      rescale[0] = __expf(rowmax[mma_id_q][0] - this_rowmax[0]);
      rescale[1] = __expf(rowmax[mma_id_q][1] - this_rowmax[1]);
      for (int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d++) {
        O_rmem[mma_id_q][mma_id_d][0] *= rescale[0];
        O_rmem[mma_id_q][mma_id_d][1] *= rescale[0];
        O_rmem[mma_id_q][mma_id_d][2] *= rescale[1];
        O_rmem[mma_id_q][mma_id_d][3] *= rescale[1];
      }

      rowmax[mma_id_q][0] = this_rowmax[0];
      rowmax[mma_id_q][1] = this_rowmax[1];

      float this_rowsumexp[2];
      for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
        float *regs = S_rmem[mma_id_q][mma_id_kv];
        regs[0] = __expf(regs[0] - rowmax[mma_id_q][0]);
        regs[1] = __expf(regs[1] - rowmax[mma_id_q][0]);
        regs[2] = __expf(regs[2] - rowmax[mma_id_q][1]);
        regs[3] = __expf(regs[3] - rowmax[mma_id_q][1]);

        if (mma_id_kv == 0) {
          this_rowsumexp[0] = regs[0] + regs[1];
          this_rowsumexp[1] = regs[2] + regs[3];
        } else {
          this_rowsumexp[0] += regs[0] + regs[1];
          this_rowsumexp[1] += regs[2] + regs[3];
        }

        nv_bfloat162 *this_P_rmem = reinterpret_cast<nv_bfloat162 *>(P_rmem[mma_id_q][mma_id_kv / 2]);
        this_P_rmem[(mma_id_kv % 2) * 2]     = __float22bfloat162_rn({regs[0], regs[1]});
        this_P_rmem[(mma_id_kv % 2) * 2 + 1] = __float22bfloat162_rn({regs[2], regs[3]});
      }

      this_rowsumexp[0] += __shfl_xor_sync(0xFFFF'FFFF, this_rowsumexp[0], 1);
      this_rowsumexp[0] += __shfl_xor_sync(0xFFFF'FFFF, this_rowsumexp[0], 2);
      this_rowsumexp[1] += __shfl_xor_sync(0xFFFF'FFFF, this_rowsumexp[1], 1);
      this_rowsumexp[1] += __shfl_xor_sync(0xFFFF'FFFF, this_rowsumexp[1], 2);

      rowsumexp[mma_id_q][0] = rowsumexp[mma_id_q][0] * rescale[0] + this_rowsumexp[0];
      rowsumexp[mma_id_q][1] = rowsumexp[mma_id_q][1] * rescale[1] + this_rowsumexp[1];
    }

    asm volatile("cp.async.wait_group 1;");
    __syncthreads();

    for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_K; mma_id_kv++)
      for (int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d += 2) {
        uint32_t addr = V_smem_thread;
        addr += mma_id_kv * MMA_K * DIM * sizeof(nv_bfloat16);
        addr ^= mma_id_d * MMA_N * sizeof(nv_bfloat16);
        ldmatrix_x4_trans(V_rmem[mma_id_kv][mma_id_d], addr);
      }

    for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
      for (int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d++)
        for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_K; mma_id_kv++)
          mma_m16n8k16(P_rmem[mma_id_q][mma_id_kv], V_rmem[mma_id_kv][mma_id_d], O_rmem[mma_id_q][mma_id_d]);
  }

  for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
    for (int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d++) {
      const int row = warp_id * WARP_Q + mma_id_q * MMA_M + (lane_id / 4);
      const int col = mma_id_d * MMA_N + (lane_id % 4) * 2;

      float *regs = O_rmem[mma_id_q][mma_id_d];
      regs[0] /= rowsumexp[mma_id_q][0];
      regs[1] /= rowsumexp[mma_id_q][0];
      regs[2] /= rowsumexp[mma_id_q][1];
      regs[3] /= rowsumexp[mma_id_q][1];

      reinterpret_cast<nv_bfloat162 *>(O + (row + 0) * DIM + col)[0] = __float22bfloat162_rn({regs[0], regs[1]});
      reinterpret_cast<nv_bfloat162 *>(O + (row + 8) * DIM + col)[0] = __float22bfloat162_rn({regs[2], regs[3]});
    }
}

// Host API
void prefix_shared_attention_forward(
  const __nv_bfloat16 *Q, const __nv_bfloat16 *K, const __nv_bfloat16 *V, __nv_bfloat16 *O,
  int bs, int G, int len_q, int len_kv, int dim) {

  if (dim != 128) { std::cerr << "Only dim=128 supported." << std::endl; exit(1); }

  const int BLOCK_Q = 64;
  const int BLOCK_KV = 64;
  const int DIM = 128;
  const int NUM_WARPS = 4;

  dim3 grid(cdiv(len_q, BLOCK_Q), G, bs);
  const int TB_SIZE = NUM_WARPS * WARP_SIZE;
  const int smem_size = max(BLOCK_Q, BLOCK_KV * 3) * DIM * sizeof(nv_bfloat16);

  auto kernel = prefix_shared_attention_kernel<BLOCK_Q, BLOCK_KV, DIM, NUM_WARPS>;
  launch_kernel(kernel, grid, TB_SIZE, smem_size, Q, K, V, O, bs, G, len_q, len_kv);
}
