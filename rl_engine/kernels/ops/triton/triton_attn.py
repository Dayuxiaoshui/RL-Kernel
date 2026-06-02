import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(
    Q, K, V, sm_scale,
    L, M, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    
    # 初始化偏移量
    qvk_offset = off_hz * stride_qh
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(BLOCK_DMODEL, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_vn, stride_vk),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0)
    )

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    
    # 初始化统计量
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    
    # 加载 Q
    q = tl.load(Q_block_ptr)
    
    # 确定 K 和 V 的循环范围
    lo = 0
    hi = (start_m + 1) * BLOCK_M if IS_CAUSAL else N_CTX
    
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # 计算 QK^T
        k = tl.load(K_block_ptr)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk *= sm_scale
        
        # Block-wise Causal Masking
        if IS_CAUSAL:
            qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))
            
        # 计算 softmax 的 max 和 sum
        m_ij = tl.max(qk, 1)
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(qk - m_i_new[:, None])
        l_i_new = alpha * l_i + tl.sum(beta, 1)
        
        # 更新 scale 和 V 的累加
        p_scale = beta / l_i_new[:, None]
        acc_scale = l_i / l_i_new * alpha
        
        acc = acc * acc_scale[:, None]
        v = tl.load(V_block_ptr)
        p = p_scale.to(v.dtype)
        acc += tl.dot(p, v)
        
        # 更新统计量
        l_i = l_i_new
        m_i = m_i_new
        
        # 移动指针到下一个 block
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

    # 写入输出
    O_block_ptr = tl.make_block_ptr(
        base=Out + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    acc = acc.to(Out.dtype.element_ty)
    tl.store(O_block_ptr, acc)
    
    # 保存 l 和 m 用于反向传播
    off_zh = off_hz * N_CTX
    l_ptrs = L + off_zh + offs_m
    m_ptrs = M + off_zh + offs_m
    tl.store(l_ptrs, l_i)
    tl.store(m_ptrs, m_i)


@triton.jit
def _bwd_preprocess(
    Out, DO,
    Delta,
    stride_oz, stride_oh, stride_om, stride_on,
    stride_doz, stride_doh, stride_dom, stride_don,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr, D_HEAD: tl.constexpr,
):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_hz = tl.program_id(1)
    
    o_offset = off_hz * stride_oh
    do_offset = off_hz * stride_doh
    
    # 加载 Out 和 dOut
    O_block_ptr = tl.make_block_ptr(
        base=Out + o_offset,
        shape=(N_CTX, D_HEAD),
        strides=(stride_om, stride_on),
        offsets=(tl.program_id(0) * BLOCK_M, 0),
        block_shape=(BLOCK_M, D_HEAD),
        order=(1, 0)
    )
    DO_block_ptr = tl.make_block_ptr(
        base=DO + do_offset,
        shape=(N_CTX, D_HEAD),
        strides=(stride_dom, stride_don),
        offsets=(tl.program_id(0) * BLOCK_M, 0),
        block_shape=(BLOCK_M, D_HEAD),
        order=(1, 0)
    )
    
    o = tl.load(O_block_ptr)
    do = tl.load(DO_block_ptr).to(o.dtype)
    
    # 计算 Delta: row sum of (Out * dOut)
    delta = tl.sum(o * do, axis=1)
    
    # 存储 Delta
    off_zh = off_hz * N_CTX
    tl.store(Delta + off_zh + off_m, delta)


@triton.jit
def _bwd_kernel(
    Q, K, V, sm_scale,
    Out, DO, DQ, DK, DV,
    L, M, Delta,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    start_n = tl.program_id(0)
    off_hz = tl.program_id(1)
    
    # 偏移量计算
    qvk_offset = off_hz * stride_qh
    
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_kn, stride_kk),
        offsets=(start_n * BLOCK_N, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0)
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_vn, stride_vk),
        offsets=(start_n * BLOCK_N, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0)
    )
    
    # 初始化 dK 和 dV 累加器 (fp32高精度累加)
    dk = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    
    k = tl.load(K_block_ptr)
    v = tl.load(V_block_ptr)
    
    lo = start_n * BLOCK_N if IS_CAUSAL else 0
    hi = N_CTX
    
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(lo, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    DO_block_ptr = tl.make_block_ptr(
        base=DO + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=(lo, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    
    # 用于原子加的地址偏移量
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    for start_m in range(lo, hi, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        offs_m = start_m + tl.arange(0, BLOCK_M)
        
        q = tl.load(Q_block_ptr)
        do = tl.load(DO_block_ptr)
        
        # 加载前向统计量
        off_zh = off_hz * N_CTX
        m_i = tl.load(M + off_zh + offs_m)
        l_i = tl.load(L + off_zh + offs_m)
        delta = tl.load(Delta + off_zh + offs_m)
        
        # 计算 QK^T
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))
        qk *= sm_scale
        
        if IS_CAUSAL:
            qk = tl.where(offs_m[:, None] >= offs_n[None, :], qk, float("-inf"))
            
        p = tl.exp(qk - m_i[:, None]) / l_i[:, None]
        
        # 计算 dV = P^T * dO
        dv += tl.dot(tl.trans(p.to(do.dtype)), do)
        
        # 计算 dP = dO * V^T
        dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        dp += tl.dot(do, tl.trans(v))
        
        # 计算 dS = P * (dP - Delta)
        ds = p * (dp - delta[:, None]) * sm_scale
        
        # =======================================================
        # 核心修复区：计算 dQ 并使用 atomic_add 和严格的类型转换
        # =======================================================
        dq_val = tl.dot(ds.to(q.dtype), k)  # 这里计算出来的 dq_val 是 fp32
        
        # 计算原始内存地址 (因为 atomic_add 不支持 block_ptr)
        dq_ptrs = DQ + qvk_offset + (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk)
        
        # 将 fp32 强转回 fp16 并原子相加
        tl.atomic_add(dq_ptrs, dq_val.to(q.dtype)) 
        
        # 计算 dK = dS^T * Q
        dk += tl.dot(tl.trans(ds.to(q.dtype)), q)
        
        # 移动指针
        Q_block_ptr = tl.advance(Q_block_ptr, (BLOCK_M, 0))
        DO_block_ptr = tl.advance(DO_block_ptr, (BLOCK_M, 0))
        
    # 写入 dK, dV (这里我们将累加完毕的 fp32 转换回外层 tensor 的类型)
    DK_block_ptr = tl.make_block_ptr(
        base=DK + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_kn, stride_kk),
        offsets=(start_n * BLOCK_N, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0)
    )
    DV_block_ptr = tl.make_block_ptr(
        base=DV + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_vn, stride_vk),
        offsets=(start_n * BLOCK_N, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0)
    )
    tl.store(DK_block_ptr, dk.to(k.dtype))
    tl.store(DV_block_ptr, dv.to(v.dtype))


class _TritonAttention(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale):
        # 形状检查 [batch, num_heads, seq_len, head_dim]
        # Triton tutorial standard layout requires specific contiguity
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Lq == Lk == Lv
        assert Lq in {16, 32, 64, 128, 256}
        
        ctx.sm_scale = sm_scale
        ctx.causal = causal
        
        batch, heads, seq_len, head_dim = q.shape
        
        # 准备输出和中间变量
        out = torch.empty_like(q)
        M = torch.empty((batch, heads, seq_len), device=q.device, dtype=torch.float32)
        L = torch.empty((batch, heads, seq_len), device=q.device, dtype=torch.float32)
        
        # 针对前向的 Block 划分配置
        BLOCK_M = 64
        BLOCK_N = 64 if head_dim > 64 else 128
        
        grid = (triton.cdiv(seq_len, BLOCK_M), batch * heads, 1)
        
        _fwd_kernel[grid](q,k,v,sm_scale,L,M,out,q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            batch, heads, seq_len,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=head_dim,
            IS_CAUSAL=causal,
            num_warps=4, num_stages=2,
        )
        
        ctx.save_for_backward(q, k, v, out, L, M)
        return out

    @staticmethod
    def backward(ctx, do):
        q, k, v, out, L, M = ctx.saved_tensors
        
        # 必须保证 do 连续
        do = do.contiguous()
        dq = torch.zeros_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        
        batch, heads, seq_len, head_dim = q.shape
        delta = torch.empty_like(L)
        
        BLOCK_M = 64
        BLOCK_N = 64
        
        # Preprocess grid
        grid_prep = (triton.cdiv(seq_len, BLOCK_M), batch * heads)
        _bwd_preprocess[grid_prep](out,do,delta,out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            batch, heads, seq_len,
            BLOCK_M=BLOCK_M, D_HEAD=head_dim,
        )
        
        # Backward grid
        grid_bwd = (triton.cdiv(seq_len, BLOCK_N), batch * heads, 1)
        _bwd_kernel[grid_bwd](q,k,v,ctx.sm_scale,out,do,dq,dk,dv,L,M,delta,q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            batch, heads, seq_len,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=head_dim,
            IS_CAUSAL=ctx.causal,
            num_warps=4, num_stages=1,
        )
        
        return dq, dk, dv, None, None


def triton_flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = True, sm_scale: float = None):
    """
    通用后备的 Triton FlashAttention (支持 Forward / Backward)
    
    Args:
        q, k, v: Tensors of shape [batch, num_heads, seq_len, head_dim]. 建议转为连续内存(.contiguous())。
        causal: 是否开启因果掩码 (Causal Masking)。
        sm_scale: Softmax 的缩放因子，默认 1.0 / sqrt(head_dim)。
    Returns:
        Attention Output tensor of shape [batch, num_heads, seq_len, head_dim]
    """
    if sm_scale is None:
        sm_scale = 1.0 / (q.shape[-1] ** 0.5)
        
    return _TritonAttention.apply(q, k, v, causal, sm_scale)
