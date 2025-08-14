import torch
import triton
import triton.language as tl

CONFIGS = [
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 256}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 256}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2),
]


@triton.autotune(
    configs=CONFIGS,
    key=[
        "M",
        "N",
    ],
)
@triton.jit
def bias_gelu_dropout_residual_fwd(
    x_ptr,
    bias_ptr,
    residual_ptr,
    y_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    stride_xm,
    stride_xn,
    stride_rm,
    stride_rn,
    stride_ym,
    stride_yn,
    stride_bias,
    p,
    seed,
    training: tl.constexpr,
    has_bias: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = off_m < M
    mask_n = off_n < N
    mm = off_m[:, None]
    nn = off_n[None, :]
    mask = mask_m[:, None] & mask_n[None, :]

    x = tl.load(x_ptr + mm * stride_xm + nn * stride_xn, mask=mask, other=0.0)
    if has_bias:
        b = tl.load(bias_ptr + nn * stride_bias, mask=mask_n[None, :], other=0.0)
        z = x + b
    else:
        z = x

    # GELU tanh approximation, with tanh(x) = 2 * sigmoid(2x) - 1 (see README for details)
    sqrt_2_over_pi = 0.7978845608
    c = sqrt_2_over_pi * (z + 0.044715 * z * z * z)
    g = z * (tl.sigmoid(2.0 * c))

    # Dropout (inverted) only if training and p>0
    if training and p > 0.0:
        # logical linear index i*N + j → layout-agnostic deterministic mask
        linear_idx = (mm * N + nn).to(tl.int32)
        r = tl.rand(seed, linear_idx)
        keep = r > p
        scale = 1.0 / (1.0 - p)
        y_mid = tl.where(keep, g * scale, 0.0)
    else:
        y_mid = g

    res = tl.load(residual_ptr + mm * stride_rm + nn * stride_rn, mask=mask, other=0.0)
    out = y_mid + res
    tl.store(y_ptr + mm * stride_ym + nn * stride_yn, out, mask=mask)


def triton_forward(
    x: torch.Tensor,
    bias: torch.Tensor | None,
    residual: torch.Tensor,
    p: float = 0.0,
    training: bool = True,
    seed: int | None = None,
):
    """ """
    assert residual.shape == x.shape

    # ---- normalize shapes to [M, N] with N = last dim ----
    N = x.shape[-1]
    M = x.numel() // N
    if bias is not None:
        assert bias.ndim == 1 and bias.shape[0] == N

    # reshape (view if possible, else copy). make last-dim contiguous for fast kernel
    x2 = x.reshape(M, N).contiguous()
    r2 = residual.reshape(M, N).contiguous()

    x32 = x2.to(torch.float32)
    r32 = r2.to(torch.float32)
    b32 = (
        bias.to(torch.float32).contiguous()
        if bias is not None
        else torch.empty(1, dtype=torch.float32, device=x.device)
    )
    y32 = torch.empty_like(x32, dtype=torch.float32)

    if seed is None:
        seed = torch.randint(0, 2**31 - 1, (), device=x.device).item()

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )

    # actual kernel launch
    bias_gelu_dropout_residual_fwd[grid](
        x32,
        b32,
        r32,
        y32,
        M,
        N,
        x32.stride(0),
        x32.stride(1),
        r32.stride(0),
        r32.stride(1),
        y32.stride(0),
        y32.stride(1),
        (b32.stride(0) if bias is not None else 0),
        float(p),
        seed,
        training,
        bias is not None,
    )

    # convert to original shape & dtype
    return y32.to(x.dtype).reshape(x.shape)
