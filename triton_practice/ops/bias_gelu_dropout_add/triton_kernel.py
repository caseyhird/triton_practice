import torch
import triton
import triton.language as tl


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

    # GELU (tanh approximation)
    c = 0.7978845608 * (z + 0.044715 * z * z * z)
    g = 0.5 * z * (1.0 + tl.tanh(c))

    # Dropout (inverted) only if training and p>0
    if training:
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


def forward(
    x: torch.Tensor,
    bias: torch.Tensor | None,
    residual: torch.Tensor,
    p: float = 0.0,
    training: bool = True,
    *,
    block_m=64,
    block_n=128,
    num_warps=4,
    num_stages=2,
    seed: int | None = None,
):
    """ """
    assert x.ndim == 2 and residual.shape == x.shape
    if bias is not None:
        assert bias.ndim == 1 and bias.shape[0] == x.shape[1]

    M, N = x.shape
    # Work in fp32 for math; cast IO around the kernel
    x32 = x.to(torch.float32)
    r32 = residual.to(torch.float32)
    b32 = (
        bias.to(torch.float32)
        if bias is not None
        else torch.empty(1, dtype=torch.float32, device=x.device)
    )
    y32 = torch.empty_like(x32)

    if seed is None:
        # make it reproducible if you pass a specific seed
        seed = torch.randint(0, 2**31 - 1, (), device=x.device).item()

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )

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
        training and p > 0.0,
        bias is not None,
    )
    return y32.to(x.dtype)
