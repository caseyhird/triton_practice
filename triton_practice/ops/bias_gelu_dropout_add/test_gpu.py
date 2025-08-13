# triton_practice/ops/bias_gelu_dropout_add/tests.py

import torch
import pytest
from .api import forward
from ...utils.check import assert_allclose

# We can run these on CPU for eager, but Triton requires CUDA
cuda_available = torch.cuda.is_available()


@pytest.mark.parametrize("B,S,H", [(1, 8, 16), (2, 4, 32), (4, 4, 64)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("p", [0.0, 0.1])
def test_forward_matches_eager(B, S, H, dtype, p):
    device = "cuda" if cuda_available else "cpu"

    # Seed so dropout matches exactly
    torch.manual_seed(1234)

    x = torch.randn(B, S, H, dtype=dtype, device=device)
    bias = torch.randn(H, dtype=dtype, device=device)
    res = torch.randn_like(x)

    y_eager = forward(x, bias, res, p, backend="eager", seed=1234)
    y_triton = forward(x, bias, res, p, backend="triton", seed=1234)

    # compare in float32 for safety (BF16/FP16 tolerance)
    assert_allclose(y_eager.float(), y_triton.float(), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("p", [0.0, 0.2])
def test_no_bias_equivalence(p):
    device = "cuda" if cuda_available else "cpu"
    torch.manual_seed(42)
    x = torch.randn(2, 4, 8, dtype=torch.float32, device=device)
    res = torch.randn_like(x)

    y1 = forward(x, None, res, p, backend="eager", seed=42)
    y2 = forward(x, None, res, p, backend="triton", seed=42)

    assert_allclose(y1, y2, rtol=1e-5, atol=1e-5)


def test_invalid_shapes_raise():
    device = "cuda" if cuda_available else "cpu"
    x = torch.randn(2, 4, 8, device=device)
    bias = torch.randn(5, device=device)  # wrong shape
    res = torch.randn_like(x)

    with pytest.raises(AssertionError):
        forward(x, bias, res, 0.0, backend="triton")
