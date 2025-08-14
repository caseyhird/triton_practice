import torch
import pytest
import numpy as np
from .api import forward
from ...utils.check import assert_allclose

cuda_available = torch.cuda.is_available()


@pytest.mark.parametrize("B,S,H", [(1, 8, 16), (2, 4, 32), (4, 4, 64)])
def test_forward_matches_eager(B, S, H):
    if not cuda_available:
        raise Exception("CUDA is not available, can't run triton tests")

    device = "cuda"
    dtype = torch.float32  # only implemented triton in fp32

    # Seed so dropout matches exactly
    torch.manual_seed(1234)
    # pytorch dropout is not deterministic, so we need p=0 for exact matching
    p = 0.0

    x = torch.randn(B, S, H, dtype=dtype, device=device)
    bias = torch.randn(H, dtype=dtype, device=device)
    res = torch.randn_like(x)

    y_eager = forward(x, bias, res, p, backend="eager", seed=1234)
    y_triton = forward(x, bias, res, p, backend="triton", seed=1234)

    # compare in float32 for safety (BF16/FP16 tolerance)
    assert_allclose(y_eager.float(), y_triton.float(), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("p", [0.1, 0.2, 0.5])
def test_triton_dropout_zero_fraction(p):
    """Test that dropout produces approximately the correct fraction of zeros."""
    if not cuda_available:
        raise Exception("CUDA is not available, can't run triton tests")

    device = "cuda"
    backend = "triton"

    B, S, H = 4, 64, 128
    x = torch.randn(B, S, H, dtype=torch.float32, device=device)
    bias = torch.randn(H, dtype=torch.float32, device=device)
    res = torch.randn_like(x)

    # multiple runs to get statistical properties
    n_runs = 100
    zero_counts = []

    for i in range(n_runs):
        torch.manual_seed(i)
        y = forward(x, bias, res, p, backend=backend, seed=i, training=True)
        # Count zeros in the output - residual to isolate dropout part
        dropout_output = y - res
        zero_count = (dropout_output == 0).float().mean().item()
        zero_counts.append(zero_count)

    # fraction of zeros should be approx p
    mean_zero_fraction = np.mean(zero_counts)
    zero_fraction_diff = abs(mean_zero_fraction - p)
    assert (
        zero_fraction_diff < 0.05
    ), f"Zero fraction difference {zero_fraction_diff} too large for p={p}. Expected ~{p}, got ~{mean_zero_fraction}"


@pytest.mark.parametrize("p", [0.1, 0.2])
def test_triton_dropout_scaling_factor(p):
    """Test that dropout applies the correct scaling factor (1/(1-p)) to non-zero elements."""
    if not cuda_available:
        raise Exception("CUDA is not available, can't run triton tests")

    device = "cuda"
    backend = "triton"

    B, S, H = 2, 4, 8
    x = torch.ones(B, S, H, dtype=torch.float32, device=device)
    bias = torch.zeros(H, dtype=torch.float32, device=device)
    res = torch.zeros_like(x)

    n_runs = 50
    non_zero_values = []

    for i in range(n_runs):
        torch.manual_seed(i)
        y = forward(x, bias, res, p, backend=backend, seed=i, training=True)
        # Get all values non-zero after dropout (just y since res is 0)
        non_zero_mask = y != 0
        if non_zero_mask.any():
            non_zero_vals = y[non_zero_mask].cpu().numpy()
            non_zero_values.extend(non_zero_vals)

    if non_zero_values:
        # non-zero values should be approximately scaled by 1/(1-p)
        # since input was all ones, bias was zero, and GELU(1) ≈ 0.841
        expected_gelu = 0.841  # GELU(1) approximation
        expected_scaled = expected_gelu / (1 - p)

        actual_mean = np.mean(non_zero_values)
        scaling_diff = abs(actual_mean - expected_scaled)
        assert (
            scaling_diff < 0.1
        ), f"Scaling difference {scaling_diff} too large for p={p}. Expected ~{expected_scaled}, got ~{actual_mean}"


@pytest.mark.parametrize("p", [0.1, 0.2])
def test_dropout_deterministic_with_same_seed(p):
    """Test that dropout is deterministic when using the same seed."""
    device = "cuda" if cuda_available else "cpu"

    x = torch.randn(2, 4, 8, dtype=torch.float32, device=device)
    bias = torch.randn(8, dtype=torch.float32, device=device)
    res = torch.randn_like(x)

    # Run twice with same seed - should get identical results
    torch.manual_seed(42)
    y1 = forward(x, bias, res, p, backend="triton", seed=42, training=True)

    torch.manual_seed(42)
    y2 = forward(x, bias, res, p, backend="triton", seed=42, training=True)

    # Should be exactly equal since same seed
    assert_allclose(y1, y2, rtol=1e-7, atol=1e-7)


def test_invalid_shapes_raise():
    device = "cuda" if cuda_available else "cpu"
    x = torch.randn(2, 4, 8, device=device)
    bias = torch.randn(5, device=device)  # wrong shape
    res = torch.randn_like(x)

    with pytest.raises(AssertionError):
        forward(x, bias, res, 0.0, backend="triton")
