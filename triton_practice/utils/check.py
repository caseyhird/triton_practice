import torch


def assert_allclose(a, b, rtol=1e-3, atol=1e-3):
    if not torch.allclose(a, b, rtol=rtol, atol=atol):
        mx = (a - b).abs().max().item()
        raise AssertionError(f"mismatch, max abs diff={mx}")
