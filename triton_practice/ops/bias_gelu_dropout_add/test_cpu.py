import torch
from .eager import forward as eager_forward


# uv run pytest triton_practice/ops/bias_gelu_dropout_add/test_cpu.py
def test_eager_manual_gold_values():
    """
    Golden-value correctness test for the eager implementation on CPU
    (other implementations compared to eager for correctness).
    small, hand-checked inputs, dropout p=0.0 to avoid randomness.
    y = residual + GELU(x + bias)
    """
    device = "cpu"
    dtype = torch.float32

    # Shape: (B=1, S=1, H=5)
    x = torch.tensor([[[-1.0, -0.5, 2.0, -0.2, 1.0]]], dtype=dtype, device=device)
    bias = torch.tensor(
        [0.0, 0.5, -0.4, 1.0, 1.1], dtype=dtype, device=device
    )  # broadcast over last dim
    residual = torch.tensor([[[0.1, 0.0, 0.3, 0.0, 0.2]]], dtype=dtype, device=device)

    # Dropout disabled (p=0.0)
    out = eager_forward(x, bias, residual, p=0.0, training=True)

    # Manually computed expected values
    expected = torch.tensor(
        [
            [
                -0.05880799293518066,  # 0.1 + gelu(-1.0)
                0.0,  # 0.0 + gelu( 0.0)
                1.8121370553970337,  # 0.3 + gelu( 1.6)
                0.6304317116737366,  # 0.0 + gelu( 0.8)
                2.262668800354004,  # 0.2 + gelu( 2.1)
            ]
        ],
        dtype=dtype,
        device=device,
    ).unsqueeze(
        0
    )  # (1,1,5)

    # Compare with a tight tolerance since we're in fp32 and no RNG
    torch.testing.assert_close(out, expected, rtol=1e-7, atol=1e-7)
