import torch
from .eager import forward as eager_forward
from .triton_kernel import triton_forward
from ...compile.inductor import compiled


def forward(x, bias, residual, p=0.0, backend="triton", seed=1234, training=True):
    if backend == "eager":
        return eager_forward(x, bias, residual, p, training)
    if backend == "compiled":
        return compiled(eager_forward)(x, bias, residual, p, training)
    return triton_forward(x, bias, residual, p, seed)
