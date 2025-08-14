# WIP

import argparse, torch
from contextlib import contextmanager
from .api import forward

try:
    import torch.cuda.nvtx as nvtx
except Exception:

    class _NVTX:
        def range_push(self, *_):
            pass

        def range_pop(self):
            pass

    nvtx = _NVTX()


@contextmanager
def nvtx_range(name):
    nvtx.range_push(name)
    try:
        yield
    finally:
        nvtx.range_pop()


def main():
    if not torch.cuda.is_available():
        raise Exception("CUDA is not available, can't run triton tests")
    device = "cuda"
    dtype = torch.float32

    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="triton")
    args = ap.parse_args()

    torch.cuda.synchronize()
    # warmups (don’t record)
    for _ in range(5):
        _ = forward_once(args.backend, device, dtype)
    torch.cuda.synchronize()

    # one short recorded run
    with nvtx_range("forward+backward"):
        out = forward_once(args.backend, device, dtype)
    torch.cuda.synchronize()


def forward_once(backend, device, dtype):
    def make_inputs(B, S, H, device, dtype, p, seed=1234):
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        x = torch.randn(B, S, H, dtype=dtype, device=device, generator=g)
        bias = torch.randn(H, dtype=dtype, device=device, generator=g)
        res = torch.randn(B, S, H, dtype=dtype, device=device, generator=g)
        return x, bias, res, p

    B, S, H = 1, 1024, 1024
    x, bias, res, p = make_inputs(B, S, H, device, dtype, 0.0)

    forward(x, bias, res, p, backend=backend)
    return torch.randn(()).cuda()


if __name__ == "__main__":
    main()
