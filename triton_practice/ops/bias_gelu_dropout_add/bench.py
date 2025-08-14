import argparse, torch
from .api import forward
from ...bench.harness import run_bench
from ...utils.check import assert_allclose


def make_inputs(B, S, H, device, p):
    x = torch.randn(B, S, H, dtype=torch.float32, device=device)
    bias = torch.randn(H, dtype=torch.float32, device=device)
    res = torch.randn_like(x)
    return x, bias, res, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--S", type=int, default=2048)
    ap.add_argument("--H", type=int, default=4096)
    ap.add_argument("--p", type=float, default=0.1)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    x, bias, res, p = make_inputs(args.B, args.S, args.H, args.device, args.p)

    # if p=0 we can check correctness before benchmarking
    if p == 0.0:
        y_eager = forward(x, bias, res, p, backend="eager")
        y_triton = forward(x, bias, res, p, backend="triton")
        y_comp = forward(x, bias, res, p, backend="compiled")

        assert_allclose(y_eager.float(), y_triton.float())
        assert_allclose(y_eager.float(), y_comp.float())
        print("Correctness checked")

    # Benchmarking
    def fe():
        forward(x, bias, res, p, backend="eager")

    def ft():
        forward(x, bias, res, p, backend="triton")

    # run torch.compile once here so it doesn't run during the benchmark
    fc = forward(x, bias, res, p, backend="compiled")

    te = run_bench(fe)
    tt = run_bench(ft)
    tc = run_bench(fc)

    n_elem = x.numel()
    print(f"Eager:   {te*1e3:.3f} ms  ({1e-9*n_elem/te:.2f} Gelem/s)")
    print(f"Triton:  {tt*1e3:.3f} ms  ({1e-9*n_elem/tt:.2f} Gelem/s)")
    print(f"Compile: {tc*1e3:.3f} ms  ({1e-9*n_elem/tc:.2f} Gelem/s)")


# uv run python -m triton_practice.ops.bias_gelu_dropout_add.bench
if __name__ == "__main__":
    main()
