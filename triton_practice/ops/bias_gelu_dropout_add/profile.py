import argparse
import torch
import torch.profiler
from .api import forward
from ...bench.harness import run_bench
from ...utils.check import assert_allclose


def make_inputs(B, S, H, device, p):
    """
    Creates random input tensors for the model.
    """
    x = torch.randn(B, S, H, dtype=torch.float32, device=device)
    bias = torch.randn(H, dtype=torch.float32, device=device)
    res = torch.randn_like(x)
    return x, bias, res, p


def profile_forward(fn, name, *args):
    """
    Profiles a given function and prints the results.
    """
    print(f"\n--- Profiling {name} ---")
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
    ) as prof:
        with torch.profiler.record_function(f"{name}_forward"):
            fn(*args)

    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))


def main():
    """
    Main function to run correctness checks, benchmarking, and profiling.
    """
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
        print("--- Checking Correctness ---")
        y_eager = forward(x, bias, res, p, backend="eager")
        y_triton = forward(x, bias, res, p, backend="triton")
        y_comp = forward(x, bias, res, p, backend="compiled")

        assert_allclose(y_eager.float(), y_triton.float())
        assert_allclose(y_eager.float(), y_comp.float())
        print("Correctness checked successfully.\n")

    # --- Benchmarking ---
    print("--- Benchmarking ---")

    def fe():
        forward(x, bias, res, p, backend="eager")

    def ft():
        forward(x, bias, res, p, backend="triton")

    # run torch.compile once here so it doesn't run during the benchmark
    fc_compiled = torch.compile(lambda *args: forward(*args, backend="eager"))

    def fc():
        fc_compiled(x, bias, res, p)

    te = run_bench(fe)
    tt = run_bench(ft)
    tc = run_bench(fc)

    n_elem = x.numel()
    print(f"Eager:   {te*1e3:.3f} ms  ({1e-9*n_elem/te:.2f} Gelem/s)")
    print(f"Triton:  {tt*1e3:.3f} ms  ({1e-9*n_elem/tt:.2f} Gelem/s)")
    print(f"Compile: {tc*1e3:.3f} ms  ({1e-9*n_elem/tc:.2f} Gelem/s)")

    # --- Profiling ---
    print("\n--- Profiling GPU Kernels ---")
    print(
        "Name=operation, Self=time spent in this operation *not* including sub-operations, Total=time spent in this operation *including* sub-operations, Calls=number of kernel launches"
    )

    profile_forward(lambda *a: forward(*a, backend="eager"), "Eager", x, bias, res, p)

    profile_forward(lambda *a: forward(*a, backend="triton"), "Triton", x, bias, res, p)

    profile_forward(fc_compiled, "Compiled", x, bias, res, p)


# uv run python -m triton_practice.ops.bias_gelu_dropout_add.profile
if __name__ == "__main__":
    main()
