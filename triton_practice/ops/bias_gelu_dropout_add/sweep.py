import argparse, itertools, json, time, csv, torch
from .api import forward
from ...utils.check import assert_allclose


def make_inputs(B, S, H, device, dtype, p, seed=1234):
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    x = torch.randn(B, S, H, dtype=dtype, device=device, generator=g)
    bias = torch.randn(H, dtype=dtype, device=device, generator=g)
    res = torch.randn(B, S, H, dtype=dtype, device=device, generator=g)
    return x, bias, res, p


# def time_one(fn, warmups=30, iters=100):
#     for _ in range(warmups):
#         fn()
#     torch.cuda.synchronize()
#     t0 = time.perf_counter()
#     for _ in range(iters):
#         fn()
#     torch.cuda.synchronize()
#     t1 = time.perf_counter()
#     return (t1 - t0) / iters  # seconds
import time, torch


def time_one(
    fn, warmups: int = 10, min_run_time_s: float = 0.25, max_iters: int = 1_000_000
):
    """
    Times `fn()` on CUDA by running until at least `min_run_time_s` has elapsed.
    Returns mean seconds per call.
    """
    torch.cuda.synchronize()
    with torch.inference_mode():
        # Warmups (don’t measure)
        for _ in range(warmups):
            fn()
        torch.cuda.synchronize()

        # Time: geometric growth of iters until we hit the target wall-time
        iters = 1
        while True:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                fn()
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            elapsed = t1 - t0

            if elapsed >= min_run_time_s or iters >= max_iters:
                return elapsed / iters

            # grow iter count to reach target next round, with a safety factor
            # (avoids too many short measurements)
            scale = max(2.0, (min_run_time_s / max(elapsed, 1e-9)) * 1.2)
            iters = min(max_iters, int(iters * scale))


def run_point(B, S, H, dtype, p, device, repeats):
    x, bias, res, p = make_inputs(B, S, H, device, dtype, p)
    n_elem = x.numel()

    def fe():
        forward(x, bias, res, p, backend="eager")

    def ft():
        forward(x, bias, res, p, backend="triton")

    def fc():
        forward(x, bias, res, p, backend="compiled")

    results = {}
    for name, fn in [("eager", fe), ("triton", ft), ("compiled", fc)]:
        times = []
        for _ in range(repeats):
            t = time_one(fn)
            times.append(t)
        mean = sum(times) / len(times)
        var = sum((t - mean) ** 2 for t in times) / max(1, len(times) - 1)
        geps = (n_elem / mean) / 1e9
        results[name] = dict(time_s=mean, std_s=var**0.5, geps=geps)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_csv", default="bias_gelu_dropout_add_results.csv")
    ap.add_argument("--out_json", default="bias_gelu_dropout_add_results.json")
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise Exception("CUDA is not available, can't run triton tests")
    device = "cuda"
    dtype = torch.float32

    shapes = [
        (1, 128, 1024),
        (2, 256, 2048),
        (4, 512, 4096),
        (8, 1024, 4096),
        (8, 2048, 4096),
        (16, 2048, 8192),
    ]
    ps = [0.0, 0.1]

    rows = []
    all_json = []
    for (B, S, H), p in itertools.product(shapes, ps):
        torch.cuda.empty_cache()
        res = run_point(
            B,
            S,
            H,
            dtype=dtype,
            p=p,
            device=device,
            repeats=args.repeats,
        )
        row = {
            "B": B,
            "S": S,
            "H": H,
            "dtype": "fp32",
            "p": p,
            "eager_ms": res["eager"]["time_s"] * 1e3,
            "triton_ms": res["triton"]["time_s"] * 1e3,
            "compiled_ms": res["compiled"]["time_s"] * 1e3,
            "eager_geps": res["eager"]["geps"],
            "triton_geps": res["triton"]["geps"],
            "compiled_geps": res["compiled"]["geps"],
            "triton_vs_eager_speedup": res["eager"]["time_s"] / res["triton"]["time_s"],
            "triton_vs_compiled_speedup": res["compiled"]["time_s"]
            / res["triton"]["time_s"],
        }
        rows.append(row)
        all_json.append(
            {"B": B, "S": S, "H": H, "dtype": "fp32", "p": p, "results": res}
        )
        print(
            f"[{row['dtype']} p={p}] (B,S,H)=({B},{S},{H}) "
            f"Triton {row['triton_ms']:.3f} ms | "
            f"vs Eager {row['triton_vs_eager_speedup']:.2f}× | "
            f"vs Compiled {row['triton_vs_compiled_speedup']:.2f}×"
        )

    # Write CSV
    fieldnames = list(rows[0].keys())
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # Write JSON (full stats incl std)
    with open(args.out_json, "w") as f:
        json.dump(
            {
                "env": {
                    "gpu_name": (
                        torch.cuda.get_device_name(0)
                        if torch.cuda.is_available()
                        else "cpu"
                    ),
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "triton": getattr(torch, "version", None)
                    and getattr(torch.version, "triton", "unknown"),
                },
                "data": all_json,
            },
            f,
            indent=2,
        )

# uv run python -m triton_practice.ops.bias_gelu_dropout_add.sweep
if __name__ == "__main__":
    main()
