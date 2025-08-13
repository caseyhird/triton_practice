import torch, time
from contextlib import contextmanager


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@contextmanager
def timed():
    cuda_sync()
    t0 = time.perf_counter()
    yield
    cuda_sync()
    t1 = time.perf_counter()
    print(f"{(t1 - t0) * 1e3:.3f} ms")


def run_bench(fn, warmup=25, iters=100):
    for _ in range(warmup):
        fn()
    cuda_sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    cuda_sync()
    dt = (time.perf_counter() - t0) / iters
    return dt
