# Triton Practice: Fused Bias-GELU-Dropout-Add Kernel

## Project Overview
This project includes implementations of a fused GPU kernels using [Triton](https://github.com/triton-lang/triton), starting with a kernel combining (bias addition + GELU + dropout + residual) in a single operation. Fused kernels like this are common in deep learning to optimize performance.

### What is Triton?
Triton is a "language and compiler for writing highly efficient custom Deep-Learning primitives". It is a DSL for writing GPU kernels with Python syntax, making it easy to author fused kernels that are specific to niche use cases.
For example, [torch.compile](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) will use triton on its backend to generate code for fused kernels when it finds operations in your model that can be easily fused.

### Fused Operations
The kernel implements the following fused operations:
1. bias_gelu_dropout_add
```
y = dropout(GELU(x + bias)) + residual
```
Note that we use the tanh approximation for GELU. See the [pytorch gelu docs](https://docs.pytorch.org/docs/stable/generated/torch.nn.GELU.html) for details about this.

Other operations to come.

## Implementation Details

### Kernel Architecture
The Triton kernel uses a 2D grid of thread blocks (tensors with more than 2 dims are flattened to 2D before the kernel and then reshaped back after the kernel):
- **Block dimensions**: Configurable `BLOCK_M × BLOCK_N` with autotuning
- **Operation dimensions**: Bias is applied to each row of the tensor, so it has a single dimension matching the last dim of the input. Gelu and dropout are applied elementwise. Residual is added per element, so it has the same dimensions as the input.
- **Memory access**: Loads and stores are done in a coalesced manner, with proper boundary handling for non-divisible tensor dimensions.

### Key Optimizations

1. **Autotuning**: The kernel includes multiple configurations that are benchmarked for each combination of input dimensions. The most performant configuration is then selected based on input dimensions for all future calls.
  ```python
  CONFIGS = [
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 256}, num_warps=8, num_stages=2),
    ...
  ]

  @triton.autotune(
    configs=CONFIGS,
    key=[
      "M",
      "N",
    ],
  )
  ```
  - `configs=CONFIGS` specifies which configs to benchmark.
  - `key=[ "M", "N" ]` specifies the dimensions to autotune on, so we will re-tune anytime M or N changes.

2. **GELU Implementation**: Uses the tanh approximation for efficiency:
  ```python
  sqrt_2_over_pi = 0.7978845608
  c = sqrt_2_over_pi * (z + 0.044715 * z * z * z)
  g = z * (tl.sigmoid(2.0 * c))
  ```

3. **Deterministic Dropout**: Layout-agnostic deterministic dropout using linear indexing:
  ```python
  linear_idx = (mm * N + nn).to(tl.int32)
  r = tl.rand(seed, linear_idx)
  keep = r > p
  scale = 1.0 / (1.0 - p)
  ```
  In a backward pass, this same indexing can be re-generated from the seed.

4. **Memory Layout Optimization**: Ensures contiguous memory access patterns and proper stride handling.

## Performance Analysis

All results were run on an Nvidia A40 GPU with cuda 12.8.1 using [runpod](https://www.runpod.io/).

### Simple benchmark results
Triton implementation is comparable to pytorch compiled (which makes sense since pytorch compiled is using triton on its backend -- see below for more details). Both offer ~3x speedup compared to pytorch eager.
Use `uv run python -m triton_practice.ops.bias_gelu_dropout_add.bench` to recreate output similar to:
```
Eager:   4.444 ms  (15.10 Gelem/s)
Triton:  1.505 ms  (44.60 Gelem/s)
Compile: 1.406 ms  (47.74 Gelem/s)
```

### "Sweep" results
On smaller tensors, pytorch eager is faster due to kernel launch overhead. On larger tensors, triton is faster.
Similar to our simple benchmark, pytorch compiled and triton have similar performance, both offering ~3x speedup compared to pytorch eager.
Use `uv run python -m triton_practice.ops.bias_gelu_dropout_add.sweep` to recreate output similar to:
```
[fp32 p=0.0] (B,S,H)=(1,128,1024) Triton 0.108 ms | vs Eager 0.23× | vs Compiled 0.38×
[fp32 p=0.1] (B,S,H)=(1,128,1024) Triton 0.089 ms | vs Eager 0.43× | vs Compiled 0.73×
[fp32 p=0.0] (B,S,H)=(2,256,2048) Triton 0.188 ms | vs Eager 0.21× | vs Compiled 0.29×
[fp32 p=0.1] (B,S,H)=(2,256,2048) Triton 0.090 ms | vs Eager 0.72× | vs Compiled 0.96×
[fp32 p=0.0] (B,S,H)=(4,512,4096) Triton 0.241 ms | vs Eager 1.77× | vs Compiled 0.77×
[fp32 p=0.1] (B,S,H)=(4,512,4096) Triton 0.249 ms | vs Eager 2.28× | vs Compiled 0.76×
[fp32 p=0.0] (B,S,H)=(8,1024,4096) Triton 0.779 ms | vs Eager 2.16× | vs Compiled 1.05×
[fp32 p=0.1] (B,S,H)=(8,1024,4096) Triton 0.783 ms | vs Eager 2.85× | vs Compiled 0.95×
[fp32 p=0.0] (B,S,H)=(8,2048,4096) Triton 1.504 ms | vs Eager 2.23× | vs Compiled 0.95×
[fp32 p=0.1] (B,S,H)=(8,2048,4096) Triton 1.499 ms | vs Eager 2.97× | vs Compiled 0.97×
[fp32 p=0.0] (B,S,H)=(16,2048,8192) Triton 5.790 ms | vs Eager 2.32× | vs Compiled 0.98×
[fp32 p=0.1] (B,S,H)=(16,2048,8192) Triton 5.778 ms | vs Eager 3.07× | vs Compiled 0.98×
```

### GPU profiling results
Looking at GPU operations, we can see the single `bias_gelu_dropout_residual_fwd` operation used in the triton implementation. Notice how the eager implementation spends much more time in e.g. the `aten::add` operation. 
Also notice how the compiled implementation uses the `triton_poi_fused_add_gelu_native_dropout_0` operation. This is similar to our triton kernel, except it's auto-generated by torch.compile.

Use `uv run python -m triton_practice.ops.bias_gelu_dropout_add.profile` to recreate output similar to:
```
--- Benchmarking ---
Eager:   4.445 ms  (15.10 Gelem/s)
Triton:  1.505 ms  (44.59 Gelem/s)
Compile: 1.405 ms  (47.76 Gelem/s)

--- Profiling GPU Kernels ---
Name=operation, Self=time spent in this operation *not* including sub-operations, Total=time spent in this operation *including* sub-operations, Calls=number of kernel launches

--- Profiling Eager ---
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                          Eager_forward         0.00%       0.000us         0.00%       0.000us       0.000us       5.028ms       113.46%       5.028ms       5.028ms             1  
                                              aten::add         1.10%      90.429us        25.48%       2.086ms       1.043ms       2.391ms        53.97%       3.346ms       1.673ms             2  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       1.437ms        32.43%       1.437ms       1.437ms             1  
                                   aten::native_dropout         0.34%      28.101us        14.59%       1.195ms       1.195ms       1.081ms        24.38%       1.081ms       1.081ms             1  
void at::native::(anonymous namespace)::fused_dropou...         0.00%       0.000us         0.00%       0.000us       0.000us       1.081ms        24.38%       1.081ms       1.081ms             1  
                                             aten::gelu         0.33%      27.347us         0.49%      40.145us      40.145us     959.259us        21.65%     959.259us     959.259us             1  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us     959.259us        21.65%     959.259us     959.259us             1  
                                Activity Buffer Request        23.72%       1.942ms        23.72%       1.942ms       1.942ms     954.620us        21.54%     954.620us     954.620us             1  
void at::native::elementwise_kernel<128, 2, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us     954.620us        21.54%     954.620us     954.620us             1  
                                          Eager_forward        30.17%       2.470ms        70.89%       5.805ms       5.805ms       0.000us         0.00%       5.386ms       5.386ms             1  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 8.188ms
Self CUDA time total: 4.431ms


--- Profiling Triton ---
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                         Triton_forward         0.00%       0.000us         0.00%       0.000us       0.000us       1.714ms       118.13%       1.714ms       1.714ms             1  
                         bias_gelu_dropout_residual_fwd         0.00%       0.000us         0.00%       0.000us       0.000us       1.447ms        99.71%       1.447ms       1.447ms             1  
                                          aten::random_         0.79%      30.995us        51.85%       2.047ms       2.047ms       2.495us         0.17%       4.990us       4.990us             1  
                                Activity Buffer Request        50.09%       1.977ms        50.09%       1.977ms       1.977ms       2.495us         0.17%       2.495us       2.495us             1  
void at::native::(anonymous namespace)::distribution...         0.00%       0.000us         0.00%       0.000us       0.000us       2.495us         0.17%       2.495us       2.495us             1  
                              aten::_local_scalar_dense         0.38%      15.137us         1.90%      74.993us      74.993us       1.760us         0.12%       1.760us       1.760us             1  
                         Memcpy DtoH (Device -> Pinned)         0.00%       0.000us         0.00%       0.000us       0.000us       1.760us         0.12%       1.760us       1.760us             1  
                                         Triton_forward         9.76%     385.086us        65.96%       2.604ms       2.604ms       0.000us         0.00%       6.750us       6.750us             1  
                                          aten::reshape         0.31%      12.379us         0.84%      33.111us      11.037us       0.000us         0.00%       0.000us       0.000us             3  
                                             aten::view         0.53%      20.732us         0.53%      20.732us       6.911us       0.000us         0.00%       0.000us       0.000us             3  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 3.947ms
Self CUDA time total: 1.451ms


--- Profiling Compiled ---
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                       Compiled_forward         0.00%       0.000us         0.00%       0.000us       0.000us       1.502ms       107.08%       1.502ms       1.502ms             1  
             triton_poi_fused_add_gelu_native_dropout_0         0.63%      26.973us         0.85%      36.602us      36.602us       1.400ms        99.84%       1.400ms       1.400ms             1  
             triton_poi_fused_add_gelu_native_dropout_0         0.00%       0.000us         0.00%       0.000us       0.000us       1.400ms        99.84%       1.400ms       1.400ms             1  
                                          aten::random_         0.51%      21.821us         1.11%      47.601us      47.601us       2.304us         0.16%       2.304us       2.304us             1  
void at::native::(anonymous namespace)::distribution...         0.00%       0.000us         0.00%       0.000us       0.000us       2.304us         0.16%       2.304us       2.304us             1  
                                       Compiled_forward         1.85%      79.153us        69.40%       2.973ms       2.973ms       0.000us         0.00%       1.403ms       1.403ms             1  
                               TorchDynamo Cache Lookup         0.74%      31.760us         0.74%      31.760us      31.760us       0.000us         0.00%       0.000us       0.000us             1  
                             Torch-Compiled Region: 0/0         5.41%     231.725us        66.81%       2.862ms       2.862ms       0.000us         0.00%       1.403ms       1.403ms             1  
                                      Pregraph bytecode         0.10%       4.405us         0.10%       4.405us       4.405us       0.000us         0.00%       0.000us       0.000us             1  
                 AOTDispatcher Runtime Wrapper Prologue         0.21%       8.885us         0.21%       8.885us       8.885us       0.000us         0.00%       0.000us       0.000us             1  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 4.284ms
Self CUDA time total: 1.403ms
```

## Testing and Validation

### CPU Tests
test_cpu.py includes one simple "golden-value" test. This copmares the output of the pytorch implementation with values that were hand-calculated to ensure that the kernel is correct.

Run these tests with `uv run pytest triton_practice/ops/bias_gelu_dropout_add/test_cpu.py`

### GPU Tests
test_gpu.py includes a few tests that verify the correctness of the triton kernel. These tests include:
1. Some that run each implementation and check that they produce the same outputs
2. Since dropout is a random operation and we can't make this deterministic inside the pytorch implementation, we have separate tests for the statistical properties of the outputs after dropout.

Run these tests with `uv run pytest triton_practice/ops/bias_gelu_dropout_add/test_gpu.py`


## Project Structure

```
triton_practice/
├── triton_practice/
│   ├── ops/
│   │   └── bias_gelu_dropout_add/
│   │       ├── triton_kernel.py    # Main Triton kernel implementation
│   │       ├── eager.py            # PyTorch eager reference implementation
│   │       ├── api.py              # Unified API for all backends
│   │       ├── test_gpu.py         # GPU correctness tests
│   │       ├── test_cpu.py         # CPU correctness tests
│   │       ├── bench.py            # Basic benchmarking script
│   │       ├── sweep.py            # Comprehensive performance sweep
│   │       └── profile.py          # GPU profiling and analysis
│   ├── bench/
│   │   └── harness.py              # Benchmarking utilities
│   └── utils/
│       └── check.py                # Testing utilities
├── bias_gelu_dropout_add_results.csv  # Benchmark results
└── bias_gelu_dropout_add_results.json # Detailed performance data
```

## Future Improvements
1. Implement backward passes
2. Add additional fused kernel ops
3. Support more data types (currently only fp32)

## Conclusion
This project gives a quick introduction to the power and complexity of GPU kernel optimization using Triton. This implementation shows significant performance improvements for larger tensors, demonstrates how these improvements can be achieved with torch.compile, and highlights the trade-offs between development effort and performance gains.

---

## Recreating this project with runpod
You can run this repo using a pod with a GPU at runpod.io.

To get setup, go to runpod.io, login, and create a new pod.
You can then connect to this new pod and prepare it for use with these steps:
1. Connect to your pod via [ssh from the terminal](https://docs.runpod.io/pods/configuration/use-ssh) or with [vscode or cursor](https://docs.runpod.io/pods/configuration/connect-to-ide). For cursor the tl;dr is:
  a. Copy ssh details into ~/.ssh/config (can do in cursor). NOTE: make sure to get port > changes on every restart
  b. Connect to pod via cursor ssh
2. Once connected to the pod setup github creds so you can clone the repo
  a. Generate an SSH key on this pod with `ssh-keygen -t ed25519 -C "your_email@example.com"`. Note: this will be placed in `~/.ssh/id_ed25519` by default in the root dir, which with the default runpod config will be lost on restart (I don't have a good way around this yet).
  b. cat ~/.ssh/id_ed25519.pub (or the path you chose to store your SSH key)
  c. Copy paste into github ssh keys in your github account
3. (Optional) Setup git config. This isn't needed unless you want to e.g. create a new commit.
  a. git config --global user.email "your_email@example.com"
  b. git config --global user.name "Your Name"
4. Now clone the repo with `git clone git@github.com:caseyhird/triton_practice.git`
6. Move into the project dir with `cd triton_practice`
7. Setup project dependencies using uv
  a. Install uv itself with `pip install uv`
  b. Install the project dependencies with `uv sync`
8. You can now run the project with `uv run python -m <path to script>`. Most scripts, e.g. `bench.py` and `profile.py` have a comment by the main() function that shows how to run the script.
