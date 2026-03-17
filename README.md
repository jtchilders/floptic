# Floptic 🔬

**A lens on your FLOP throughput.**

Floptic is a portable floating-point benchmark tool that measures actual FLOP/s throughput across all precision levels on modern GPUs and CPUs. It reports structured JSON and markdown with rates, percentage of theoretical peak, and timing statistics.

## Motivation

Modern GPUs invest transistors in reduced-precision arithmetic (FP16, BF16, TF32, FP8, FP4) while FP64 throughput stagnates. On NVIDIA's Blackwell B200:

| Precision | Measured Rate | Ratio vs FP64 |
|-----------|--------------|---------------|
| FP64 (CUDA cores) | 36.1 TF/s | 1× |
| FP32 (CUDA cores) | 66.5 TF/s | 1.8× |
| TF32 (tensor cores) | 1.1 PF/s | 30× |
| FP16 (tensor cores) | 2.2 PF/s | **60×** |
| INT8 (tensor cores) | 4.4 PF/s | 121× |
| FP8 E4M3 (tensor cores) | 4.4 PF/s | 121× |
| FP4 (tensor cores) | 7.4 PF/s | **205×** |

This 60–205× gap between FP64 and low-precision tensor core throughput motivates research into **precision emulation schemes** (e.g., Ozaki) that achieve FP64-accurate results using low-precision hardware:

| Emulation Method | Rate | Speedup vs Native |
|-----------------|------|-------------------|
| Ozaki FP64 (via INT8 TC) | 83.8 TF/s | **2.3× native FP64** |
| BF16×9 FP32 (via BF16 TC) | 243 TF/s | **3.7× native FP32** |

Floptic quantifies these ratios to guide algorithm design for scientific computing on modern hardware.

## Features

- **All precision levels**: FP64, FP32, TF32, FP16, BF16, FP8 (E4M3/E5M2), FP4, INT8
- **Block-scaled formats**: MXFP8 (Blackwell), NVFP4 (Blackwell)
- **Emulated precision**: BF16×9 FP32 emulation, Ozaki FP64 emulation
- **Tensor core aware**: Separate TC vs CUDA-core measurements
- **Memory bandwidth**: STREAM triad/copy (HBM bandwidth in TB/s)
- **CPU benchmarks**: AVX2/AVX-512 explicit SIMD intrinsics
- **Auto-sweep**: Matrix kernels sweep sizes to find true peak
- **Architecture guards**: Kernels only run on supported hardware (no zero rows)
- **Structured output**: JSON + markdown reports with system info

## Supported Hardware

| GPU | Architecture | Precisions | Status |
|-----|-------------|------------|--------|
| NVIDIA A100 | sm_80 (Ampere) | FP64–BF16, FP64 TC | ✅ Tested |
| NVIDIA H100 / GH200 | sm_90 (Hopper) | FP64–FP8, FP64 TC | ✅ Tested |
| NVIDIA B200 | sm_100 (Blackwell) | FP64–FP4, MXFP8, NVFP4, emulated | ✅ Tested |
| AMD MI250X / MI300 | gfx90a / gfx942 | — | Planned |
| Intel Max (Ponte Vecchio) | — | — | Planned |

| CPU | ISA | Status |
|-----|-----|--------|
| x86_64 (AVX2) | FP64, FP32 | ✅ Tested |
| x86_64 (AVX-512) | FP64, FP32 | ✅ Tested |
| aarch64 (Grace) | FP64, FP32 | ✅ Tested (scalar) |

## Quick Start

### Prerequisites

- CMake ≥ 3.22
- C++17 compiler (GCC ≥ 9)
- CUDA Toolkit ≥ 11.8 (for GPU benchmarks)
- CUDA ≥ 12.8 for FP8/FP4 block-scaled formats
- CUDA ≥ 13.1 for emulated precision (BF16×9, Ozaki)

### Build

```bash
git clone git@github.com:jtchilders/floptic.git
cd floptic

# For A100 (sm_80)
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=80
make -C build -j

# For H100/GH200 (sm_90)
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=90
make -C build -j

# For B200 (sm_100)
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=100
make -C build -j
```

### Run

```bash
# Full benchmark (all precisions, all kernels)
./build/floptic --device=cuda --precision=all

# Save results
./build/floptic --device=cuda --precision=all \
    --output=results.json \
    --output-md=results.md

# Specific kernel only
./build/floptic --device=cuda --kernel=gemm_cublas --precision=all

# CPU only
./build/floptic --device=cpu --precision=fp64,fp32

# List available kernels
./build/floptic --list

# Device info
./build/floptic --info
```

## CLI Reference

```
Usage: floptic [options]

Options:
  --device=<DEV>       Target device(s): cpu, cuda:N, all (default: all)
  --precision=<PREC>   Precisions: fp64, fp32, fp16, bf16, tf32, int8,
                       fp8e4m3, fp8e5m2, fp4, all (default: fp64,fp32)
  --kernels=<CAT>      Kernel categories: scalar, vector, matrix, memory,
                       all (default: all)
  --kernel=<NAME>      Run a specific kernel by name
  --trials=<N>         Measurement trials (default: 100)
  --inner-iters=<N>    Inner loop iterations per trial (default: 100000)
  --warmup=<N>         Warmup iterations (default: 10)
  --output=<PATH>      JSON output file
  --output-md=<PATH>   Markdown report output file

Thread/block control:
  --cpu-threads=<N>    CPU threads (default: all physical cores)
  --gpu-blocks=<N>     CUDA grid blocks (default: auto)
  --gpu-tpb=<N>        Threads per block (default: 256)
  --gpu-bpsm=<N>       Blocks per SM (default: 4)

Other:
  --list               List available kernels and exit
  --info               Print device info and exit
  --help               Show help
```

## Kernels

### Scalar (`scalar`)

| Kernel | Description | Modes | Backend |
|--------|-------------|-------|---------|
| `scalar_fma` | Fused multiply-add throughput & latency | throughput, latency | CUDA, CPU |

**Throughput mode**: 8 independent FMA chains to saturate all FMA units.
**Latency mode**: Single dependent FMA chain measuring instruction latency.

CPU implementation uses explicit SIMD intrinsics (AVX2 `_mm256_fmadd_pd`, AVX-512 `_mm512_fmadd_pd`) with scalar fallback.

### Vector / Memory (`vector`, `memory`)

| Kernel | Description | Measures | Backend |
|--------|-------------|----------|---------|
| `vector_axpy` | BLAS L1 y = αx + y | GFLOP/s + bandwidth | CUDA, CPU |
| `stream_triad` | a = b + c×scalar | HBM bandwidth (TB/s) | CUDA |
| `stream_copy` | a = b | HBM bandwidth (TB/s) | CUDA |

### Matrix (`matrix`)

| Kernel | Description | Precisions | Architecture |
|--------|-------------|------------|--------------|
| `gemm_cublas` | cuBLAS GEMM (auto tensor core) | FP64, FP32, TF32, FP16, BF16 | sm_70+ |
| `gemm_cublas_notc` | cuBLAS GEMM forced to CUDA cores | FP64, FP32 | sm_70+ |
| `gemm_cublas_int8` | cublasLt INT8 GEMM with auto-tuning | INT8 | sm_70+ |
| `gemm_cublas_fp8` | cublasLt FP8 per-tensor scaling | FP8_E4M3, FP8_E5M2 | sm_89–99 (Hopper) |
| `gemm_cublas_mxfp8` | cublasLt MXFP8 block-scaled | FP8_E4M3 | sm_100+ (Blackwell) |
| `gemm_cublas_nvfp4` | cublasLt NVFP4 block-scaled | FP4 | sm_100+ (Blackwell) |
| `gemm_cublas_emu_fp32` | cuBLAS BF16×9 emulated FP32 | FP32 | sm_90+ |
| `gemm_cublas_emu_fp64` | cuBLAS Ozaki emulated FP64 | FP64 | sm_100+ (Blackwell) |

All GEMM kernels use **auto-sweep** over M=N=K ∈ {1024, 2048, 4096, 8192, 16384} and report the peak result. INT8 and FP8 kernels use cublasLt auto-tuning to find the fastest algorithm.

## Output

### Console (stderr)

A box-drawing table printed during execution:

```
╔════════════════════════════════════════════════════════════════════════════════════════╗
║  cuda:0 (NVIDIA B200)
╠════════════════════════════════════════════════════════════════════════════════════════╣
║ Kernel                   │ Prec     │ Mode       │         Rate │  Peak% │ Median (ms) ║
╟──────────────────────────┼──────────┼────────────┼──────────────┼────────┼─────────────╢
║ gemm_cublas              │ FP64     │ throughput │    36.1 TF/s │  96.9% │     243.753 ║
║ gemm_cublas              │ FP16     │ throughput │     2.2 PF/s │  96.8% │       4.039 ║
║ gemm_cublas_mxfp8        │ FP8_E4M3 │ throughput │     4.4 PF/s │  96.7% │       2.020 ║
║ gemm_cublas_nvfp4        │ FP4      │ throughput │     7.4 PF/s │  82.3% │       0.148 ║
║ gemm_cublas_emu_fp64     │ FP64     │ throughput │    83.8 TF/s │ 225.0% │     103.966 ║
╚════════════════════════════════════════════════════════════════════════════════════════╝
```

### JSON (`--output=<PATH>`)

Structured report with system info, device properties, theoretical peaks, and per-kernel results including min/max/mean/median timing and GFLOP/s.

### Markdown (`--output-md=<PATH>`)

Human-readable report with tables, suitable for committing to a repository or embedding in papers. See [`results/`](results/) for examples.

## Architecture

```
floptic/
├── include/floptic/
│   ├── kernel_base.hpp        # KernelBase interface
│   ├── kernel_registry.hpp    # Self-registration macro
│   ├── precision.hpp          # Precision enum + traits
│   ├── device_info.hpp        # DeviceInfo struct
│   ├── cli_parser.hpp         # CLI options
│   ├── report.hpp             # Report structs
│   └── timer.hpp              # Timer interface
├── src/
│   ├── main.cpp               # Entry point + dispatch
│   ├── cli_parser.cpp         # --key=value parser
│   ├── kernel_registry.cpp    # Global registry
│   ├── device/
│   │   ├── cpu_device_info.cpp    # /proc/cpuinfo, CPUID SIMD detection
│   │   └── cuda_device_info.cu   # cudaGetDeviceProperties + peak tables
│   ├── kernels/
│   │   ├── scalar/            # FMA throughput/latency
│   │   ├── vector/            # AXPY, STREAM
│   │   └── matrix/            # cuBLAS GEMM variants
│   ├── report/
│   │   ├── json_writer.cpp    # JSON output
│   │   ├── md_writer.cpp      # Markdown output
│   │   └── system_info.cpp    # hostname, modules, CUDA versions
│   └── harness/               # Timer, warmup, validation, power
├── results/                   # Committed benchmark results
└── tests/                     # Standalone diagnostic tests
```

### Adding a New Kernel

1. Create a source file in the appropriate `src/kernels/<category>/` directory
2. Inherit from `KernelBase` and implement:
   - `name()`, `category()`, `backend()` — identity
   - `supported_precisions()` — which precisions this kernel handles
   - `supported_modes()` — typically `{"throughput"}` and/or `{"latency"}`
   - `is_available(const DeviceInfo&)` — runtime architecture check
   - `run(config, device, trials)` — the benchmark logic
3. Add `REGISTER_KERNEL(YourKernel);` at file scope
4. Add a force-link reference in `src/main.cpp` (needed for static library linking)
5. Add the source file to `CMakeLists.txt`

### Theoretical Peak Calculation

Peaks are computed from per-architecture tables of FMA units per SM per clock:

```
peak_gflops = SMs × clock_GHz × fma_per_sm_per_clock × 2
```

Values are sourced from NVIDIA architecture whitepapers and product datasheets. The tables live in `src/device/cuda_device_info.cu` as `switch(major)` functions.

## Results

Pre-generated benchmark results are in [`results/`](results/):

- [`jlse_gpu_b200.md`](results/jlse_gpu_b200.md) — NVIDIA B200 (Blackwell, sm_100)
- [`jlse_gpu_gh200.md`](results/jlse_gpu_gh200.md) — NVIDIA GH200 (Hopper, sm_90)

## Known Limitations

- **Single GPU only** — no multi-device benchmarks
- **User-level privileges** — cannot lock GPU clocks; boost frequency varies
- **CPU detection** — aarch64 (Grace) reports 0 physical cores and 0 MHz (no `/proc/cpuinfo` clock info)
- **FP8 per-tensor** — only works on Hopper x86_64; fails on aarch64 and Blackwell
- **Peak% > 100%** — emulated kernels (BF16×9, Ozaki) use tensor cores but are compared against CUDA core peaks for their output precision

## License

TBD

## Authors

- Taylor Childers (ANL)
- Reet Barik (ANL)
