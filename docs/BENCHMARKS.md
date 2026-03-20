# Benchmark Methodology

This document describes every kernel that Floptic runs, what it measures, how the measurement is performed, and how the reported metrics are computed.

---

## Overview

Floptic benchmarks fall into three categories:

| Category | Purpose | Primary Metric | Unit |
|----------|---------|----------------|------|
| **Scalar** | Measure FMA throughput/latency per core | Compute rate | GFLOP/s, TFLOP/s |
| **Vector** | BLAS Level-1 (memory-bound) | Compute rate | GFLOP/s |
| **Memory** | Sustained HBM/DRAM bandwidth | Bandwidth | GB/s, TB/s |
| **Matrix** | GEMM (compute-bound, exercises tensor/matrix cores) | Compute rate | GFLOP/s, TFLOP/s, PFLOP/s |

### Common methodology

All benchmarks follow this pattern:

1. **Warmup** — one or more untimed invocations to fill caches and trigger JIT compilation
2. **Timed trials** — `N` iterations (default 100, set via `--iterations`) timed individually using hardware event timers (`cudaEventRecord`, `hipEventRecord`, `std::chrono`)
3. **Statistics** — trials are sorted; the **median** is used as the primary timing result (robust to outliers from OS jitter, frequency scaling, etc.)
4. **FLOP counting** — each kernel computes its total FLOPs from first principles (not from any hardware counter)
5. **Rate** — `GFLOP/s = total_FLOPs / median_time`
6. **Peak%** — `rate / theoretical_peak × 100`, where the theoretical peak is computed from hardware specs (clock × units × ops_per_unit_per_clock)

### Timing

| Backend | Timer | Resolution |
|---------|-------|------------|
| CUDA | `cudaEventRecord` + `cudaEventElapsedTime` | ~0.5 μs |
| HIP | `hipEventRecord` + `hipEventElapsedTime` | ~0.5 μs |
| CPU | `std::chrono::high_resolution_clock` | ~1 ns |

GPU timers include a `cudaDeviceSynchronize` / `hipDeviceSynchronize` before starting the timer to ensure no prior work is in flight.

### FLOP Counting Convention

- **FMA** (fused multiply-add) = **2 FLOPs** (1 multiply + 1 add)
- **GEMM** C = α·A·B + β·C where A is M×K, B is K×N: **FLOPs = 2·M·N·K** (each output element requires K multiply-adds)
- Memory kernels report **bytes/second**, not FLOP/s, but use the `gflops` field internally for unified reporting

---

## Scalar Kernels

### `scalar_fma` — FMA Throughput & Latency

**Source**: `src/kernels/scalar/scalar_fma_{cuda,hip,cpu}.{cu,cpp}`
**Category**: scalar
**Precisions**: FP64, FP32, FP16, BF16
**Modes**: throughput, latency
**Backends**: CUDA, HIP, CPU

#### What it measures

Raw FMA (fused multiply-add) instruction throughput or latency per compute unit, without any memory traffic. This measures the theoretical peak compute rate of the hardware's floating-point units.

#### Kernel design

Each thread executes a tight loop of FMA operations on register-resident values:

```
for i in 0..iters:
    r = fma(a, r, b)    // r = a * r + b
```

- **Throughput mode**: 8 independent FMA chains per thread. Independent chains can be issued simultaneously, saturating the FMA pipeline (ILP). Measures peak sustained throughput.
- **Latency mode**: 1 dependent FMA chain per thread. Each FMA depends on the previous result, measuring the pipeline latency of a single FMA instruction.

#### GPU launch configuration

- **Threads**: `blocks × threads_per_block` where `blocks = SMs × blocks_per_SM`
- **Default**: auto-configured to fill the GPU (`gpu_blocks = 0` means all SMs × 32 blocks/SM, `gpu_tpb = 256`)
- **Inner iterations**: 100,000 (default `--iterations`)

#### CPU SIMD dispatch

The CPU kernel uses explicit SIMD intrinsics, dispatched at runtime based on CPUID detection:

| SIMD Path | FP64 Lanes | FP32 Lanes | Chains | FLOPs/iter/thread |
|-----------|-----------|-----------|--------|-------------------|
| AVX-512 | 8 | 16 | 8 | 128 (FP64), 256 (FP32) |
| AVX2 | 4 | 8 | 8 | 64 (FP64), 128 (FP32) |
| Scalar | 1 | 1 | 8 | 16 |

#### FLOP calculation

```
FLOPs = threads × chains × iterations × 2
```

Where `chains = 8` (throughput) or `chains = 1` (latency), and `×2` counts the multiply and add in each FMA.

#### Peak% reference

Compared against the per-precision vector peak:
- **GPU**: `clock_MHz × compute_units × ops_per_CU_per_clock × 2 (FMA) / 1000`
- **CPU**: `clock_MHz × physical_cores × SIMD_lanes × 2 (FMA units) × 2 (FMA) / 1000`

---

## Vector Kernels

### `vector_axpy` — AXPY (y = a·x + y)

**Source**: `src/kernels/vector/{vector_axpy_cuda.cu,vector_axpy_hip.cpp,vector_axpy_cpu.cpp}`
**Category**: vector
**Precisions**: FP64, FP32, FP16, BF16
**Modes**: throughput
**Backends**: CUDA, HIP, CPU

#### What it measures

BLAS Level-1 AXPY operation: `y[i] = a * x[i] + y[i]`. This is memory-bound — each element requires 1 FMA (2 FLOPs) but 3 memory operations (read x, read y, write y). The measured GFLOP/s will be far below the compute peak, reflecting memory bandwidth limitations.

#### Kernel design

```
for each element i:
    y[i] = a * x[i] + y[i]
```

- **Vector length**: `iterations × 100` elements (default: 100,000 × 100 = 10M elements)
- Each element = 1 FMA = 2 FLOPs

#### FLOP calculation

```
FLOPs = N × 2
```

Where N is the vector length.

#### Arithmetic intensity

| Precision | Bytes/element (read x + read y + write y) | FLOPs/element | AI (FLOP/Byte) |
|-----------|------------------------------------------|---------------|-----------------|
| FP64 | 24 | 2 | 0.083 |
| FP32 | 12 | 2 | 0.167 |
| FP16 | 6 | 2 | 0.333 |
| BF16 | 6 | 2 | 0.333 |

This is firmly in the memory-bound regime for all GPU architectures.

---

## Memory Kernels

### `stream_triad` — STREAM Triad (a = b + s·c)

**Source**: `src/kernels/vector/{stream_triad_cuda.cu,stream_triad_hip.cpp}`
**Category**: memory
**Precisions**: FP64, FP32
**Modes**: throughput
**Backends**: CUDA, HIP

#### What it measures

Sustained memory bandwidth using the STREAM Triad pattern, the standard HPC bandwidth benchmark. Reports GB/s (not FLOP/s).

#### Kernel design

```
for each element i:
    a[i] = b[i] + scalar * c[i]
```

Three arrays: 2 reads (b, c) + 1 write (a) = 3 memory operations per element.

- **Array size**: ~256 MB per array (total ~768 MB for 3 arrays)
- **Element count**: `256 × 1024 × 1024 / sizeof(element)`

#### Bandwidth calculation

```
bytes = 3 × N × sizeof(element)     // 2 reads + 1 write
GB/s  = bytes / median_time
```

#### Peak% reference

Compared against `HBM_BW` theoretical peak, computed as:
- `memoryClockRate × (memoryBusWidth / 8) × multiplier`
- Multiplier: ×2 (DDR) for HBM2/HBM2e, ×4 (QDR) for HBM3/HBM3e

### `stream_copy` — STREAM Copy (a = b)

**Source**: `src/kernels/vector/{stream_triad_cuda.cu,stream_triad_hip.cpp}`
**Category**: memory
**Precisions**: FP64
**Modes**: throughput
**Backends**: CUDA, HIP

#### What it measures

Pure copy bandwidth — no arithmetic. 1 read + 1 write per element.

```
for each element i:
    a[i] = b[i]
```

#### Bandwidth calculation

```
bytes = 2 × N × sizeof(double)     // 1 read + 1 write
GB/s  = bytes / median_time
```

---

## Matrix Kernels (GEMM)

All GEMM kernels compute C = α·A·B + β·C with α=1, β=0 (pure matrix multiply). They use vendor BLAS libraries (cuBLAS, cuBLASLt, rocBLAS, hipBLASLt) and are designed to exercise tensor cores / matrix cores.

### Common GEMM methodology

#### Auto-sweep for peak finding

GEMM performance varies significantly with matrix size. Small matrices underutilize the hardware; very large matrices may cause cache thrashing or memory pressure. All GEMM kernels sweep M=N=K across multiple sizes to find the hardware's peak rate:

| Backend | Sweep sizes |
|---------|-------------|
| CUDA (cuBLAS) | {1024, 2048, 4096, 8192, 16384} |
| CUDA (cuBLASLt) | {1024, 2048, 4096, 8192, 16384} |
| HIP (rocBLAS) | {1024, 2048, 4096, 8192, 16384, 32768} |
| HIP (hipBLASLt) | {1024, 2048, 4096, 8192, 16384, 32768} |

**Sweep phase**: 3 trials per size (quick measurement), report median
**Measurement phase**: full trial count (default 100) at the best size found in the sweep

#### Memory limit

GPU memory is checked before each sweep size. Sizes that would exceed 75% of free GPU memory (70% for hipBLASLt, which needs workspace) are skipped. This prevents OOM crashes on smaller-memory GPUs.

#### FLOP calculation (all GEMM kernels)

```
FLOPs = 2 × M × N × K
```

For square matrices (M = N = K): `FLOPs = 2 × M³`

#### Matrix layout

- **Standard GEMM**: Column-major (Fortran order), `OP_N` for both A and B
- **Block-scaled GEMM** (MXFP8, NVFP4): `OP_T` for A, `OP_N` for B (NVIDIA hardware requirement)
- **hipBLASLt**: `OP_N` for both A and B

---

### `gemm_cublas` — cuBLAS GEMM (Standard Precisions)

**Source**: `src/kernels/matrix/gemm_cublas.cu`
**Category**: matrix
**Precisions**: FP64, FP32, TF32, FP16, BF16
**Modes**: throughput
**Backend**: CUDA
**Availability**: All CUDA GPUs

#### Precision details

| Precision | Input Type | Output Type | Compute Type | Math Mode | Notes |
|-----------|-----------|-------------|--------------|-----------|-------|
| FP64 | `CUDA_R_64F` | `CUDA_R_64F` | `CUBLAS_COMPUTE_64F` | Default | Auto-uses FP64 tensor cores where available (IEEE-exact) |
| FP32 | `CUDA_R_32F` | `CUDA_R_32F` | `CUBLAS_COMPUTE_32F` | Default | CUDA cores only (no tensor core acceleration) |
| TF32 | `CUDA_R_32F` | `CUDA_R_32F` | `CUBLAS_COMPUTE_32F` | `CUBLAS_TF32_TENSOR_OP_MATH` | FP32 inputs, truncated to TF32 (10-bit mantissa) for tensor core multiply, FP32 accumulate |
| FP16 | `CUDA_R_16F` | `CUDA_R_16F` | `CUBLAS_COMPUTE_16F` | Default | FP16 tensor cores |
| BF16 | `CUDA_R_16BF` | `CUDA_R_16BF` | `CUBLAS_COMPUTE_32F` | Default | BF16 tensor cores with FP32 accumulate |

#### Peak% reference

- FP64: prefers `FP64_TC` peak if available, falls back to `FP64` (CUDA core)
- TF32: uses `TF32_TC` peak
- FP16: uses `FP16_TC` peak
- BF16: uses `BF16_TC` peak
- FP32: uses `FP32` (CUDA core) peak

---

### `gemm_cublas_notc` — cuBLAS GEMM (No Tensor Cores)

**Source**: `src/kernels/matrix/gemm_cublas.cu`
**Category**: matrix
**Precisions**: FP64, FP32
**Modes**: throughput
**Backend**: CUDA
**Availability**: All CUDA GPUs

Forces CUDA core path using `CUBLAS_PEDANTIC_MATH` to measure raw CUDA core GEMM performance without tensor core acceleration. Useful for comparing TC vs non-TC performance ratios.

**Note**: In practice, cuBLAS may still route through tensor cores on some architectures despite `PEDANTIC_MATH`. This results in peak% > 100% when compared against the CUDA core theoretical peak.

---

### `gemm_cublas_int8` — cuBLASLt INT8 GEMM

**Source**: `src/kernels/matrix/gemm_cublas_8bit.cu`
**Category**: matrix
**Precisions**: INT8
**Modes**: throughput
**Backend**: CUDA
**Availability**: sm_75+ (Turing and later)

#### Method

Uses `cublasLtMatmul` with heuristic auto-tuning:
1. Queries up to 16 heuristic solutions from cuBLASLt
2. Probes each solution with a test run, selects the fastest
3. Runs the full measurement with the best algorithm

| Property | Value |
|----------|-------|
| Input type | `CUDA_R_8I` (signed INT8) |
| Output type | `CUDA_R_32I` (INT32 accumulate) |
| Compute type | `CUBLAS_COMPUTE_32I` |
| Scale type | `CUDA_R_32I` |
| Workspace | 32 MB |

---

### `gemm_cublas_fp8` — cuBLASLt FP8 Per-Tensor Scaling

**Source**: `src/kernels/matrix/gemm_cublas_8bit.cu`
**Category**: matrix
**Precisions**: FP8_E4M3, FP8_E5M2
**Modes**: throughput
**Backend**: CUDA
**Availability**: sm_89–sm_99 (Ada Lovelace, Hopper)

#### Method

Uses `cublasLtMatmul` with per-tensor scaling factors (single scale per matrix):

| Property | FP8_E4M3 | FP8_E5M2 |
|----------|----------|----------|
| Input A | `CUDA_R_8F_E4M3` | `CUDA_R_8F_E5M2` |
| Input B | `CUDA_R_8F_E4M3` | `CUDA_R_8F_E5M2` |
| Output C/D | `CUDA_R_16BF` | `CUDA_R_16BF` |
| Compute | `CUBLAS_COMPUTE_32F` | `CUBLAS_COMPUTE_32F` |
| Scale pointers | A_scale, B_scale (device FP32) | A_scale, B_scale (device FP32) |

**Note**: Blackwell (sm_100+) requires block scaling instead — see `gemm_cublas_mxfp8`.

**Known issue**: Currently returns 0 GFLOP/s on all tested NVIDIA platforms. Under investigation.

---

### `gemm_cublas_mxfp8` — cuBLASLt MXFP8 Block-Scaled GEMM

**Source**: `src/kernels/matrix/gemm_cublas_blkscale.cu`
**Category**: matrix
**Precisions**: MXFP8
**Modes**: throughput
**Backend**: CUDA
**Availability**: sm_100+ (Blackwell)

#### Method

MX (Microscaling) FP8: each block of 32 elements along the K dimension shares a single E8M0 scale factor.

| Property | Value |
|----------|-------|
| Input type | `CUDA_R_8F_E4M3` |
| Output type | `CUDA_R_32F` |
| Compute type | `CUBLAS_COMPUTE_32F` |
| Scale type | `CUDA_R_8UF_E8M0` (`__nv_fp8_e8m0`) |
| Scale block size | 32 elements along K |
| Matrix layout | A transposed (`OP_T`), B normal (`OP_N`) |
| Scale value | All scales set to 1.0 (byte value 127 for E8M0) |

---

### `gemm_cublas_nvfp4` — cuBLASLt NVFP4 Block-Scaled GEMM

**Source**: `src/kernels/matrix/gemm_cublas_blkscale.cu`
**Category**: matrix
**Precisions**: NVFP4
**Modes**: throughput
**Backend**: CUDA
**Availability**: sm_100+ (Blackwell)

#### Method

NVIDIA FP4: 4-bit floating-point with block scaling. Each block of 16 elements along K shares an E4M3 scale factor.

| Property | Value |
|----------|-------|
| Input type | `CUDA_R_4F_E2M1` (packed, 2 elements per byte) |
| Output type | `CUDA_R_32F` |
| Compute type | `CUBLAS_COMPUTE_32F` |
| Scale type | `CUDA_R_8UF_E4M3` (`__nv_fp8_e4m3`, UE4M3 variant) |
| Scale block size | 16 elements along K |
| Matrix layout | A transposed (`OP_T`), B normal (`OP_N`) |
| Scale value | All scales set to 1.0 (byte value 0x38 for UE4M3) |

---

### `gemm_cublas_emu_fp32` — BF16×9 Emulated FP32 GEMM

**Source**: `src/kernels/matrix/gemm_cublas_emulated.cu`
**Category**: matrix
**Precisions**: FP32 (emulated)
**Modes**: throughput
**Backend**: CUDA
**Availability**: sm_90+ (Hopper, Blackwell)

#### Method

cuBLAS emulation: decomposes each FP32 value into 9 BF16 components and performs multiple BF16 tensor core GEMMs to reconstruct an FP32-accurate result. This trades compute for precision — using the much faster BF16 tensor cores to perform what would otherwise be a slow FP32 CUDA core operation.

| Property | Value |
|----------|-------|
| Compute type | `CUBLAS_COMPUTE_32F_EMULATED_16BFX9` (value 78) |
| Math mode | `CUBLAS_EMULATION_STRATEGY_EAGER` |
| Mantissa control | `CUDA_EMULATION_MANTISSA_CONTROL_DYNAMIC` |

**Peak% reference**: compared against FP32 CUDA core peak. Values >100% indicate the emulation is faster than native FP32.

**Known limitation**: On Hopper (sm_90), cuBLAS does not appear to engage emulation — returns the same rate as native FP32 GEMM.

---

### `gemm_cublas_emu_fp64` — Ozaki Emulated FP64 GEMM

**Source**: `src/kernels/matrix/gemm_cublas_emulated.cu`
**Category**: matrix
**Precisions**: FP64 (emulated)
**Modes**: throughput
**Backend**: CUDA
**Availability**: sm_100+ (Blackwell)

#### Method

cuBLAS Ozaki scheme: decomposes FP64 values into multiple fixed-point components and uses low-precision tensor core GEMMs to reconstruct an FP64-accurate result. This exploits the massive FP16/BF16 tensor core throughput to accelerate FP64 computation.

| Property | Value |
|----------|-------|
| Compute type | `CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT` (value 79) |
| Math mode | `CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH` (value 8) |

**Peak% reference**: compared against FP64 CUDA core peak. Values >100% indicate the Ozaki scheme outperforms native FP64.

---

### `gemm_rocblas` — rocBLAS GEMM

**Source**: `src/kernels/matrix/gemm_rocblas.cpp`
**Category**: matrix
**Precisions**: FP64, FP32, FP16, BF16, INT8
**Modes**: throughput
**Backend**: HIP
**Availability**: All AMD GPUs with rocBLAS

#### Precision details

| Precision | Input Type | Output Type | Compute Type | API |
|-----------|-----------|-------------|--------------|-----|
| FP64 | `rocblas_datatype_f64_r` | `rocblas_datatype_f64_r` | FP64 | `rocblas_dgemm` |
| FP32 | `rocblas_datatype_f32_r` | `rocblas_datatype_f32_r` | FP32 | `rocblas_sgemm` |
| FP16 | `rocblas_datatype_f16_r` | `rocblas_datatype_f16_r` | FP32 | `rocblas_gemm_ex` |
| BF16 | `rocblas_datatype_bf16_r` | `rocblas_datatype_bf16_r` | FP32 | `rocblas_gemm_ex` |
| INT8 | `rocblas_datatype_i8_r` | `rocblas_datatype_i32_r` | INT32 | `rocblas_gemm_ex` |

#### Peak% reference

Uses MFMA (Matrix Fused Multiply-Add) peak when available:
- FP64 → `FP64_MFMA` (if available, else `FP64` vector)
- FP32 → `FP32_MFMA`
- FP16 → `FP16_MFMA`
- BF16 → `BF16_MFMA`
- INT8 → `INT8_MFMA`

---

### `gemm_hipblaslt` — hipBLASLt GEMM (TF32, FP8)

**Source**: `src/kernels/matrix/gemm_hipblaslt.cpp`
**Category**: matrix
**Precisions**: TF32, FP8_E4M3, FP8_E5M2
**Modes**: throughput
**Backend**: HIP (optional — requires hipBLASLt library)
**Availability**: gfx940+ (CDNA3: MI300X, MI300A)

#### Method

Uses `hipblasLtMatmul` with heuristic solution search:
1. Queries up to 16 heuristic solutions
2. Probes each with a test run
3. Selects the fastest algorithm
4. Workspace: 32 MB

#### Precision details

| Precision | Input A/B | Output C/D | Compute | Scale | Notes |
|-----------|----------|-----------|---------|-------|-------|
| TF32 | `HIP_R_32F` | `HIP_R_32F` | `HIPBLAS_COMPUTE_32F_FAST_TF32` | `HIP_R_32F` | FP32 inputs, TF32 tensor math |
| FP8_E4M3 | `HIP_R_8F_E4M3_FNUZ` | `HIP_R_16BF` | `HIPBLAS_COMPUTE_32F` | `HIP_R_32F` | FNUZ variant (gfx942-specific) |
| FP8_E5M2 | `HIP_R_8F_E5M2_FNUZ` | `HIP_R_16BF` | `HIPBLAS_COMPUTE_32F` | `HIP_R_32F` | FNUZ variant (gfx942-specific) |

**Known issue**: FP8 E4M3/E5M2 return "No valid solution found" on ROCm 7.0.2. Likely requires a newer hipBLASLt version.

---

## Theoretical Peak Computation

### NVIDIA CUDA GPUs

Peaks are looked up from per-architecture tables (Volta through Blackwell) stored in `src/device/cuda_device_info.cu`:

```
peak_gflops = SMs × clock_MHz × ops_per_SM_per_clock × 2 (FMA) / 1000
```

Where `ops_per_SM_per_clock` varies by precision and architecture (CUDA cores vs tensor cores).

### AMD HIP GPUs

Peaks are computed from CDNA generation rates stored in `src/device/hip_device_info.cpp`:

```
peak_gflops = CUs × clock_MHz × ops_per_CU_per_clock / 1000
```

The per-CU rates already include the ×2 for FMA. Rates are stored for both vector (SIMD) and matrix (MFMA) execution units.

### CPU

```
peak_gflops = physical_cores × clock_MHz × SIMD_lanes × FMA_units × 2 / 1000
```

- Physical cores (not logical — SMT threads share FMA units)
- SIMD_lanes: 8 (AVX-512 FP64), 16 (AVX-512 FP32), 4 (AVX2 FP64), 8 (AVX2 FP32)
- FMA_units: 2 (typical for modern x86)

---

## Architecture Availability Matrix

Each kernel declares which architectures it supports via `is_available(device)`. The dispatch loop skips unavailable kernels. Kernels that run but produce 0 GFLOP/s are also excluded from the final report.

| Kernel | CUDA (all) | sm_89–99 | sm_100+ | HIP (all) | gfx94x |
|--------|-----------|----------|---------|-----------|--------|
| scalar_fma | ✅ | ✅ | ✅ | ✅ | ✅ |
| vector_axpy | ✅ | ✅ | ✅ | ✅ | ✅ |
| stream_triad | ✅ | ✅ | ✅ | ✅ | ✅ |
| stream_copy | ✅ | ✅ | ✅ | ✅ | ✅ |
| gemm_cublas | ✅ | ✅ | ✅ | — | — |
| gemm_cublas_notc | ✅ | ✅ | ✅ | — | — |
| gemm_cublas_int8 | ✅ | ✅ | ✅ | — | — |
| gemm_cublas_fp8 | — | ✅ | — | — | — |
| gemm_cublas_mxfp8 | — | — | ✅ | — | — |
| gemm_cublas_nvfp4 | — | — | ✅ | — | — |
| gemm_cublas_emu_fp32 | — | ✅ | ✅ | — | — |
| gemm_cublas_emu_fp64 | — | — | ✅ | — | — |
| gemm_rocblas | — | — | — | ✅ | ✅ |
| gemm_hipblaslt | — | — | — | — | ✅ |

---

## Report Output

Results are written in three formats:

| Format | Flag | Content |
|--------|------|---------|
| Console | (always) | Box-drawing table with fixed-width columns |
| JSON | `--output=<path>` | Structured data with system info, device specs, all measurements |
| Markdown | `--output-md=<path>` | Human-readable tables with system info, suitable for documentation |

### Table columns

| Column | Description |
|--------|-------------|
| Kernel | Kernel name (e.g., `gemm_cublas`, `scalar_fma`) |
| Prec | Precision (FP64, FP32, TF32, FP16, BF16, INT8, etc.) |
| Mode | `throughput` or `latency` |
| Rate | SI-prefixed rate with suffix: `GF/s`, `TF/s`, `PF/s` (compute) or `TB/s` (memory) |
| Peak% | Percentage of theoretical peak |
| Median (ms) | Median execution time across all trials |

---

*Generated from [Floptic](https://github.com/jtchilders/floptic) source, commit `7583516`*
