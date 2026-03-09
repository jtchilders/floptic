# Floptic Implementation Plan

## Project Structure

```
floptic/
├── README.md
├── CMakeLists.txt                  # Top-level CMake
├── cmake/
│   ├── FindKokkos.cmake
│   ├── DetectBackends.cmake        # Auto-detect CUDA, HIP, SYCL, OpenMP
│   └── CompilerFlags.cmake         # Per-backend optimization flags
├── docs/
│   ├── IMPLEMENTATION_PLAN.md      # This file
│   ├── output-format.md            # JSON report schema
│   └── adding-kernels.md           # Guide for adding new benchmarks
├── src/
│   ├── main.cpp                    # CLI entry point
│   ├── cli/
│   │   ├── cli_parser.hpp          # Argument parsing
│   │   └── cli_parser.cpp
│   ├── device/
│   │   ├── device_info.hpp         # Abstract device info interface
│   │   ├── device_info.cpp
│   │   ├── cuda_device_info.cpp    # NVIDIA-specific capabilities
│   │   ├── hip_device_info.cpp     # AMD GPU-specific
│   │   ├── sycl_device_info.cpp    # Intel GPU-specific
│   │   └── cpu_device_info.cpp     # CPU feature detection (AVX, AMX, etc.)
│   ├── kernels/
│   │   ├── kernel_base.hpp         # Abstract kernel interface
│   │   ├── kernel_registry.hpp     # Kernel registration & discovery
│   │   ├── kernel_registry.cpp
│   │   ├── scalar/
│   │   │   ├── scalar_fma.hpp      # FMA throughput (latency & throughput modes)
│   │   │   ├── scalar_div.hpp      # Division throughput
│   │   │   ├── scalar_sqrt.hpp     # Square root
│   │   │   └── scalar_transcendental.hpp  # sin, cos, exp, log
│   │   ├── vector/
│   │   │   ├── vector_fma.hpp      # Streaming FMA
│   │   │   ├── vector_dot.hpp      # Dot product (reduction)
│   │   │   └── vector_axpy.hpp     # AXPY (BLAS Level 1)
│   │   ├── matrix/
│   │   │   ├── gemm_naive.hpp      # Hand-written GEMM (baseline)
│   │   │   ├── gemm_vendor.hpp     # cuBLAS / rocBLAS / MKL dispatch
│   │   │   └── gemm_tensor.hpp     # Tensor core / matrix engine GEMM
│   │   ├── sparse/
│   │   │   ├── spmv_csr.hpp        # SpMV in CSR format
│   │   │   ├── spmv_structured.hpp # 2:4 structured sparsity (NVIDIA)
│   │   │   └── sparse_gemm.hpp     # Sparse × Dense GEMM
│   │   └── emulated/
│   │       ├── dd_arithmetic.hpp   # Double-double (Bailey) kernels
│   │       └── ozaki_gemm.hpp      # Ozaki scheme GEMM
│   ├── harness/
│   │   ├── timer.hpp               # High-resolution timing (GPU events, CPU clocks)
│   │   ├── warmup.hpp              # Warmup strategy
│   │   ├── validator.hpp           # Correctness check (prevent dead code elimination)
│   │   └── power_monitor.hpp       # Power sampling (nvidia-smi, rocm-smi, RAPL)
│   └── report/
│       ├── report.hpp              # Report data structures
│       ├── json_writer.hpp         # JSON output
│       └── csv_writer.hpp          # CSV output
├── tests/
│   ├── test_scalar_fma.cpp
│   ├── test_device_info.cpp
│   └── test_report.cpp
└── scripts/
    ├── run_full_sweep.sh           # Run all benchmarks, all precisions
    ├── compare_devices.py          # Compare reports across devices
    └── plot_results.py             # Generate charts from reports
```

---

## Phased Implementation

### Phase 1: Foundation (Weeks 1-2)
**Goal**: Build system, device detection, measurement harness, one working kernel

#### 1.1 Build System
- [ ] Top-level `CMakeLists.txt` with Kokkos integration
- [ ] Backend auto-detection (`DetectBackends.cmake`)
  - Check for CUDA toolkit → enable CUDA backend
  - Check for HIP → enable HIP backend
  - Check for oneAPI/SYCL → enable SYCL backend
  - Always enable OpenMP for CPU
- [ ] Conditional compilation guards for backend-specific code

#### 1.2 CLI & Driver
- [ ] `cli_parser`: Parse command-line arguments
  - `--device=<cpu|cuda:N|hip:N|sycl:N|all>` — target device(s)
  - `--precision=<fp64|fp32|fp16|bf16|fp8|all>` — precisions to test
  - `--kernels=<scalar|vector|matrix|sparse|all>` — kernel categories
  - `--iterations=<N>` — measurement iterations (default: 100)
  - `--warmup=<N>` — warmup iterations (default: 10)
  - `--report=<json|csv|stdout>` — output format
  - `--output=<path>` — output file (default: stdout)
  - `--sizes=<S,M,L>` — problem sizes (kernel-dependent)
  - `--clock-lock` — attempt to lock GPU clocks (requires permissions)
- [ ] Main driver loop: enumerate devices → select kernels → run → report

#### 1.3 Device Discovery
- [ ] Abstract `DeviceInfo` interface:
  ```cpp
  struct DeviceInfo {
      std::string name;
      std::string vendor;         // "nvidia", "amd", "intel"
      std::string arch;           // "sm_90", "gfx942", "pvc"
      std::string type;           // "gpu", "cpu"
      size_t memory_bytes;
      int compute_units;          // SMs, CUs, EUs
      std::vector<Precision> supported_precisions;
      std::vector<Feature> features;  // tensor_cores, amx, mfma, etc.
      std::map<Precision, double> theoretical_peak_gflops;
  };
  ```
- [ ] CUDA implementation: `cudaGetDeviceProperties`, compute capability parsing
- [ ] CPU implementation: CPUID for AVX2, AVX-512, AMX detection

#### 1.4 Measurement Harness
- [ ] `Timer`:
  - GPU: `cudaEvent_t` / `hipEvent_t` based (avoids sync overhead)
  - CPU: `std::chrono::high_resolution_clock` or `rdtsc`
  - Report median of N trials (not mean — avoids outlier skew)
- [ ] `Warmup`: Run kernel N times before measurement, wait for GPU boost clocks to stabilize
- [ ] `Validator`: Accumulate results into a volatile sink to prevent dead code elimination;
      optionally compare against a reference value for correctness

#### 1.5 First Kernel: `scalar_fma`
- [ ] Implement for FP64, FP32, FP16 via Kokkos
- [ ] Template on precision type
- [ ] Measure both throughput (independent FMAs) and latency (dependent chain)
- [ ] Validate output, produce first JSON report

**Phase 1 Deliverable**: `./floptic --device=cuda:0 --kernels=scalar --precision=fp64,fp32,fp16` produces a valid JSON report

---

### Phase 2: Scalar & Vector Kernels (Weeks 3-4)
**Goal**: Complete scalar suite, add vector kernels, CPU backend

#### 2.1 Remaining Scalar Kernels
- [ ] `scalar_div` — division throughput/latency
- [ ] `scalar_sqrt` — square root
- [ ] `scalar_transcendental` — sin, cos, exp, log
  - Note: GPU transcendentals may use SFU (special function units)
  - Measure `__sinf` vs `sin` on CUDA (fast-math vs IEEE)

#### 2.2 Vector Kernels
- [ ] `vector_fma` — streaming `c[i] = a[i] * b[i] + c[i]`
  - Vary vector length: 1K, 1M, 100M elements
  - Report FLOP/s and effective bandwidth (to distinguish compute vs memory bound)
- [ ] `vector_dot` — reduction pattern
  - Tests warp/wavefront reduction on GPU, SIMD reduction on CPU
- [ ] `vector_axpy` — BLAS Level 1 reference

#### 2.3 CPU OpenMP Backend
- [ ] Implement scalar/vector kernels with OpenMP parallel regions
- [ ] Auto-detect SIMD width and report
- [ ] Pin threads to cores for consistent results
- [ ] Detect and report AVX2 vs AVX-512 usage

#### 2.4 Precision Type System
- [ ] Define precision types that work across backends:
  ```cpp
  enum class Precision {
      FP64, FP32, FP16, BF16, TF32,
      FP8_E4M3, FP8_E5M2, FP4,
      INT8, INT4,
      DD_FP32,   // double-double via FP32 pairs (emulated FP64)
      DD_FP64    // double-double via FP64 pairs (emulated FP128)
  };
  ```
- [ ] Traits class mapping Precision → C++ type, bytes, FLOPs-per-op semantics

**Phase 2 Deliverable**: Scalar + vector benchmarks on CUDA and CPU, all standard precisions

---

### Phase 3: Dense Matrix Kernels (Weeks 5-6)
**Goal**: GEMM benchmarks including tensor cores, vendor BLAS comparison

#### 3.1 Naive GEMM
- [ ] Simple triple-loop GEMM (Kokkos parallel)
- [ ] Serves as baseline — how far from peak without vendor tuning?
- [ ] Sizes: 256², 1024², 2048², 4096², 8192²

#### 3.2 Vendor BLAS GEMM
- [ ] cuBLAS dispatch: `cublasDgemm`, `cublasSgemm`, `cublasHgemm`, `cublasGemmEx`
- [ ] rocBLAS dispatch (AMD)
- [ ] MKL dispatch (Intel CPU)
- [ ] Abstract wrapper: `vendor_gemm(A, B, C, precision, device)`
- [ ] Each call reports: GFLOP/s, time, % of theoretical peak

#### 3.3 Tensor Core GEMM (NVIDIA)
- [ ] WMMA API kernels for:
  - FP16 → FP16 accumulate
  - FP16 → FP32 accumulate
  - BF16 → FP32 accumulate
  - TF32 → FP32 accumulate
  - FP64 → FP64 (Ampere+)
  - FP8 → FP32 (Hopper+)
  - INT8 → INT32
- [ ] Also benchmark via `cublasGemmEx` with `CUBLAS_COMPUTE_*` flags
      (easier, and what most users will actually call)
- [ ] Report tensor core utilization metrics if available

#### 3.4 AMD Matrix Core GEMM
- [ ] MFMA intrinsics for MI-series GPUs
- [ ] FP16, BF16, FP8 (MI300+), INT8
- [ ] Also via rocBLAS with compute type flags

#### 3.5 Intel AMX (CPU)
- [ ] Tile-based BF16 and INT8 matmul via AMX intrinsics
- [ ] Compare against MKL auto-dispatch

**Phase 3 Deliverable**: GEMM benchmarks across all precision×hardware combinations, tensor core characterization

---

### Phase 4: Sparse & Emulated Kernels (Weeks 7-8)
**Goal**: Sparse operations, FP64 emulation schemes

#### 4.1 Sparse Kernels
- [ ] `spmv_csr` — SpMV with standard CSR format
  - Generate synthetic sparse matrices (diagonal, banded, random)
  - Report FLOP/s and bandwidth (sparse ops are typically memory-bound)
- [ ] `spmv_structured` — NVIDIA 2:4 structured sparsity (Ampere+)
  - Compare vs dense and vs CSR
- [ ] `sparse_gemm` — Sparse × Dense via cuSPARSE / rocSPARSE

#### 4.2 Emulated Precision Kernels
- [ ] `dd_arithmetic` — Double-double arithmetic (Bailey)
  - dd_add, dd_mul, dd_div using TwoSum / TwoProduct EFTs
  - FP32 pairs → emulated FP64 throughput
  - FP64 pairs → emulated FP128 throughput
  - Key metric: emulated GFLOP/s vs native, and precision achieved
- [ ] `ozaki_gemm` — Ozaki scheme GEMM
  - Split FP64 → multiple FP16/FP32 components
  - Multiply via tensor cores
  - Recombine for FP64 result
  - Compare against native FP64 GEMM
  - Report: throughput, precision loss (ULP error), energy

#### 4.3 Precision Verification
- [ ] For emulated kernels, also measure **accuracy**:
  - Run a reference computation in FP128 (quadmath on CPU)
  - Compare emulated result: max ULP error, RMS error, significant digits
  - Include in report as `"accuracy": { "max_ulp": ..., "sig_digits": ... }`

**Phase 4 Deliverable**: Full kernel suite including emulation, accuracy characterization

---

### Phase 5: Reporting, Analysis & Polish (Weeks 9-10)
**Goal**: Publication-quality output, comparison tools, documentation

#### 5.1 Report Enhancements
- [ ] Summary table: device × precision × kernel_type → GFLOP/s
- [ ] Precision ratio table: FP64:FP32:FP16:FP8 for each kernel type
- [ ] Roofline data: include memory bandwidth measurements for roofline plots
- [ ] Power/energy: GFLOP/s per watt column

#### 5.2 Analysis Scripts
- [ ] `compare_devices.py`:
  - Load multiple JSON reports
  - Side-by-side comparison tables
  - Highlight precision ratios across architectures
- [ ] `plot_results.py`:
  - Bar charts: GFLOP/s by precision for each kernel
  - Heatmap: device × precision matrix
  - Roofline plots with measured data points
  - Scaling plots: GFLOP/s vs matrix size

#### 5.3 Documentation
- [ ] `output-format.md` — Full JSON schema documentation
- [ ] `adding-kernels.md` — How to add a new kernel to Floptic
- [ ] Architecture-specific notes (what to expect on A100 vs H100 vs B200 vs MI300)

#### 5.4 CI / Testing
- [ ] Unit tests for non-device code (report generation, CLI parsing)
- [ ] Integration tests on available hardware (GitHub Actions for CPU, manual for GPU)
- [ ] Regression tracking: store reference results, flag anomalies

**Phase 5 Deliverable**: Release-ready toolkit with documentation and analysis tools

---

## Kernel Design Principles

### 1. Prevent Dead Code Elimination
Every kernel must consume its result in a way the compiler cannot optimize away:
```cpp
// Bad: compiler may eliminate entire loop
for (int i = 0; i < N; i++) result = fma(a, b, result);

// Good: accumulate and write to volatile or global
volatile double sink;
for (int i = 0; i < N; i++) result = fma(a, b, result);
sink = result;
```

### 2. Throughput vs Latency Modes
Each kernel should have two modes:
- **Throughput**: Independent operations (fills pipelines)
  ```cpp
  // 8 independent FMA chains
  r0 = fma(a, b, r0); r1 = fma(a, b, r1); ... r7 = fma(a, b, r7);
  ```
- **Latency**: Dependent chain (measures instruction latency)
  ```cpp
  // Serial dependency
  r = fma(a, b, r); r = fma(a, b, r); r = fma(a, b, r);
  ```

### 3. Sufficient Work
- GPU kernels: Launch enough threads to fill all SMs/CUs
- CPU kernels: Use all cores, fill SIMD lanes
- Matrix kernels: Large enough to amortize launch overhead
- Rule of thumb: kernel should run ≥10ms per trial

### 4. FLOP Counting
Consistent definitions:
- FMA = 2 FLOPs (multiply + add)
- Division = 1 FLOP
- GEMM (M×N×K) = 2×M×N×K FLOPs
- SpMV (NNZ non-zeros) = 2×NNZ FLOPs

---

## Platform-Specific Notes

### NVIDIA GPUs
| Architecture | Compute Cap | Notable Features |
|-------------|------------|------------------|
| Volta (V100) | 7.0 | FP64 tensor cores (limited), 1:2 FP64:FP32 |
| Ampere (A100) | 8.0 | FP64 tensor cores, TF32, BF16, structured sparsity |
| Hopper (H100) | 9.0 | FP8, FP64 on tensor cores (limited), TMA |
| Blackwell (B200) | 10.0 | FP4, FP6, removed IEEE tensor core formats, Ozaki reference impl |

### AMD GPUs
| Architecture | Notable Features |
|-------------|------------------|
| CDNA2 (MI250) | MFMA FP64/FP32/FP16/BF16/INT8, 1:2 FP64:FP32 |
| CDNA3 (MI300) | + FP8, improved MFMA, APU (HBM + CPU) |

### Intel
| Platform | Notable Features |
|----------|------------------|
| Sapphire Rapids (CPU) | AMX (BF16, INT8), AVX-512 |
| Ponte Vecchio (GPU) | XMX (systolic) FP64/FP32/FP16/BF16/INT8 |

---

## Report JSON Schema (Summary)

```json
{
  "floptic_version": "0.1.0",
  "timestamp": "ISO-8601",
  "system": {
    "hostname": "string",
    "os": "string",
    "compiler": "string",
    "kokkos_version": "string"
  },
  "devices": [
    {
      "id": "cuda:0",
      "name": "NVIDIA A100-SXM4-80GB",
      "vendor": "nvidia",
      "arch": "sm_80",
      "type": "gpu",
      "memory_gb": 80.0,
      "compute_units": 108,
      "features": ["tensor_cores", "fp64_tensor", "structured_sparsity"],
      "supported_precisions": ["FP64", "FP32", "TF32", "FP16", "BF16", "INT8"],
      "theoretical_peak_gflops": {
        "FP64": 9700, "FP32": 19500, "TF32_TC": 156000,
        "FP16_TC": 312000, "BF16_TC": 312000, "INT8_TC": 624000
      }
    }
  ],
  "benchmarks": [
    {
      "device_id": "cuda:0",
      "kernel": "scalar_fma",
      "category": "scalar",
      "precision": "FP64",
      "mode": "throughput",
      "problem_size": {},
      "iterations": 100,
      "results": {
        "gflops": 9650.0,
        "peak_percent": 99.5,
        "median_time_ms": 1.035,
        "min_time_ms": 1.030,
        "max_time_ms": 1.089,
        "power_watts": 385.0,
        "gflops_per_watt": 25.06
      },
      "accuracy": null
    }
  ],
  "summary": {
    "precision_ratios": {
      "FP32_to_FP64": 2.01,
      "FP16_TC_to_FP64": 32.16,
      "FP8_TC_to_FP64": 64.33
    }
  }
}
```

---

## Dependencies

| Dependency | Purpose | Required? |
|-----------|---------|-----------|
| Kokkos | Portable parallel kernels | Yes |
| CMake ≥ 3.22 | Build system | Yes |
| CUDA Toolkit | NVIDIA GPU backend | Optional |
| ROCm / HIP | AMD GPU backend | Optional |
| oneAPI / SYCL | Intel backend | Optional |
| cuBLAS | Vendor GEMM (NVIDIA) | Optional |
| rocBLAS | Vendor GEMM (AMD) | Optional |
| MKL | Vendor GEMM (Intel) | Optional |
| nlohmann/json | JSON report output | Yes (header-only) |
| Python 3.8+ | Analysis scripts | Optional |
| matplotlib | Plotting | Optional |

---

## Open Questions for Discussion

1. **Kokkos-only or hybrid?**
   - Kokkos handles scalar/vector/naive-GEMM portably
   - Tensor core WMMA, AMX intrinsics require native code
   - Proposal: Kokkos for portable kernels, `#ifdef` native backends for specialty

2. **How to handle precisions Kokkos doesn't natively support?**
   - FP16/BF16: Kokkos has `Kokkos::Experimental::half_t`
   - FP8: No Kokkos support — need CUDA `__nv_fp8_e4m3` directly
   - TF32: Only via cuBLAS compute mode, not a real type

3. **Clock management**:
   - Lock GPU clocks? (`nvidia-smi -lgc`) Requires root/admin
   - Or sample clock frequency and report alongside GFLOP/s

4. **Multiple GPUs**:
   - Benchmark each independently? Or include multi-GPU scaling?
   - Phase 1: single device. Future: multi-device.

5. **Emulated precision baselines**:
   - Should dd-FP32 scalar FMA be directly comparable to native FP64?
   - Need to define "effective GFLOP/s" for emulated operations
   - Proposal: report raw throughput AND equivalent FP64 GFLOP/s
