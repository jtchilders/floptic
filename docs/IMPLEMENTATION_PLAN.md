# Floptic Implementation Plan

## Design Decisions

1. **No Kokkos** — Use templated C++ with compile-time backend selection.
   Hardware-specific code (tensor cores, AMX, MFMA) is inevitable, so we
   use a clean abstraction layer with backend-specific implementations rather
   than fighting a portability framework.
2. **User-level privileges only** — No clock locking, no root-level power APIs.
   Sample and report clock frequencies and available power data opportunistically.
3. **Emulated precision reports both raw and effective GFLOP/s** — e.g., an
   Ozaki GEMM uses many FP16 tensor core ops internally; report the raw
   FP16 GFLOP/s *and* the effective "equivalent FP64" GFLOP/s for direct
   comparison with native.
4. **Single device only** — No multi-GPU benchmarks.

---

## Project Structure

```
floptic/
├── README.md
├── LICENSE
├── CMakeLists.txt                      # Top-level CMake
├── cmake/
│   ├── DetectBackends.cmake            # Auto-detect CUDA, HIP, SYCL, OpenMP
│   └── CompilerFlags.cmake             # Per-backend optimization flags
├── docs/
│   ├── IMPLEMENTATION_PLAN.md          # This file
│   ├── output-format.md               # JSON report schema
│   └── adding-kernels.md              # Guide for adding new benchmarks
├── include/
│   └── floptic/
│       ├── floptic.hpp                 # Master include
│       ├── precision.hpp               # Precision enum, traits, type mapping
│       ├── device_info.hpp             # Abstract device info interface
│       ├── kernel_base.hpp             # Abstract kernel interface
│       ├── kernel_registry.hpp         # Kernel registration & discovery
│       ├── timer.hpp                   # High-resolution timing abstraction
│       ├── result.hpp                  # Benchmark result data structures
│       └── report.hpp                  # Report generation interface
├── src/
│   ├── main.cpp                        # CLI entry point & driver
│   ├── cli_parser.cpp                  # Argument parsing
│   ├── kernel_registry.cpp             # Registry implementation
│   │
│   ├── device/                         # Device discovery backends
│   │   ├── cpu_device_info.cpp         # CPU: CPUID, /proc/cpuinfo
│   │   ├── cuda_device_info.cu         # NVIDIA: cudaGetDeviceProperties
│   │   ├── hip_device_info.cpp         # AMD GPU: hipGetDeviceProperties
│   │   └── sycl_device_info.cpp        # Intel: sycl::device queries
│   │
│   ├── kernels/                        # Benchmark kernels
│   │   ├── scalar/
│   │   │   ├── scalar_fma_cuda.cu      # CUDA scalar FMA
│   │   │   ├── scalar_fma_hip.cpp      # HIP scalar FMA
│   │   │   ├── scalar_fma_cpu.cpp      # CPU scalar FMA (OpenMP + intrinsics)
│   │   │   ├── scalar_div_cuda.cu
│   │   │   ├── scalar_div_hip.cpp
│   │   │   ├── scalar_div_cpu.cpp
│   │   │   ├── scalar_sqrt_cuda.cu
│   │   │   ├── scalar_sqrt_hip.cpp
│   │   │   ├── scalar_sqrt_cpu.cpp
│   │   │   ├── scalar_transcendental_cuda.cu   # sin, cos, exp, log
│   │   │   ├── scalar_transcendental_hip.cpp
│   │   │   └── scalar_transcendental_cpu.cpp
│   │   │
│   │   ├── vector/
│   │   │   ├── vector_fma_cuda.cu
│   │   │   ├── vector_fma_hip.cpp
│   │   │   ├── vector_fma_cpu.cpp
│   │   │   ├── vector_dot_cuda.cu
│   │   │   ├── vector_dot_hip.cpp
│   │   │   ├── vector_dot_cpu.cpp
│   │   │   ├── vector_axpy_cuda.cu
│   │   │   ├── vector_axpy_hip.cpp
│   │   │   └── vector_axpy_cpu.cpp
│   │   │
│   │   ├── matrix/
│   │   │   ├── gemm_naive_cuda.cu      # Hand-written (baseline)
│   │   │   ├── gemm_naive_hip.cpp
│   │   │   ├── gemm_naive_cpu.cpp
│   │   │   ├── gemm_cublas.cu          # cuBLAS (NVIDIA)
│   │   │   ├── gemm_rocblas.cpp        # rocBLAS (AMD)
│   │   │   ├── gemm_mkl.cpp           # MKL (Intel CPU)
│   │   │   ├── gemm_wmma.cu           # NVIDIA tensor core via WMMA
│   │   │   ├── gemm_mfma.cpp          # AMD matrix core via MFMA
│   │   │   └── gemm_amx.cpp           # Intel AMX tiles
│   │   │
│   │   ├── sparse/
│   │   │   ├── spmv_csr_cuda.cu
│   │   │   ├── spmv_csr_hip.cpp
│   │   │   ├── spmv_csr_cpu.cpp
│   │   │   ├── spmv_structured_cuda.cu # NVIDIA 2:4 sparsity
│   │   │   ├── sparse_gemm_cusparse.cu
│   │   │   └── sparse_gemm_rocsparse.cpp
│   │   │
│   │   └── emulated/
│   │       ├── ozaki_gemm_cuda.cu      # Ozaki scheme via tensor cores
│   │       └── ozaki_gemm_hip.cpp      # Ozaki scheme via matrix cores
│   │
│   ├── harness/
│   │   ├── timer_cuda.cu              # cudaEvent-based timing
│   │   ├── timer_hip.cpp              # hipEvent-based timing
│   │   ├── timer_cpu.cpp              # chrono-based timing
│   │   ├── warmup.cpp                 # Warmup strategy
│   │   ├── validator.cpp              # Anti-DCE & correctness checking
│   │   └── power_monitor.cpp          # Best-effort power sampling
│   │
│   └── report/
│       ├── json_writer.cpp            # JSON output
│       └── csv_writer.cpp             # CSV output
│
├── tests/
│   ├── test_precision.cpp
│   ├── test_device_info.cpp
│   ├── test_report.cpp
│   └── test_validator.cpp
│
└── scripts/
    ├── run_full_sweep.sh              # Run all benchmarks, all precisions
    ├── compare_devices.py             # Compare reports across devices
    └── plot_results.py                # Generate charts from reports
```

---

## Core Abstractions

### Precision Type System

```cpp
// include/floptic/precision.hpp

enum class Precision {
    FP64, FP32, FP16, BF16, TF32,
    FP8_E4M3, FP8_E5M2, FP4,
    INT8, INT4
};

// Maps Precision → native C++ type, size, FLOP semantics
template <Precision P> struct PrecisionTraits;

template <> struct PrecisionTraits<Precision::FP64> {
    using type = double;
    static constexpr size_t bytes = 8;
    static constexpr const char* name = "FP64";
    static constexpr int fma_flops = 2;         // multiply + add
};

template <> struct PrecisionTraits<Precision::FP32> {
    using type = float;
    static constexpr size_t bytes = 4;
    static constexpr const char* name = "FP32";
    static constexpr int fma_flops = 2;
};

template <> struct PrecisionTraits<Precision::FP16> {
    // using type = __half;  // CUDA-specific, handled per-backend
    static constexpr size_t bytes = 2;
    static constexpr const char* name = "FP16";
    static constexpr int fma_flops = 2;
};
```

### Kernel Interface

```cpp
// include/floptic/kernel_base.hpp

struct KernelConfig {
    Precision precision;
    std::string mode;           // "throughput" or "latency"
    std::map<std::string, size_t> sizes;  // kernel-specific params
};

struct KernelResult {
    double gflops;              // measured GFLOP/s
    double effective_gflops;    // for emulated: equivalent target-precision GFLOP/s
                                // for native: same as gflops
    double peak_percent;        // % of theoretical peak
    double median_time_ms;
    double min_time_ms;
    double max_time_ms;
    double clock_mhz;           // sampled clock (best-effort)
    double power_watts;         // sampled power (best-effort, 0 if unavailable)
    int64_t total_flops;        // total FLOPs executed
    bool accuracy_measured;
    double max_ulp_error;       // vs reference (emulated kernels only)
    double sig_digits;          // significant correct digits (emulated only)
};

class KernelBase {
public:
    virtual ~KernelBase() = default;

    virtual std::string name() const = 0;
    virtual std::string category() const = 0;   // "scalar", "vector", "matrix", "sparse", "emulated"
    virtual std::vector<Precision> supported_precisions() const = 0;
    virtual std::vector<std::string> supported_modes() const = 0;

    virtual KernelResult run(const KernelConfig& config) = 0;
};
```

### Kernel Registration

```cpp
// include/floptic/kernel_registry.hpp

class KernelRegistry {
public:
    static KernelRegistry& instance();

    void register_kernel(std::unique_ptr<KernelBase> kernel);
    std::vector<KernelBase*> get_kernels(
        const std::string& category = "",
        Precision precision = Precision::FP64
    ) const;
    KernelBase* get_kernel(const std::string& name) const;
};

// Macro for self-registering kernels at static init
#define REGISTER_KERNEL(KernelClass) \
    static bool _reg_##KernelClass = [] { \
        KernelRegistry::instance().register_kernel( \
            std::make_unique<KernelClass>()); \
        return true; \
    }()
```

### Device Info

```cpp
// include/floptic/device_info.hpp

enum class Feature {
    TENSOR_CORES,           // NVIDIA tensor cores
    FP64_TENSOR,            // FP64 on tensor cores (Ampere+)
    STRUCTURED_SPARSITY,    // NVIDIA 2:4 sparsity
    MFMA,                   // AMD matrix fused multiply-add
    AMX_BF16,               // Intel AMX BF16 tiles
    AMX_INT8,               // Intel AMX INT8 tiles
    AVX2,                   // 256-bit SIMD
    AVX512,                 // 512-bit SIMD
    AVX512_FP16,            // AVX-512 FP16 support
    FMA_HW,                 // Hardware FMA
};

struct DeviceInfo {
    std::string id;             // "cuda:0", "cpu:0", "hip:0"
    std::string name;           // "NVIDIA A100-SXM4-80GB"
    std::string vendor;         // "nvidia", "amd", "intel"
    std::string arch;           // "sm_80", "gfx942", "sapphire_rapids"
    std::string type;           // "gpu", "cpu"
    size_t memory_bytes;
    int compute_units;          // SMs, CUs, EUs, cores
    int clock_mhz;              // base clock
    int boost_clock_mhz;        // boost clock
    std::vector<Precision> supported_precisions;
    std::vector<Feature> features;
    std::map<Precision, double> theoretical_peak_gflops;
};

// Factory: returns appropriate DeviceInfo for this system
std::vector<DeviceInfo> discover_devices();
```

---

## Phased Implementation

### Phase 1: Foundation (Weeks 1-2)
**Goal**: Build system, device detection, measurement harness, one working kernel

#### 1.1 Build System
- [ ] Top-level `CMakeLists.txt`
  - C++17 minimum
  - Backend detection: find CUDA toolkit, ROCm/HIP, oneAPI
  - Compile `.cu` files with nvcc, `.cpp` with host compiler
  - Conditional targets: `floptic_cuda`, `floptic_hip`, `floptic_cpu`
  - Single binary links available backends
  - Include nlohmann/json as header-only dependency (FetchContent or vendored)
- [ ] `DetectBackends.cmake`
  - `find_package(CUDAToolkit)` → set `FLOPTIC_HAS_CUDA`
  - `find_package(hip)` → set `FLOPTIC_HAS_HIP`
  - `find_package(IntelSYCL)` → set `FLOPTIC_HAS_SYCL`
  - OpenMP always attempted
  - Print summary of detected backends at configure time
- [ ] `CompilerFlags.cmake`
  - CUDA: `-arch=native` or user-specified `-arch=sm_XX`
  - HIP: `--offload-arch=gfxXXX`
  - CPU: `-march=native -O3 -fopenmp`

#### 1.2 CLI & Driver
- [ ] `cli_parser` using simple getopt or lightweight header-only lib
  - `--device=<cpu|cuda:N|hip:N|all>` — target device(s)
  - `--precision=<fp64|fp32|fp16|bf16|fp8|all>` — precisions to test
  - `--kernels=<scalar|vector|matrix|sparse|emulated|all>` — kernel categories
  - `--kernel=<name>` — run a specific kernel by name
  - `--iterations=<N>` — measurement iterations (default: 100)
  - `--warmup=<N>` — warmup iterations (default: 10)
  - `--report=<json|csv|stdout>` — output format
  - `--output=<path>` — output file (default: stdout)
  - `--sizes=<S,M,L>` — problem sizes (kernel-dependent defaults)
  - `--list` — list available kernels and exit
  - `--info` — print device info and exit
- [ ] Main driver loop:
  1. Parse args
  2. Discover devices, filter by `--device`
  3. Query kernel registry, filter by `--kernels` / `--precision`
  4. For each (device, kernel, precision, mode): warmup → measure → collect
  5. Generate report

#### 1.3 Device Discovery
- [ ] `cpu_device_info.cpp`:
  - Parse `/proc/cpuinfo` (Linux) or sysctl (macOS)
  - CPUID intrinsics for AVX2, AVX-512, AMX, FMA detection
  - Calculate theoretical peaks from core count × clock × SIMD width × FMA
- [ ] `cuda_device_info.cu`:
  - `cudaGetDeviceProperties` for name, memory, SMs, clocks
  - Compute capability → feature mapping (tensor cores, FP64 tensor, etc.)
  - Theoretical peaks from SM count × clock × ops/clock/SM per precision
- [ ] Theoretical peak calculation:
  - GPU: `SMs × clock_GHz × ops_per_clock_per_SM(precision)`
  - CPU: `cores × clock_GHz × SIMD_width(precision) × 2(FMA)`

#### 1.4 Measurement Harness
- [ ] `timer_cuda.cu`:
  - `cudaEventCreate`, `cudaEventRecord`, `cudaEventSynchronize`,
    `cudaEventElapsedTime`
  - Wraps kernel launch in event pairs
- [ ] `timer_cpu.cpp`:
  - `std::chrono::steady_clock` with `omp_get_wtime()` fallback
- [ ] Measurement loop:
  ```
  warmup(N_warmup)
  times = []
  for trial in range(N_iterations):
      start_timer()
      run_kernel()
      stop_timer()
      times.append(elapsed)
  report median(times), min(times), max(times)
  ```
- [ ] `validator.cpp`:
  - Writes kernel output to a device-global / volatile to prevent DCE
  - Optional: checksums accumulated results against known-good values
- [ ] `power_monitor.cpp` (best-effort, user-level):
  - NVIDIA: `nvmlDeviceGetPowerUsage()` via NVML (user-level API)
  - AMD: `rsmi_dev_power_avg_get()` via ROCm SMI lib
  - CPU: read `/sys/class/powercap/intel-rapl/` (Linux, if readable)
  - If unavailable, report `power_watts: null` — no failure

#### 1.5 First Kernel: `scalar_fma`
- [ ] `scalar_fma_cuda.cu`:
  - Template on precision type (double, float, __half)
  - **Throughput mode**: Each thread runs 8 independent FMA chains, unrolled
    ```cpp
    template <typename T>
    __global__ void scalar_fma_throughput(T* sink, T a, T b, int64_t iters) {
        T r0=1, r1=2, r2=3, r3=4, r4=5, r5=6, r6=7, r7=8;
        for (int64_t i = 0; i < iters; i++) {
            r0 = fma(a, r0, b); r1 = fma(a, r1, b);
            r2 = fma(a, r2, b); r3 = fma(a, r3, b);
            r4 = fma(a, r4, b); r5 = fma(a, r5, b);
            r6 = fma(a, r6, b); r7 = fma(a, r7, b);
        }
        sink[threadIdx.x + blockIdx.x * blockDim.x] = r0+r1+r2+r3+r4+r5+r6+r7;
    }
    ```
  - **Latency mode**: Single dependent chain per thread
  - Launch config: enough blocks × threads to fill all SMs
  - FLOP count: `threads × chains × iters × 2`
- [ ] `scalar_fma_cpu.cpp`:
  - OpenMP parallel, each thread runs independent FMA chains
  - Use compiler intrinsics or `std::fma` to ensure HW FMA
  - FLOP count: `threads × chains × iters × 2`
- [ ] Wire into registry, produce first JSON report

**Phase 1 Deliverable**: `./floptic --device=cuda:0 --kernels=scalar --precision=fp64,fp32,fp16` produces a valid JSON report with scalar FMA throughput and latency.

---

### Phase 2: Scalar & Vector Kernels (Weeks 3-4)
**Goal**: Complete scalar suite, vector kernels, CPU backend

#### 2.1 Remaining Scalar Kernels
- [ ] `scalar_div`: Division throughput/latency
  - Same structure as FMA: independent chains (throughput) vs dependent (latency)
  - Values chosen to avoid denormals and infinities
  - FLOP count: 1 FLOP per division
- [ ] `scalar_sqrt`: Square root
  - GPU: `sqrt()` / `sqrtf()` / `hsqrt()`
  - CPU: `std::sqrt` / `_mm256_sqrt_pd`
  - FLOP count: 1 FLOP per sqrt
- [ ] `scalar_transcendental`: sin, cos, exp, log
  - GPU: measure both fast-math (`__sinf`) and IEEE (`sinf`) variants
  - Report as separate sub-kernels: `scalar_sin_fast`, `scalar_sin_ieee`
  - FLOP count: 1 FLOP per transcendental call (conventional)

#### 2.2 Vector Kernels
- [ ] `vector_fma`: Streaming `c[i] = a[i] * b[i] + c[i]`
  - Sizes: 1K, 10K, 100K, 1M, 10M, 100M elements
  - Report both GFLOP/s and GB/s (bandwidth)
  - Identify compute-bound vs memory-bound crossover
  - FLOP count: 2 × N
- [ ] `vector_dot`: Dot product `sum += a[i] * b[i]`
  - GPU: warp shuffle reduction, then block reduction
  - CPU: SIMD horizontal add
  - FLOP count: 2 × N
- [ ] `vector_axpy`: `y[i] = alpha * x[i] + y[i]`
  - Classic BLAS Level 1, well-understood roofline behavior
  - FLOP count: 2 × N

#### 2.3 CPU Backend Completion
- [ ] All scalar kernels with OpenMP + SIMD
- [ ] Detect and report SIMD ISA used:
  - Compile multiple paths: AVX2, AVX-512, SSE
  - Runtime dispatch or report which was compiled
- [ ] Thread pinning via `OMP_PROC_BIND=close` / `OMP_PLACES=cores`
  - Set via environment in harness, document in CLI help

#### 2.4 HIP Backend (if AMD GPU available)
- [ ] Port CUDA scalar/vector kernels to HIP
  - Most are trivial: `__global__` → `__global__`, `cudaEvent` → `hipEvent`
  - FP16: `__half` → `_Float16` (or `__half` via HIP headers)

**Phase 2 Deliverable**: Scalar + vector benchmarks on CUDA, HIP (if available), and CPU across FP64, FP32, FP16, BF16.

---

### Phase 3: Dense Matrix Kernels (Weeks 5-6)
**Goal**: GEMM benchmarks, tensor cores, vendor BLAS

#### 3.1 Naive GEMM
- [ ] Simple triple-loop GEMM
  - GPU: one thread per output element (intentionally naive)
  - CPU: triple loop with OpenMP on outer loop
  - Sizes: 256², 512², 1024², 2048², 4096²
  - Serves as "how far from peak without libraries" baseline
  - FLOP count: 2 × M × N × K

#### 3.2 Vendor BLAS GEMM
- [ ] `gemm_cublas.cu`:
  - `cublasDgemm` (FP64), `cublasSgemm` (FP32)
  - `cublasGemmEx` with compute types:
    - `CUBLAS_COMPUTE_16F` (FP16)
    - `CUBLAS_COMPUTE_32F_FAST_TF32` (TF32 on tensor cores)
    - `CUBLAS_COMPUTE_32F_FAST_16BF` (BF16 on tensor cores)
  - Square matrices: 1024², 2048², 4096², 8192²
- [ ] `gemm_rocblas.cpp`:
  - `rocblas_dgemm`, `rocblas_sgemm`, `rocblas_hgemm`
  - `rocblas_gemm_ex` for mixed precision
- [ ] `gemm_mkl.cpp`:
  - `cblas_dgemm`, `cblas_sgemm`
  - `cblas_gemm_bf16bf16f32` (if available)

#### 3.3 Tensor Core GEMM (NVIDIA)
- [ ] `gemm_wmma.cu` using `nvcuda::wmma`:
  - Fragment sizes: 16×16×16 (FP16), 8×8×4 (FP64)
  - Precision combinations:
    | Input | Accumulator | Min Arch |
    |-------|-------------|----------|
    | FP16  | FP16        | Volta    |
    | FP16  | FP32        | Volta    |
    | BF16  | FP32        | Ampere   |
    | TF32  | FP32        | Ampere   |
    | FP64  | FP64        | Ampere   |
    | FP8   | FP32        | Hopper   |
    | INT8  | INT32       | Volta    |
  - Skip unsupported combinations based on compute capability
  - Report utilization: compare vs `cublasGemmEx` to gauge efficiency

#### 3.4 AMD Matrix Core GEMM
- [ ] `gemm_mfma.cpp`:
  - Use `__builtin_amdgcn_mfma_*` intrinsics
  - FP16, BF16, INT8; FP8 on MI300+
  - Compare vs rocBLAS

#### 3.5 Intel AMX (CPU)
- [ ] `gemm_amx.cpp`:
  - `_tile_loadd`, `_tile_dpbf16ps`, `_tile_dpbusd`
  - BF16 and INT8 tile matmuls
  - Compare vs MKL

**Phase 3 Deliverable**: GEMM benchmarks across naive/vendor/hardware-intrinsic for all supported precisions per device.

---

### Phase 4: Sparse & Emulated Kernels (Weeks 7-8)
**Goal**: Sparse operations, FP64 emulation via lower precision

#### 4.1 Sparse Kernels
- [ ] `spmv_csr`: y = A*x in CSR format
  - Synthetic matrices: diagonal, banded (bandwidth 5,10,50), random (0.1%, 1%, 10% fill)
  - Report GFLOP/s *and* GB/s (SpMV is almost always memory-bound)
  - FLOP count: 2 × NNZ
- [ ] `spmv_structured_cuda.cu`: NVIDIA 2:4 structured sparsity
  - Uses `cusparseLtMatmul` (cuSPARSELt library)
  - Compare same matrix: dense GEMM vs 2:4 sparse
  - Only on Ampere+ (sm_80+)
- [ ] `sparse_gemm`: Sparse × Dense via cuSPARSE / rocSPARSE

#### 4.2 Ozaki Scheme GEMM
- [ ] `ozaki_gemm_cuda.cu`:
  - Split FP64 matrix into K slices of FP16/FP32
  - K depends on input range and target accuracy
  - Perform K² FP16 GEMMs on tensor cores
  - Accumulate in FP64 for final result
  - Report:
    - `raw_gflops`: total FP16 tensor core GFLOP/s
    - `effective_gflops`: equivalent FP64 GFLOP/s
    - `num_splits`: K (varies with input)
    - `accuracy`: max ULP error vs native FP64 DGEMM
  - Compare vs native cuBLAS DGEMM:
    - Crossover point: at what matrix size does Ozaki win?

#### 4.3 Accuracy Verification Framework
- [ ] Reference computation: FP128 via `__float128` / `_Quad` on CPU
- [ ] Metrics:
  - Max ULP error
  - RMS ULP error
  - Significant correct decimal digits
  - Correctly rounded results (% of elements)
- [ ] Included in report for all emulated kernels

**Phase 4 Deliverable**: Full kernel suite including emulated precision with accuracy characterization.

---

### Phase 5: Reporting, Analysis & Polish (Weeks 9-10)
**Goal**: Publication-quality output, comparison tools, documentation

#### 5.1 Report Enhancements
- [ ] Summary tables in JSON:
  - Device × Precision → peak GFLOP/s (per kernel category)
  - Precision ratio table: FP64:FP32:FP16:FP8 for each operation type
  - Emulated vs native comparison table
- [ ] Roofline data:
  - Add memory bandwidth measurement kernel (stream copy)
  - Report arithmetic intensity for each kernel
  - Enough data to plot roofline externally

#### 5.2 Analysis Scripts (Python)
- [ ] `compare_devices.py`:
  - Load multiple JSON reports
  - Side-by-side tables: same kernel across devices
  - Highlight: which device has best FP64, best ratio, best emulated
- [ ] `plot_results.py`:
  - Bar charts: GFLOP/s by precision for each kernel category
  - Heatmap: kernel × precision → GFLOP/s (normalized to peak)
  - Scaling plots: GFLOP/s vs problem size (vector length, matrix dim)
  - Emulation comparison: native FP64 vs Ozaki

#### 5.3 Documentation
- [ ] `output-format.md`: Complete JSON schema with field descriptions
- [ ] `adding-kernels.md`: Step-by-step guide to add a new kernel
  1. Implement KernelBase subclass
  2. Add REGISTER_KERNEL macro
  3. Add CMake source
- [ ] Per-architecture expected results / known-good baselines

#### 5.4 Testing
- [ ] Unit tests (no GPU required):
  - CLI parsing
  - Report generation
  - Precision traits
  - Validator logic
- [ ] Integration: manual runs on available hardware, compare vs theoretical

**Phase 5 Deliverable**: Release-ready toolkit with documentation and analysis tools.

---

## Kernel Design Principles

### 1. Prevent Dead Code Elimination
Every kernel must write its result to device-global memory:
```cpp
// GPU: write to global memory buffer
sink[threadIdx.x + blockIdx.x * blockDim.x] = result;

// CPU: write to volatile
volatile double sink;
sink = result;
```

### 2. Throughput vs Latency Modes
Each scalar/vector kernel has two modes:
- **Throughput**: Multiple independent operation chains per thread (saturate execution units)
- **Latency**: Single dependent chain per thread (measure instruction latency)

Matrix/sparse kernels run in throughput mode only (inherently parallel).

### 3. Sufficient Work
- GPU: Launch enough threads to fill all SMs at max occupancy
- CPU: Use all cores with OpenMP, fill SIMD lanes
- Matrix: Large enough to amortize launch overhead (≥1024² for GPU GEMM)
- Minimum kernel runtime: ≥1ms per trial (adjust iterations if needed)

### 4. FLOP Counting
| Operation | FLOPs |
|-----------|-------|
| FMA (a×b+c) | 2 |
| ADD, MUL | 1 |
| DIV, SQRT | 1 |
| Transcendental (sin, exp, ...) | 1 |
| GEMM (M×N×K) | 2×M×N×K |
| SpMV (NNZ non-zeros) | 2×NNZ |
| Ozaki GEMM (emulated) | raw: K² low-prec GEMMs; effective: 2×M×N×K |

### 5. Emulated Precision Reporting
For emulated kernels (e.g., Ozaki scheme), always report both:
```json
{
    "raw_gflops": 250000.0,        // FP16 tensor core ops actually executed
    "raw_precision": "FP16",
    "effective_gflops": 25000.0,   // equivalent FP64 ops achieved
    "effective_precision": "FP64",
    "num_splits": 3,               // Ozaki splitting factor K
    "accuracy": {
        "max_ulp_error": 0.5,
        "sig_digits": 15.7,
        "correctly_rounded_pct": 99.8
    }
}
```

---

## Report JSON Schema

```json
{
    "floptic_version": "0.1.0",
    "timestamp": "2026-03-09T14:00:00Z",
    "system": {
        "hostname": "polaris-node-001",
        "os": "Linux 5.15.0",
        "compiler": "nvcc 12.4 / g++ 13.2",
        "build_backends": ["cuda", "cpu"]
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
            "base_clock_mhz": 1095,
            "boost_clock_mhz": 1410,
            "features": ["tensor_cores", "fp64_tensor", "structured_sparsity"],
            "supported_precisions": ["FP64", "FP32", "TF32", "FP16", "BF16", "INT8"],
            "theoretical_peak_gflops": {
                "FP64": 9746,
                "FP32": 19492,
                "FP64_TC": 19492,
                "TF32_TC": 155936,
                "FP16_TC": 311872,
                "BF16_TC": 311872,
                "INT8_TC": 623744
            }
        }
    ],
    "config": {
        "iterations": 100,
        "warmup": 10
    },
    "benchmarks": [
        {
            "device_id": "cuda:0",
            "kernel": "scalar_fma",
            "category": "scalar",
            "precision": "FP64",
            "mode": "throughput",
            "problem_size": {
                "threads": 1048576,
                "chains_per_thread": 8,
                "iterations_per_chain": 100000
            },
            "results": {
                "gflops": 9650.0,
                "effective_gflops": 9650.0,
                "peak_percent": 99.0,
                "median_time_ms": 1.035,
                "min_time_ms": 1.030,
                "max_time_ms": 1.089,
                "total_flops": 1677721600000,
                "clock_mhz": 1410,
                "power_watts": 385.0,
                "gflops_per_watt": 25.06
            },
            "accuracy": null
        },
        {
            "device_id": "cuda:0",
            "kernel": "ozaki_gemm",
            "category": "emulated",
            "precision": "FP16_TC",
            "effective_precision": "FP64",
            "mode": "throughput",
            "problem_size": {
                "M": 4096,
                "N": 4096,
                "K": 4096,
                "num_splits": 3
            },
            "results": {
                "gflops": 250000.0,
                "raw_precision": "FP16",
                "effective_gflops": 25000.0,
                "effective_precision": "FP64",
                "peak_percent": 80.2,
                "median_time_ms": 0.55,
                "min_time_ms": 0.54,
                "max_time_ms": 0.59,
                "total_flops": 137438953472,
                "clock_mhz": 1410,
                "power_watts": 390.0,
                "gflops_per_watt": 64.1
            },
            "accuracy": {
                "max_ulp_error": 0.5,
                "rms_ulp_error": 0.1,
                "sig_digits": 15.7,
                "correctly_rounded_pct": 99.8,
                "reference": "native_FP64_DGEMM"
            }
        }
    ],
    "summary": {
        "precision_ratios": {
            "scalar": {
                "FP32_to_FP64": 2.01,
                "FP16_to_FP64": 4.02
            },
            "gemm_vendor": {
                "FP32_to_FP64": 2.01,
                "TF32_TC_to_FP64": 16.0,
                "FP16_TC_to_FP64": 32.0,
                "INT8_TC_to_FP64": 64.0
            }
        },
        "emulation_viability": {
            "ozaki_vs_native_fp64": {
                "speedup": 2.5,
                "accuracy": "near-FP64",
                "num_splits": 3
            }
        }
    }
}
```

---

## Platform-Specific Notes

### NVIDIA GPUs
| Architecture | Compute Cap | Key FP Features |
|-------------|------------|-----------------|
| Volta (V100) | 7.0 | 1:2 FP64:FP32, FP16 tensor, INT8 tensor |
| Ampere (A100) | 8.0 | 1:2 FP64:FP32, FP64 tensor, TF32, BF16, 2:4 sparsity |
| Hopper (H100) | 9.0 | 1:2 FP64:FP32, FP8 tensor, TMA |
| Blackwell (B200) | 10.0 | ~1:2 FP64:FP32, FP4/FP6, no IEEE tensor, Ozaki ref impl |
| Consumer (RTX 4090) | 8.9 | 1:64 FP64:FP32 — worst case scenario |

### AMD GPUs
| Architecture | Key FP Features |
|-------------|-----------------|
| CDNA2 (MI250X) | 1:2 FP64:FP32, MFMA FP64/FP32/FP16/BF16/INT8 |
| CDNA3 (MI300X) | 1:2 FP64:FP32, + FP8 MFMA, unified HBM |

### Intel CPUs
| Architecture | Key FP Features |
|-------------|-----------------|
| Ice Lake | AVX-512, VNNI (INT8) |
| Sapphire Rapids | AVX-512, AMX (BF16 + INT8 tiles), AVX-512 FP16 |

### Intel GPUs
| Architecture | Key FP Features |
|-------------|-----------------|
| Ponte Vecchio (Max 1550) | XMX: FP64/FP32/FP16/BF16/INT8/INT4, 1:2 FP64:FP32 |

---

## Dependencies

| Dependency | Purpose | Required? |
|-----------|---------|-----------|
| CMake ≥ 3.22 | Build system | Yes |
| C++17 compiler | Language standard | Yes |
| nlohmann/json | JSON report output | Yes (header-only, vendored) |
| CUDA Toolkit ≥ 11.0 | NVIDIA GPU backend | Optional |
| cuBLAS | Vendor GEMM (NVIDIA) | Optional (part of CUDA Toolkit) |
| cuSPARSE / cuSPARSELt | Sparse ops / structured sparsity | Optional |
| NVML | Power monitoring (NVIDIA) | Optional (part of driver) |
| ROCm ≥ 5.0 / HIP | AMD GPU backend | Optional |
| rocBLAS | Vendor GEMM (AMD) | Optional |
| rocSPARSE | Sparse ops (AMD) | Optional |
| ROCm SMI lib | Power monitoring (AMD) | Optional |
| oneAPI / SYCL | Intel GPU backend | Optional |
| Intel MKL / oneMKL | Vendor BLAS (Intel) | Optional |
| OpenMP | CPU parallelism | Optional (strongly recommended) |
| Python 3.8+ | Analysis scripts | Optional |
| matplotlib, pandas | Plotting & analysis | Optional |
| quadmath (`__float128`) | FP128 reference for Ozaki accuracy | Optional (GCC only) |
