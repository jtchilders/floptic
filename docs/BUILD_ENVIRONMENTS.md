# Build Environments

Tested build configurations for each platform. All builds use `--key=value` CLI syntax and target a single GPU architecture.

---

## JLSE (Joint Laboratory for System Evaluation)

### B200 — Blackwell (sm_100)

**Node**: `blackwell00`
**GPU**: 8× NVIDIA B200 (HGX 1000W), 148 SMs, 1965 MHz, 180 GB HBM3e
**CPU**: Intel Xeon 6960P (144 physical / 288 logical cores, AVX-512)
**OS**: Linux x86_64

```bash
module load cmake gcc/13.3.0 cuda/13.1

cmake -B build_gpu_b200 -DCMAKE_CUDA_ARCHITECTURES=100
make -C build_gpu_b200 -j
```

**Notes**:
- CUDA 13.1 supports BF16×9 emulated FP32 and Ozaki emulated FP64
- Block-scaled MXFP8 and NVFP4 require sm_100
- Per-tensor FP8 does NOT work on Blackwell (requires block scaling)
- No FP64 tensor cores on Blackwell

### GH200 — Hopper (sm_90)

**Node**: `grace01`
**GPU**: NVIDIA GH200 480GB, 132 SMs, 1980 MHz, 96 GB HBM3
**CPU**: NVIDIA Grace (ARM Neoverse V2), 72 cores, aarch64
**OS**: Linux aarch64

```bash
source /vast/projects/datascience/parton/cuda/cuda_13.2.0_595.45.04_linux_sbsa--setup.sh
module load cmake gcc/11.2.0

cmake -B build_gpu_gh200 -DCMAKE_CUDA_ARCHITECTURES=90
make -C build_gpu_gh200 -j
```

**Notes**:
- Uses custom CUDA 13.2 install (SBSA/aarch64 build), not the system module
- Driver 590.48.01 supports CUDA 13.1; forward-compatible with 13.2 toolkit
- FP8 per-tensor does NOT work on aarch64 (all algos fail AlgoCheck)
- INT8 tensor core algos (71, 70) not available on aarch64; only slow algos (64, 20)
- BF16×9 emulated FP32 compiles but may not engage emulation on Hopper
- Ozaki emulated FP64 compiles but is gated to sm_100+ at runtime
- CPU detection reports 0 physical cores / 0 MHz (aarch64 `/proc/cpuinfo` format differs)

### H100 — Hopper (sm_90)

**Node**: JLSE H100 nodes
**GPU**: NVIDIA H100 SXM, 132 SMs, 1980 MHz
**CPU**: x86_64
**OS**: Linux x86_64

```bash
module load cmake gcc/12.2.0 cuda/13.1

cmake -B build_gpu_h100 -DCMAKE_CUDA_ARCHITECTURES=90 \
    -DCMAKE_CXX_COMPILER=g++ -DCMAKE_CUDA_HOST_COMPILER=g++
make -C build_gpu_h100 -j
```

**Notes**:
- Has FP64 tensor cores (unlike Blackwell)
- CUDA 13.1 enables BF16×9 emulated FP32 (but cuBLAS doesn't engage emulation on Hopper — returns native FP32 rate)
- FP8 per-tensor still fails (0.000 GF/s) even on x86_64 with CUDA 13.1
- INT8 TC only finds slow algorithms (103–133 TF/s, ~5% of peak)

### A100 — Ampere (sm_80)

**Node**: JLSE A100 nodes
**GPU**: NVIDIA A100 PCIe, 108 SMs
**CPU**: x86_64
**OS**: Linux x86_64

```bash
module load cmake/3.26.5 gcc/11.1.0 cuda/11.6.2

cmake -B build_gpu_a100 -DCMAKE_CUDA_ARCHITECTURES=80 \
    -DCMAKE_CXX_COMPILER=g++ -DCMAKE_CUDA_HOST_COMPILER=g++
make -C build_gpu_a100 -j
```

**Notes**:
- No FP8, no FP4, no emulated precision
- Has FP64 tensor cores
- A100 PCIe 40GB (108 SMs, 1410 MHz) — lower clocks than SXM variant
- Clean 1:1:7.5:15.5 ratio for FP64:FP32:TF32:FP16
- Scalar FMA achieves 97–99% peak across all precisions (cleanest architecture)
- CPU: AMD EPYC 7532 (32 cores, AVX2)

---

## JLSE AMD GPU Nodes

### MI100 — CDNA1 (gfx908)

**Node**: `gpu_amd_mi100`
**GPU**: AMD Instinct MI100, 120 CUs, 1502 MHz, 32 GB HBM2
**Architecture**: CDNA1 (gfx908)

```bash
# Build: rocm/6.3.2 + gcc/12.2.0 (rocm/6.4.3 has lld crash with gcc/13.3.0)
module load rocm/6.3.2 cmake/3.28.3 gcc/12.2.0

cmake -B build_gpu_mi100 \
    -DCMAKE_CXX_COMPILER=hipcc \
    -DFLOPTIC_ENABLE_CUDA=OFF \
    -DFLOPTIC_ENABLE_HIP=ON \
    -DGPU_TARGETS=gfx908
make -C build_gpu_mi100 -j

# Run: swap to gcc/13.3.0 (ROCm libs need GLIBCXX_3.4.32)
module swap gcc/12.2.0 gcc/13.3.0
./build_gpu_mi100/floptic --device=hip:0 --precision=all
```

**Notes**:
- Node: `amdgpu02`/`amdgpu03`, 4× MI100 (gfx908)
- ROCm 6.3.2, HIP 6.3, AMD clang 18.0
- Build with gcc/12.2.0, run with gcc/13.3.0 (ABI mismatch workaround)
- CDNA1: FP64 = half-rate FP32 (32 FMA/CU/clk vs 64)
- Matrix cores support FP64, FP32, FP16, BF16, INT8
- No TF32, no FP8

### MI250X — CDNA2 (gfx90a)

**Node**: `gpu_amd_mi250`
**GPU**: AMD Instinct MI250X, 2 GCDs (each reported as separate HIP device)
**Per GCD**: 110 CUs, 1700 MHz, 64 GB HBM2e
**Architecture**: CDNA2 (gfx90a)

```bash
# TODO: confirm modules
module load rocm cmake gcc

cmake -B build_gpu_mi250 \
    -DCMAKE_CXX_COMPILER=hipcc \
    -DFLOPTIC_ENABLE_CUDA=OFF \
    -DFLOPTIC_ENABLE_HIP=ON \
    -DGPU_TARGETS=gfx90a
make -C build_gpu_mi250 -j
```

**Notes**:
- 2 GCDs per card — each is a separate HIP device (hip:0, hip:1)
- Full-rate FP64 (64 FMA/CU/clk)
- Matrix cores: FP64, FP32, FP16, BF16, INT8
- No TF32, no FP8

### MI300X — CDNA3 (gfx942)

**Node**: `gpu_amd_mi300x`
**GPU**: AMD Instinct MI300X, 304 CUs (8 XCDs × 38 CUs), 2100 MHz, 192 GB HBM3
**Architecture**: CDNA3 (gfx942)

```bash
# TODO: confirm modules
module load rocm cmake gcc

cmake -B build_gpu_mi300x \
    -DCMAKE_CXX_COMPILER=hipcc \
    -DFLOPTIC_ENABLE_CUDA=OFF \
    -DFLOPTIC_ENABLE_HIP=ON \
    -DGPU_TARGETS=gfx942
make -C build_gpu_mi300x -j
```

**Notes**:
- CDNA3: full-rate FP64, TF32 support, FP8 support
- Matrix cores: FP64, FP32, TF32, FP16, BF16, INT8, FP8
- Specs: 163.4 TF/s FP64 matrix, 1307.4 TF/s FP16 matrix, 2614.9 TF/s FP8 matrix
- 5.3 TB/s HBM3 bandwidth

### MI300A — CDNA3 APU (gfx942)

**Node**: `gpu_amd_mi300a`
**GPU**: AMD Instinct MI300A (APU), 228 CUs, 2100 MHz, 128 GB unified HBM3
**Architecture**: CDNA3 (gfx942)

```bash
# TODO: confirm modules
module load rocm cmake gcc

cmake -B build_gpu_mi300a \
    -DCMAKE_CXX_COMPILER=hipcc \
    -DFLOPTIC_ENABLE_CUDA=OFF \
    -DFLOPTIC_ENABLE_HIP=ON \
    -DGPU_TARGETS=gfx942
make -C build_gpu_mi300a -j
```

**Notes**:
- APU: shared CPU + GPU die with unified memory
- Same CDNA3 ISA as MI300X but fewer CUs (228 vs 304)
- Specs: 122.6 TF/s FP64 matrix, 980.6 TF/s FP16 matrix

---

## Polaris (ALCF)

**GPU**: NVIDIA A100 PCIe
**OS**: Linux x86_64

```bash
module load cmake cuda/11.8.0 gcc/12.2.0

cmake -B build -DCMAKE_CUDA_ARCHITECTURES=80
make -C build -j
```

---

## Feature Availability by CUDA Version

| Feature | Minimum CUDA | Guard |
|---------|-------------|-------|
| FP64, FP32, FP16, BF16 GEMM | 11.0 | Always |
| TF32 GEMM | 11.0 | `sm_80+` |
| INT8 GEMM (cublasLt) | 11.0 | `sm_70+` |
| FP8 per-tensor (cublasLt) | 11.8 | `sm_89–99` runtime check |
| MXFP8 block-scaled | 12.8 | `sm_100+` runtime + `#if CUDA >= 12.8` |
| NVFP4 block-scaled | 12.8 | `sm_100+` runtime + `#if CUDA >= 12.8` |
| BF16×9 emulated FP32 | 13.1 | `sm_90+` runtime + `#if CUDA >= 13.1` |
| Ozaki emulated FP64 | 13.1 | `sm_100+` runtime + `#if CUDA >= 13.1` |

## Feature Availability by Architecture

| Feature | sm_80 (A100) | sm_90 (H100/GH200) | sm_100 (B200) |
|---------|:---:|:---:|:---:|
| FP64 CUDA cores | ✅ | ✅ | ✅ |
| FP64 tensor cores | ✅ | ✅ | ❌ |
| TF32 tensor cores | ✅ | ✅ | ✅ |
| FP16/BF16 tensor cores | ✅ | ✅ | ✅ |
| INT8 tensor cores | ✅ | ✅ | ✅ |
| FP8 per-tensor | ❌ | ✅* | ❌ |
| MXFP8 block-scaled | ❌ | ❌ | ✅ |
| NVFP4 block-scaled | ❌ | ❌ | ✅ |
| BF16×9 emulated FP32 | ❌ | ✅ | ✅ |
| Ozaki emulated FP64 | ❌ | ❌ | ✅ |

*FP8 per-tensor works on H100 x86_64 only; fails on GH200 aarch64.
