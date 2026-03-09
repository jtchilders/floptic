# Floptic 🔬

**A lens on your FLOP throughput.**

Floptic is a portable FLOP measurement toolkit that characterizes floating-point performance across precisions, operation types, and hardware-specific features on modern HPC architectures.

## Motivation

Modern GPUs are shifting compute investment toward reduced-precision arithmetic (FP16, BF16, FP8, FP4) while FP64 throughput stagnates or declines relative to transistor counts. Scientific computing and Modeling & Simulation (ModSim) applications that depend on FP64 need to understand:

- What is the **actual** FLOP throughput for each precision on a given device?
- What are the **ratios** between precisions (FP64:FP32:FP16:FP8)?
- How do **scalar**, **vector**, **matrix**, and **sparse** operations differ?
- What do **hardware-specific units** (tensor cores, AMX, matrix engines) deliver?
- Is **FP64 emulation** via lower-precision arithmetic viable and at what cost?

Floptic answers these questions with a suite of micro-benchmarks that produce structured, comparable reports across architectures.

## Supported Platforms

| Platform | Backend | Status |
|----------|---------|--------|
| NVIDIA GPUs | CUDA | Planned |
| AMD GPUs | HIP | Planned |
| Intel GPUs | SYCL | Planned |
| Intel CPUs | OpenMP / Intrinsics | Planned |
| AMD CPUs | OpenMP / Intrinsics | Planned |

## Quick Start

```bash
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=80   # optional: target specific GPU arch
make -j
./floptic --device=cuda:0 --precision=all --report=json
```

## Output

Floptic generates structured JSON reports. See [docs/output-format.md](docs/output-format.md) for the full schema.

## License

TBD

## Authors

- Taylor Childers (ANL)
- Reet Barik (ANL)
