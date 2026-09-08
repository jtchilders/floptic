# Floptic Scientific Correctness and Engineering Roadmap

This roadmap tracks the work required to make Floptic results scientifically defensible, reproducible, and suitable as inputs to computer-architecture performance models.

## Measurement contract

A published Floptic result must establish all of the following:

1. **The requested operation executed successfully.** API, allocation, launch, and synchronization failures invalidate the trial.
2. **The operation computed the intended result.** Nonzero deterministic operands and a reference calculation must be used; timing alone is not validation.
3. **The operation count and metric unit are explicit.** FLOP/s, integer OP/s, and byte/s must not share an ambiguously named field.
4. **The denominator is defensible.** Peak percentages must identify the exact peak model, source, clock basis, and arithmetic convention.
5. **The run is reproducible.** Reports must include software/build provenance, selected problem and algorithm, effective execution configuration, input distribution, and validation result.
6. **Failures and unsupported cases remain visible.** They must not be silently omitted or represented as zero performance.

## P0 — Measurement validity

- [ ] Replace zero-filled GEMM operands with deterministic, configurable nonzero data.
  - Use a fixed default seed recorded in the report.
  - Provide bounded signed uniform data for throughput runs.
  - Add Gaussian, wide-exponent/log-uniform, sparse, cancellation-heavy, and special-value validation cases.
  - Keep initialization and validation outside timed regions.
- [ ] Validate every GEMM family against a reference result before accepting timings.
  - FP64/FP32: compare against a higher-accuracy or vendor-reference computation as appropriate.
  - TF32/FP16/BF16/FP8/FP4: report absolute, relative, and normwise error with precision-specific tolerances.
  - INT8/INT4: require exact integer results when accumulation cannot overflow; explicitly test overflow bounds.
  - Emulated FP32/FP64: report accuracy separately from throughput and exercise representative exponent distributions.
- [ ] Make Ozaki/ADP measurements data-aware.
  - Record input distribution and exponent span.
  - Record selected mantissa-bit count and native fallback when the API exposes them.
  - Do not generalize performance from all-zero or single-pattern operands.
- [ ] Make GPU/runtime/library errors invalidate a trial.
  - Replace print-only CUDA, HIP, cuBLAS, cuBLASLt, rocBLAS, and hipBLASLt checks with propagated status.
  - Check every allocation, descriptor, event, launch, synchronization, and library call.
  - [x] Return a nonzero process status when a requested benchmark has no valid result.
- [x] Distinguish benchmark states in reports: `ok`, `unsupported`, `failed`, `not_requested`, and `validation_failed`.
- [ ] Fix CPU work accounting when OpenMP is unavailable.
  - Effective thread count must be one for serial builds.
  - Reject or clearly downgrade requests for multiple CPU threads without a parallel runtime.
  - Record requested and effective thread counts.

## P1 — Reproducibility and metric semantics

- [x] Replace the overloaded `gflops` field with a typed metric representation.
  - Kinds: floating-point operations, integer operations, and transferred bytes.
  - Units: base SI rate plus display formatting; no bandwidth stored as “GFLOP/s.”
  - Preserve a documented compatibility field for one schema transition if needed.
- [x] Add a versioned JSON schema and validate generated reports in tests.
- [ ] Record complete run provenance:
  - Git commit and dirty state.
  - Compiler identity, version, and effective flags.
  - CUDA/HIP runtime and driver; cuBLAS/cuBLASLt/rocBLAS/hipBLASLt versions.
  - Device UUID where available.
  - Requested and effective clocks; clock-lock state when discoverable.
  - Requested and effective thread/block configuration.
  - GEMM dimensions, selected algorithm ID, workspace, math/compute mode, accumulation and output types.
  - Trial count, warmup count, seed, input distribution, and validation policy.
- [ ] Report robust statistics beyond min/median/max.
  - Mean, standard deviation, selected quantiles, and number of accepted/rejected trials.
  - Define an outlier policy before using one; retain raw samples or an optional sample artifact.
- [ ] Separate tuning from measurement.
  - Use an explicit tuning phase.
  - Recreate and verify the selected configuration for the measurement phase.
  - Do not call best-of-two a median.
  - Record the search space and winning configuration.
- [ ] Define cache-state policy for vector and memory tests and record it.
- [ ] Measure long enough to exceed timer-resolution and launch-overhead floors; calibrate work automatically when necessary.

## P1 — CLI and control correctness

- [x] Implement `--kernel=<NAME>` filtering and fail on unknown kernel names.
- [x] Pass `--warmup=<N>` into every kernel; remove hard-coded warmup counts.
- [x] Validate all numeric options before device discovery.
  - Require positive trials and inner iterations.
  - Require nonnegative warmup.
  - Validate CPU threads, GPU blocks, threads per block, and blocks per SM against sensible ranges.
  - Catch conversion/range errors and return a concise error with nonzero status.
- [x] Reject unknown precision, device, category, and report-format values; never map typos to FP64.
- [x] Deduplicate repeated device/category/precision selections.
- [x] Make an empty benchmark selection an error.
- [x] Implement or remove `--report`; make `--info` produce useful output without requiring `--output`.
- [ ] Ensure report-write failures propagate to the process exit code.

## P1 — CPU methodology

- [ ] Decouple runtime ISA detection from compile-time `-march=native`.
  - Build scalar, AVX2/FMA, and AVX-512 variants in separately targeted functions or translation units.
  - Dispatch using CPUID plus OSXSAVE/XGETBV checks.
  - Keep distributable binaries safe on machines older than the build host.
- [x] Handle SIMD tails correctly and count only executed operations.
- [ ] Validate allocations and use RAII containers/aligned allocators.
- [ ] Pin worker threads or report affinity and placement; characterize NUMA placement for multisocket systems.
- [ ] First-touch vector allocations in the same placement policy used for measurement.
- [ ] Replace instantaneous `/proc/cpuinfo` frequency with a documented clock basis.
  - Prefer measured effective frequency during the timed interval when available.
  - Otherwise use nominal/max frequency with an explicit source and uncertainty.
  - Do not label a sampled current frequency as boost frequency.
- [ ] Model CPU peaks by microarchitecture rather than assuming two full-width FMA units on every x86 CPU.
- [ ] Implement and validate AArch64 NEON/SVE paths or describe scalar-only operation accurately.

## P1 — Device peak models

- [ ] Replace permissive architecture defaults with explicit supported records; unknown architectures must yield unknown peaks.
- [ ] Add source citations and arithmetic conventions to every architecture record.
- [ ] Unit-test reference devices against published dense, nonsparse specifications.
- [ ] Correct capability distinctions, including Volta versus Turing INT8 tensor support and CDNA1 FP64 matrix capability.
- [ ] Separate vector/SIMT peaks from matrix/tensor peaks and select denominators by the operation actually executed.
- [ ] Separate FLOP/s from integer OP/s; document whether multiply-accumulate counts as one operation or two.
- [ ] Add product/configuration awareness where one compute capability has materially different unit counts or rates.
- [ ] Use measured or explicitly sourced clocks and expose uncertainty rather than reporting false precision.
- [ ] Add CUDA HBM peak bandwidth and use bandwidth units in theoretical-peak reports.
- [ ] Verify HIP memory-clock interpretation against effective data rate and published bandwidth; avoid generation-wide magic multipliers when runtime properties already encode the effective rate.

## P1 — GPU implementation correctness

- [ ] Include workspace and all auxiliary allocations in memory preflight checks; query live free memory.
- [ ] Fix the NVFP4 global output-scale allocation leak by retaining it in RAII state and freeing it.
- [ ] Remove or constrain dead generic rocBLAS INT8 allocation code that would undersize INT32 output.
- [ ] Replace process-global mutable cuBLASLt benchmark state with device-scoped RAII state.
- [ ] Validate cached FP8/INT8 state against device, datatype, dimensions, and algorithm before reuse.
- [ ] Preserve the selected hipBLASLt algorithm during full measurement instead of re-querying a one-result heuristic.
- [ ] Store complete timing statistics and FLOP/OP counts for rocBLAS and hipBLASLt results.
- [ ] Add rectangular and transposed GEMM cases to verify layout and leading-dimension logic.
- [ ] Verify that “no tensor core” modes actually use the claimed instruction family using profiler evidence or label them as requested math modes rather than guaranteed hardware paths.

## P2 — Bandwidth, power, and environment

- [ ] Add typed bandwidth results for AXPY, STREAM triad, and copy.
- [ ] Define traffic accounting explicitly, including write allocation/read-for-ownership where relevant on CPUs.
- [ ] Add CUDA and HIP theoretical HBM bandwidth utilization.
- [ ] Implement power/energy measurement or remove placeholder claims.
  - CPU: use two RAPL energy readings with wrap handling and interval timing.
  - NVIDIA: sample NVML energy/power with synchronization and disclose sampling limitations.
  - AMD: use supported ROCm SMI interfaces where available.
- [ ] Report energy per operation and sustained power only when the measurement interval is adequate.
- [ ] Record thermal, power-cap, clock-throttling, and competing-process indicators when available.

## P2 — Test and CI infrastructure

- [ ] Enable CTest and add CPU-only unit/integration tests.
- [ ] Add tests for CLI parsing, rejection paths, selection/deduplication, and exit status.
- [ ] Add peak-model fixtures for representative NVIDIA, AMD, and CPU architectures.
- [x] Add operation-count and typed-unit tests.
- [x] Add JSON schema and Markdown unit tests.
- [x] Add deterministic operand-generation and validation-policy tests.
- [x] Add CPU scalar/SIMD numerical tests and serial/OpenMP accounting tests.
- [ ] Register hardware-independent compile tests for CUDA/HIP code where toolkits are available.
- [ ] Add hardware smoke tests on representative NVIDIA and AMD systems when runners are available.
- [ ] Add GitHub Actions for CPU-only GCC and Clang builds, tests, and warning checks.
- [ ] Treat warnings in project code as errors in CI while excluding fetched dependencies.

## P2 — Documentation and release hygiene

- [ ] Reconcile README support tables with implemented and empirically tested backends.
- [ ] Correct `--iterations` versus `--inner-iters`, `--report`, `--kernel`, `--warmup`, and `--info` documentation.
- [ ] Document precise accumulation/output semantics for every reduced-precision kernel.
- [ ] State whether each result is native arithmetic, emulated effective throughput, integer OP/s, or memory bandwidth.
- [ ] Document validation distributions and tolerances.
- [ ] Add a reproducibility checklist for committed result files.
- [ ] Regenerate committed results only after the new schema and validation contract are implemented.
- [ ] Choose and add a project license.

## P3 — Future scope

- [ ] Add multi-GPU scaling benchmarks only after single-device validity is established.
- [ ] Add AMX and SVE implementations only with matching validation and peak models.
- [ ] Add Intel GPU/SYCL support only with tested hardware and explicit architecture records.
- [ ] Consider raw-sample sidecar files and uncertainty propagation for downstream future-system models.

## Acceptance criteria for trustworthy published results

A result may be called trustworthy only when:

- The requested kernel completed with no ignored API errors.
- Validation passed on nonzero operands under a documented policy.
- Metric kind, unit, operation count, and timed region are unambiguous.
- Peak percentage references a cited architecture record and documented clock basis.
- Effective execution configuration and software provenance are recorded.
- Tests cover the associated accounting and report schema.
- Unsupported or failed cases are explicit rather than absent.
- The exact report can be regenerated from a recorded command, seed, commit, and environment.