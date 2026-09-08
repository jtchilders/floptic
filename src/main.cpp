#include "floptic/cli_parser.hpp"
#include "floptic/device_info.hpp"
#include "floptic/kernel_registry.hpp"
#include "floptic/report.hpp"
#include "floptic/benchmark_status.hpp"
#include "floptic/typed_metric.hpp"
#include "floptic/dispatch_normalize.hpp"
#include <iostream>
#include <algorithm>
#include <map>
#include <set>
#include <cstdio>

// Force-link kernel translation units from static libraries.
// Without these references, the linker may discard the .o files
// and the REGISTER_KERNEL static initializers never run.
namespace floptic {
    namespace force_link {
        extern void scalar_fma_cpu_link();
        extern void vector_axpy_cpu_link();
#ifdef FLOPTIC_HAS_CUDA
        extern void scalar_fma_cuda_link();
        extern void vector_axpy_cuda_link();
        extern void stream_triad_cuda_link();
        extern void stream_copy_cuda_link();
        extern void gemm_cublas_link();
        extern void gemm_cublas_notc_link();
        extern void gemm_cublas_int8_link();
        extern void gemm_cublas_fp8_link();
        extern void gemm_cublas_mxfp8_link();
        extern void gemm_cublas_nvfp4_link();
        extern void gemm_cublas_emu_fp32_link();
        extern void gemm_cublas_emu_fp64_link();
#endif
#ifdef FLOPTIC_HAS_HIP
        extern void scalar_fma_hip_link();
        extern void vector_axpy_hip_link();
        extern void stream_triad_hip_link();
        extern void stream_copy_hip_link();
        extern void gemm_rocblas_link();
        extern void gemm_hipblaslt_link();
#endif
    }
    static void force_link_all() {
        force_link::scalar_fma_cpu_link();
        force_link::vector_axpy_cpu_link();
#ifdef FLOPTIC_HAS_CUDA
        force_link::scalar_fma_cuda_link();
        force_link::vector_axpy_cuda_link();
        force_link::stream_triad_cuda_link();
        force_link::stream_copy_cuda_link();
        force_link::gemm_cublas_link();
        force_link::gemm_cublas_notc_link();
        force_link::gemm_cublas_int8_link();
        force_link::gemm_cublas_fp8_link();
        force_link::gemm_cublas_mxfp8_link();
        force_link::gemm_cublas_nvfp4_link();
        force_link::gemm_cublas_emu_fp32_link();
        force_link::gemm_cublas_emu_fp64_link();
#endif
#ifdef FLOPTIC_HAS_HIP
        force_link::scalar_fma_hip_link();
        force_link::vector_axpy_hip_link();
        force_link::stream_triad_hip_link();
        force_link::stream_copy_hip_link();
        force_link::gemm_rocblas_link();
        force_link::gemm_hipblaslt_link();
#endif
    }
}

namespace floptic {

// Defined in device backends
std::vector<DeviceInfo> discover_devices() {
    std::vector<DeviceInfo> all;

    auto cpus = discover_cpu_devices();
    all.insert(all.end(), cpus.begin(), cpus.end());

#ifdef FLOPTIC_HAS_CUDA
    auto cuda_gpus = discover_cuda_devices();
    all.insert(all.end(), cuda_gpus.begin(), cuda_gpus.end());
#endif

#ifdef FLOPTIC_HAS_HIP
    auto hip_gpus = discover_hip_devices();
    all.insert(all.end(), hip_gpus.begin(), hip_gpus.end());
#endif

    return all;
}

} // namespace floptic

int main(int argc, char* argv[]) {
    using namespace floptic;

    // Ensure kernel registration from static libs
    force_link_all();

    std::cerr << "Floptic v0.1.0 — A lens on your FLOP throughput\n" << std::endl;

    auto opts = parse_args(argc, argv);

    if (!opts.valid) {
        std::cerr << opts.error << std::endl;
        return 1;
    }

    if (opts.help) {
        print_usage(argv[0]);
        return 0;
    }

    // Discover devices
    std::cerr << "Discovering devices..." << std::endl;
    auto all_devices = discover_devices();

    if (all_devices.empty()) {
        std::cerr << "No devices found!" << std::endl;
        return 1;
    }

    for (auto& dev : all_devices) {
        std::cerr << "  Found: " << dev.id << " — " << dev.name
                  << " (" << dev.compute_units << " units, "
                  << dev.boost_clock_mhz << " MHz)" << std::endl;
    }
    std::cerr << std::endl;

    // --info: print device info and exit
    if (opts.show_info) {
        Report info_report;
        info_report.devices = all_devices;
        info_report.build_backends = {"cpu"};
#ifdef FLOPTIC_HAS_CUDA
        info_report.build_backends.push_back("cuda");
#endif
#ifdef FLOPTIC_HAS_HIP
        info_report.build_backends.push_back("hip");
#endif
        if (opts.output_path.empty()) {
            // No output path: emit valid JSON to stdout (errors/progress stay on stderr).
            std::cout << report_to_json(info_report).dump(2) << std::endl;
        } else {
            write_json_report(info_report, opts.output_path);
        }
        return 0;
    }

    // Register kernels (happens at static init via REGISTER_KERNEL macros)
    std::cerr << "Available kernels:" << std::endl;
    auto& registry = KernelRegistry::instance();

    // --list: list kernels and exit
    if (opts.list_kernels) {
        for (auto& name : registry.list_kernel_names()) {
            std::cout << "  " << name << std::endl;
        }
        return 0;
    }

    // Filter devices
    std::vector<DeviceInfo> target_devices;
    for (auto& dev : all_devices) {
        bool selected = false;
        for (auto& d : opts.devices) {
            if (d == "all" || d == dev.id || d == dev.type) {
                selected = true;
                break;
            }
            // Allow --device=cuda or --device=hip to match by backend prefix
            if ((d == "cuda" && dev.id.substr(0, 4) == "cuda") ||
                (d == "hip"  && dev.id.substr(0, 3) == "hip")) {
                selected = true;
                break;
            }
        }
        if (selected) target_devices.push_back(dev);
    }

    if (target_devices.empty()) {
        std::cerr << "No matching devices for: ";
        for (auto& d : opts.devices) std::cerr << d << " ";
        std::cerr << std::endl;
        return 1;
    }

    // --kernel=<NAME>: must match a registered kernel for at least one of the
    // selected devices' backends, or we fail fast with a diagnostic rather
    // than silently reaching the generic "no benchmarks executed" error
    // (e.g. a CUDA-only kernel name requested with --device=cpu).
    if (!opts.kernel_name.empty()) {
        std::set<std::string> target_backends;
        for (auto& dev : target_devices) {
            std::string b = "cpu";
            if (dev.id.substr(0, 4) == "cuda") b = "cuda";
            else if (dev.id.substr(0, 3) == "hip") b = "hip";
            target_backends.insert(b);
        }

        bool found_any_backend = false;
        bool found_for_selected_backend = false;
        std::set<std::string> backends_with_this_kernel;
        for (auto& k : registry.list_kernel_names()) {
            // list_kernel_names() formats as "name [backend/category]"
            auto bracket = k.find(" [");
            std::string kname = (bracket != std::string::npos) ? k.substr(0, bracket) : k;
            if (kname != opts.kernel_name) continue;
            found_any_backend = true;
            if (bracket != std::string::npos) {
                auto slash = k.find('/', bracket);
                if (slash != std::string::npos) {
                    std::string backend = k.substr(bracket + 2, slash - (bracket + 2));
                    backends_with_this_kernel.insert(backend);
                    if (target_backends.count(backend)) found_for_selected_backend = true;
                }
            }
        }

        if (!found_any_backend) {
            std::cerr << "Unknown kernel: " << opts.kernel_name
                      << " (use --list to see available kernels)" << std::endl;
            return 1;
        }
        if (!found_for_selected_backend) {
            std::cerr << "Kernel '" << opts.kernel_name << "' is registered for backend(s) [";
            bool first = true;
            for (auto& b : backends_with_this_kernel) {
                if (!first) std::cerr << ", ";
                std::cerr << b;
                first = false;
            }
            std::cerr << "] but not for the selected device backend(s) (use --list to see "
                      << "available kernels, or --device to target a matching backend)"
                      << std::endl;
            return 1;
        }
    }

    // Build report
    Report report;
    report.devices = target_devices;
    report.iterations = opts.trials;
    report.warmup = opts.warmup;
    report.build_backends = {"cpu"};
#ifdef FLOPTIC_HAS_CUDA
    report.build_backends.push_back("cuda");
#endif
    report.system = collect_system_info();
#ifdef FLOPTIC_HAS_CUDA
    fill_cuda_system_info(report.system);
#endif

    std::cerr << "\nRunning benchmarks..." << std::endl;

    bool executed_any = false;

    for (auto& device : target_devices) {
        std::cerr << "\n=== Device: " << device.id << " (" << device.name << ") ===" << std::endl;

        // Determine backend for this device based on id prefix
        std::string dev_backend = "cpu";
        if (device.id.substr(0, 4) == "cuda") {
            dev_backend = "cuda";
        } else if (device.id.substr(0, 3) == "hip") {
            dev_backend = "hip";
        }

        // Get matching kernels
        for (auto& cat : opts.kernel_categories) {
            auto kernels = registry.get_kernels(cat, dev_backend);

            for (auto* kernel : kernels) {
                // --kernel=<NAME> restricts execution to the exact match
                if (!opts.kernel_name.empty() && kernel->name() != opts.kernel_name)
                    continue;

                // Skip kernels that aren't available on this device
                if (!kernel->is_available(device))
                    continue;

                for (auto& precision : opts.precisions) {
                    if (!kernel->supports_precision(precision))
                        continue;
                    if (!device.supports_precision(precision))
                        continue;

                    for (auto& mode : kernel->supported_modes()) {
                        KernelConfig config;
                        config.precision = precision;
                        config.mode = mode;
                        config.iterations = opts.inner_iters;
                        config.device_id = device.id;
                        config.threads = opts.cpu_threads;
                        config.warmup_trials = opts.warmup;
                        config.gpu_blocks = opts.gpu_blocks;
                        config.gpu_threads_per_block = opts.gpu_threads_per_block;
                        config.gpu_blocks_per_sm = opts.gpu_blocks_per_sm;

                        std::cerr << "\n--- " << kernel->name() << " | "
                                  << precision_to_string(precision) << " | "
                                  << mode << " ---" << std::endl;

                        auto result = kernel->run(config, device, opts.trials);

                        // Normalize legacy gflops/effective_gflops/
                        // total_flops/total_bytes fields into a typed
                        // metric and an authoritative status at the
                        // dispatch boundary, using the centralized,
                        // independently-tested mapping in
                        // dispatch_normalize.hpp. Status is never inferred
                        // from a nonpositive rate here — KernelResult::
                        // status defaults to FAILED and normalize_dispatch_
                        // result only ever promotes that default to OK on
                        // evidence of a valid measurement; kernels that
                        // already set an explicit status (OK or otherwise)
                        // are left untouched.
                        normalize_dispatch_result(result, kernel->category(), precision);

                        std::cerr << "  Result: " << format_metric_rate(result.metric)
                                  << " (status " << benchmark_status_to_string(result.status)
                                  << ", median " << result.median_time_ms << " ms"
                                  << ", peak " << result.peak_percent << "%)"
                                  << std::endl;

                        // Statuses other than ok are still recorded in the
                        // report explicitly (see json_writer/md_writer) —
                        // they are never silently omitted.
                        if (result.status != BenchmarkStatus::OK) {
                            std::cerr << "  (status: " << benchmark_status_to_string(result.status)
                                      << " — recorded, no valid performance result)" << std::endl;
                        }

                        executed_any = true;

                        BenchmarkEntry entry;
                        entry.device_id = device.id;
                        entry.kernel_name = kernel->name();
                        entry.category = kernel->category();
                        entry.precision = precision_to_string(precision);
                        entry.mode = mode;
                        entry.config = config;
                        entry.result = result;
                        report.benchmarks.push_back(entry);
                    }
                }
            }
        }
    }

    // ========================================================================
    // Summary table
    // ========================================================================
    std::cerr << "\n";

    // An empty benchmark selection is a failure, not a silently-successful
    // empty report — e.g. every combination was unsupported, or --kernel /
    // --precision / --device / --kernels narrowed selection to nothing.
    if (!executed_any) {
        std::cerr << "ERROR: no benchmarks were executed for the selected "
                   << "device/kernel/precision/category combination." << std::endl;
        return 1;
    }

    // Group benchmarks by device
    std::map<std::string, std::vector<const BenchmarkEntry*>> by_device;
    for (auto& e : report.benchmarks) {
        by_device[e.device_id].push_back(&e);
    }

    for (auto& [dev_id, entries] : by_device) {
        // Find device name
        std::string dev_name = dev_id;
        for (auto& d : report.devices) {
            if (d.id == dev_id) { dev_name = d.name; break; }
        }

        std::cerr << "╔══════════════════════════════════════════════════════════════════════════════════════════════╗" << std::endl;
        std::cerr << "║  " << dev_id << " (" << dev_name << ")" << std::endl;
        std::cerr << "╠══════════════════════════════════════════════════════════════════════════════════════════════╣" << std::endl;
        std::cerr << "║ Kernel                   │ Prec     │ Mode       │             Rate │  Peak% │ Median (ms) ║" << std::endl;
        std::cerr << "╟──────────────────────────┼──────────┼────────────┼───────────────────┼────────┼─────────────╢" << std::endl;

        std::string prev_kernel;
        for (auto* e : entries) {
            // Separator between different kernels
            if (!prev_kernel.empty() && prev_kernel != e->kernel_name) {
                std::cerr << "╟──────────────────────────┼──────────┼────────────┼───────────────────┼────────┼─────────────╢" << std::endl;
            }
            prev_kernel = e->kernel_name;

            // Format kernel name (truncate to 24 chars)
            std::string kname = e->kernel_name;
            if (kname.size() > 24) kname = kname.substr(0, 24);

            // Format rate using the typed metric's own unit — never
            // category-based guessing. Non-OK statuses render an explicit
            // status marker instead of a fabricated zero-performance rate.
            bool is_ok = (e->result.status == BenchmarkStatus::OK);
            char rate_buf[24];
            if (is_ok) {
                double val = e->result.metric.rate_per_second;
                std::string unit = metric_unit_for_kind(e->result.metric.kind);

                double scaled;
                char prefix;
                if (val >= 1e15) {
                    scaled = val / 1e15; prefix = 'P';
                } else if (val >= 1e12) {
                    scaled = val / 1e12; prefix = 'T';
                } else if (val >= 1e9) {
                    scaled = val / 1e9;  prefix = 'G';
                } else if (val >= 1e6) {
                    scaled = val / 1e6;  prefix = 'M';
                } else {
                    scaled = val;        prefix = ' ';
                }

                snprintf(rate_buf, sizeof(rate_buf), "%7.1f %c%s", scaled, prefix, unit.c_str());
            } else {
                snprintf(rate_buf, sizeof(rate_buf), "%16s", benchmark_status_to_string(e->result.status).c_str());
            }

            // Format peak%
            char peak_buf[10];
            if (is_ok) {
                snprintf(peak_buf, sizeof(peak_buf), "%6.1f%%", e->result.peak_percent);
            } else {
                snprintf(peak_buf, sizeof(peak_buf), "%6s", "—");
            }

            // Format time
            char time_buf[14];
            snprintf(time_buf, sizeof(time_buf), "%9.3f", e->result.median_time_ms);

            // Format mode (truncate)
            std::string mode = e->mode;
            if (mode.size() > 10) mode = mode.substr(0, 10);

            fprintf(stderr, "║ %-24s │ %-8s │ %-10s │ %17s │ %6s │ %11s ║\n",
                    kname.c_str(),
                    e->precision.c_str(),
                    mode.c_str(),
                    rate_buf,
                    peak_buf,
                    time_buf);
        }

        std::cerr << "╚══════════════════════════════════════════════════════════════════════════════════════════════╝" << std::endl;
        std::cerr << std::endl;
    }

    // Write JSON report: --report=stdout emits JSON to stdout (in addition to
    // any --output file); --report=json (default) preserves the existing
    // behavior of only writing when --output is given. Report write failures
    // are not propagated to the exit code here: write_json_report/
    // write_markdown_report (report.hpp / src/report/*) are out of scope for
    // this change and currently return void with no error signal — see
    // TODO.md "Ensure report-write failures propagate to the process exit
    // code" which remains unchecked pending a signature change there.
    if (opts.report_format == "stdout") {
        std::cout << report_to_json(report).dump(2) << std::endl;
    }
    write_json_report(report, opts.output_path);

    // Write markdown report if --output-md was specified
    write_markdown_report(report, opts.output_md_path);

    return 0;
}
