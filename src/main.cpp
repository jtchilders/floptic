#include "floptic/cli_parser.hpp"
#include "floptic/device_info.hpp"
#include "floptic/kernel_registry.hpp"
#include "floptic/report.hpp"
#include <iostream>
#include <algorithm>

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
        extern void gemm_cublas_link();
        extern void gemm_cublas_notc_link();
#endif
    }
    static void force_link_all() {
        force_link::scalar_fma_cpu_link();
        force_link::vector_axpy_cpu_link();
#ifdef FLOPTIC_HAS_CUDA
        force_link::scalar_fma_cuda_link();
        force_link::vector_axpy_cuda_link();
        force_link::gemm_cublas_link();
        force_link::gemm_cublas_notc_link();
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
    auto gpus = discover_cuda_devices();
    all.insert(all.end(), gpus.begin(), gpus.end());
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
        write_json_report(info_report, opts.output_path);
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
        }
        if (selected) target_devices.push_back(dev);
    }

    if (target_devices.empty()) {
        std::cerr << "No matching devices for: ";
        for (auto& d : opts.devices) std::cerr << d << " ";
        std::cerr << std::endl;
        return 1;
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

    std::cerr << "\nRunning benchmarks..." << std::endl;

    for (auto& device : target_devices) {
        std::cerr << "\n=== Device: " << device.id << " (" << device.name << ") ===" << std::endl;

        // Determine backend for this device
        std::string dev_backend = (device.type == "gpu") ? "cuda" : "cpu";
        // TODO: distinguish cuda vs hip vs sycl based on device.id prefix

        // Get matching kernels
        for (auto& cat : opts.kernel_categories) {
            auto kernels = registry.get_kernels(cat, dev_backend);

            for (auto* kernel : kernels) {
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
                        config.gpu_blocks = opts.gpu_blocks;
                        config.gpu_threads_per_block = opts.gpu_threads_per_block;
                        config.gpu_blocks_per_sm = opts.gpu_blocks_per_sm;

                        std::cerr << "\n--- " << kernel->name() << " | "
                                  << precision_to_string(precision) << " | "
                                  << mode << " ---" << std::endl;

                        auto result = kernel->run(config, device, opts.trials);

                        std::cerr << "  Result: " << result.gflops << " GFLOP/s"
                                  << " (median " << result.median_time_ms << " ms"
                                  << ", peak " << result.peak_percent << "%)"
                                  << std::endl;

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

    std::cerr << "\n=== Generating report ===" << std::endl;
    write_json_report(report, opts.output_path);

    return 0;
}
