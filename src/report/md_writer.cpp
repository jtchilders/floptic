#include "floptic/report.hpp"
#include <iostream>
#include <fstream>
#include <cstdio>
#include <ctime>
#include <map>
#include <unistd.h>

namespace floptic {

static std::string md_get_hostname() {
    char buf[256];
    if (gethostname(buf, sizeof(buf)) == 0)
        return std::string(buf);
    return "unknown";
}

static std::string md_get_timestamp() {
    time_t now = time(nullptr);
    char buf[64];
    strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", localtime(&now));
    return std::string(buf);
}

static std::string format_rate(double gflops, bool is_memory) {
    char buf[32];
    if (gflops >= 1e6) {
        snprintf(buf, sizeof(buf), "%.2f PF/s", gflops / 1e6);
    } else if (gflops >= 1e3) {
        snprintf(buf, sizeof(buf), "%.1f TF/s", gflops / 1e3);
    } else if (gflops >= 1.0) {
        snprintf(buf, sizeof(buf), "%.1f GF/s", gflops);
    } else {
        snprintf(buf, sizeof(buf), "%.3f GF/s", gflops);
    }

    if (is_memory) {
        if (gflops >= 1e3) {
            snprintf(buf, sizeof(buf), "%.1f TB/s", gflops / 1e3);
        } else {
            snprintf(buf, sizeof(buf), "%.1f GB/s", gflops);
        }
    }

    return std::string(buf);
}

void write_markdown_report(const Report& report, const std::string& output_path) {
    if (output_path.empty()) return;

    std::ofstream out(output_path);
    if (!out.is_open()) {
        std::cerr << "ERROR: Cannot open markdown output file: " << output_path << std::endl;
        return;
    }

    std::string hostname = report.system.hostname.empty() ? md_get_hostname() : report.system.hostname;
    std::string timestamp = report.timestamp.empty() ? md_get_timestamp() : report.timestamp;

    out << "# Floptic Benchmark Results\n\n";

    // System info
    out << "## System Information\n\n";
    out << "| Property | Value |\n";
    out << "|----------|-------|\n";
    out << "| Date | " << timestamp << " |\n";
    out << "| Hostname | " << hostname << " |\n";
    out << "| Floptic | v" << report.version << " |\n";
    out << "| Trials | " << report.iterations << " |\n";

    if (!report.system.os_name.empty())
        out << "| OS | " << report.system.os_name << " " << report.system.os_release << " |\n";
    if (!report.system.os_arch.empty())
        out << "| Architecture | " << report.system.os_arch << " |\n";
    if (!report.system.cpu_model.empty())
        out << "| CPU | " << report.system.cpu_model << " |\n";
    if (!report.system.compiler_name.empty())
        out << "| Compiler | " << report.system.compiler_name << " " << report.system.compiler_version << " |\n";
    if (!report.system.cuda_runtime_version.empty())
        out << "| CUDA Runtime | " << report.system.cuda_runtime_version << " |\n";
    if (!report.system.cuda_driver_version.empty())
        out << "| CUDA Driver | " << report.system.cuda_driver_version << " |\n";
    if (!report.system.nvidia_smi_driver.empty())
        out << "| NVIDIA Driver | " << report.system.nvidia_smi_driver << " |\n";
    if (!report.system.cublas_version.empty())
        out << "| cuBLAS | " << report.system.cublas_version << " |\n";
    out << "\n";

    // Loaded modules
    if (!report.system.loaded_modules.empty()) {
        out << "### Loaded Modules\n\n";
        out << "```\n";
        for (auto& m : report.system.loaded_modules)
            out << m << "\n";
        out << "```\n\n";
    }

    // Group benchmarks by device
    std::map<std::string, std::vector<const BenchmarkEntry*>> by_device;
    for (auto& e : report.benchmarks) {
        by_device[e.device_id].push_back(&e);
    }

    // Device info sections
    for (auto& dev : report.devices) {
        bool has_benchmarks = by_device.count(dev.id) > 0;
        if (!has_benchmarks) continue;

        out << "## " << dev.name << " (`" << dev.id << "`)\n\n";

        // Device details
        out << "| Property | Value |\n";
        out << "|----------|-------|\n";
        out << "| Architecture | " << dev.arch << " |\n";
        if (dev.type == "gpu") {
            out << "| SMs | " << dev.compute_units << " |\n";
        } else {
            out << "| Physical Cores | " << dev.physical_cores << " |\n";
            out << "| Logical Cores | " << dev.compute_units << " |\n";
            out << "| Threads/Core | " << dev.threads_per_core << " |\n";
        }
        out << "| Clock | " << dev.boost_clock_mhz << " MHz |\n";

        char mem_buf[32];
        snprintf(mem_buf, sizeof(mem_buf), "%.1f GB",
                 dev.memory_bytes / (1024.0 * 1024.0 * 1024.0));
        out << "| Memory | " << mem_buf << " |\n";
        out << "\n";

        // Theoretical peaks
        if (!dev.theoretical_peak_gflops.empty()) {
            out << "### Theoretical Peaks\n\n";
            out << "| Precision | Peak |\n";
            out << "|-----------|------|\n";
            for (auto& [key, val] : dev.theoretical_peak_gflops) {
                out << "| " << key << " | " << format_rate(val, false) << " |\n";
            }
            out << "\n";
        }

        // Results table
        out << "### Benchmark Results\n\n";
        out << "| Kernel | Precision | Mode | Rate | Peak% | Median (ms) |\n";
        out << "|--------|-----------|------|------|-------|-------------|\n";

        for (auto* e : by_device[dev.id]) {
            bool is_memory = (e->category == "memory");
            std::string rate = format_rate(e->result.gflops, is_memory);

            char peak_buf[16];
            snprintf(peak_buf, sizeof(peak_buf), "%.1f%%", e->result.peak_percent);

            char time_buf[16];
            snprintf(time_buf, sizeof(time_buf), "%.3f", e->result.median_time_ms);

            out << "| " << e->kernel_name
                << " | " << e->precision
                << " | " << e->mode
                << " | " << rate
                << " | " << peak_buf
                << " | " << time_buf
                << " |\n";
        }
        out << "\n";

        // Key ratios section (compute from GEMM results)
        double fp64_gemm = 0, fp32_gemm = 0, fp16_gemm = 0, bf16_gemm = 0;
        double tf32_gemm = 0, int8_gemm = 0;

        for (auto* e : by_device[dev.id]) {
            if (e->kernel_name != "gemm_cublas") continue;
            if (e->precision == "FP64") fp64_gemm = e->result.gflops;
            else if (e->precision == "FP32") fp32_gemm = e->result.gflops;
            else if (e->precision == "FP16") fp16_gemm = e->result.gflops;
            else if (e->precision == "BF16") bf16_gemm = e->result.gflops;
            else if (e->precision == "TF32") tf32_gemm = e->result.gflops;
        }
        for (auto* e : by_device[dev.id]) {
            if (e->kernel_name == "gemm_cublas_int8" && e->precision == "INT8")
                int8_gemm = e->result.gflops;
        }

        if (fp64_gemm > 0 && (fp16_gemm > 0 || tf32_gemm > 0)) {
            out << "### GEMM Precision Ratios (vs FP64)\n\n";
            out << "| Precision | Rate | Ratio vs FP64 |\n";
            out << "|-----------|------|---------------|\n";

            char buf[64];
            snprintf(buf, sizeof(buf), "%.1f TF/s", fp64_gemm / 1e3);
            out << "| FP64 | " << buf << " | 1.0× |\n";

            if (fp32_gemm > 0) {
                snprintf(buf, sizeof(buf), "%.1f TF/s", fp32_gemm / 1e3);
                out << "| FP32 | " << buf << " | " ;
                snprintf(buf, sizeof(buf), "%.1f×", fp32_gemm / fp64_gemm);
                out << buf << " |\n";
            }
            if (tf32_gemm > 0) {
                out << "| TF32 | " << format_rate(tf32_gemm, false) << " | ";
                snprintf(buf, sizeof(buf), "%.1f×", tf32_gemm / fp64_gemm);
                out << buf << " |\n";
            }
            if (fp16_gemm > 0) {
                out << "| FP16 | " << format_rate(fp16_gemm, false) << " | ";
                snprintf(buf, sizeof(buf), "%.1f×", fp16_gemm / fp64_gemm);
                out << buf << " |\n";
            }
            if (bf16_gemm > 0) {
                out << "| BF16 | " << format_rate(bf16_gemm, false) << " | ";
                snprintf(buf, sizeof(buf), "%.1f×", bf16_gemm / fp64_gemm);
                out << buf << " |\n";
            }
            if (int8_gemm > 0) {
                out << "| INT8 | " << format_rate(int8_gemm, false) << " | ";
                snprintf(buf, sizeof(buf), "%.1f×", int8_gemm / fp64_gemm);
                out << buf << " |\n";
            }
            out << "\n";
        }
    }

    out << "---\n";
    out << "*Generated by [Floptic](https://github.com/jtchilders/floptic) v"
        << report.version << "*\n";

    out.close();
    std::cerr << "Markdown report written to: " << output_path << std::endl;
}

} // namespace floptic
