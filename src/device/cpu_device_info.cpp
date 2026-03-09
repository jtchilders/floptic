#include "floptic/device_info.hpp"
#include <fstream>
#include <sstream>
#include <thread>

#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

namespace floptic {

static std::string get_cpu_name() {
#ifdef __APPLE__
    char buf[256];
    size_t len = sizeof(buf);
    if (sysctlbyname("machdep.cpu.brand_string", buf, &len, nullptr, 0) == 0)
        return std::string(buf);
    return "Apple CPU";
#else
    // Linux: parse /proc/cpuinfo
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line;
    while (std::getline(cpuinfo, line)) {
        if (line.find("model name") != std::string::npos) {
            auto pos = line.find(':');
            if (pos != std::string::npos)
                return line.substr(pos + 2);
        }
    }
    return "Unknown CPU";
#endif
}

static int get_cpu_freq_mhz() {
#ifdef __APPLE__
    uint64_t freq = 0;
    size_t len = sizeof(freq);
    // Try getting nominal frequency
    if (sysctlbyname("hw.cpufrequency", &freq, &len, nullptr, 0) == 0)
        return static_cast<int>(freq / 1000000);
    return 0;
#else
    // Linux: parse /proc/cpuinfo for "cpu MHz"
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line;
    while (std::getline(cpuinfo, line)) {
        if (line.find("cpu MHz") != std::string::npos) {
            auto pos = line.find(':');
            if (pos != std::string::npos)
                return static_cast<int>(std::stod(line.substr(pos + 2)));
        }
    }
    return 0;
#endif
}

static std::string detect_arch() {
#if defined(__x86_64__) || defined(_M_X64)
    return "x86_64";
#elif defined(__aarch64__) || defined(_M_ARM64)
    return "aarch64";
#else
    return "unknown";
#endif
}

static std::string detect_vendor() {
#ifdef __APPLE__
    return "apple";
#elif defined(__x86_64__)
    // Could use CPUID for Intel vs AMD, simplified for now
    std::string name = get_cpu_name();
    if (name.find("Intel") != std::string::npos) return "intel";
    if (name.find("AMD") != std::string::npos) return "amd";
    return "unknown";
#else
    return "unknown";
#endif
}

std::vector<DeviceInfo> discover_cpu_devices() {
    DeviceInfo cpu;
    cpu.id = "cpu:0";
    cpu.name = get_cpu_name();
    cpu.vendor = detect_vendor();
    cpu.arch = detect_arch();
    cpu.type = "cpu";
    cpu.compute_units = std::thread::hardware_concurrency();
    cpu.clock_mhz = get_cpu_freq_mhz();
    cpu.boost_clock_mhz = cpu.clock_mhz;  // best we can do without root

    // All CPUs support FP64 and FP32
    cpu.supported_precisions = {Precision::FP64, Precision::FP32};
    cpu.features.push_back(Feature::FMA_HW);

    // TODO: CPUID for AVX2, AVX-512, AMX detection

    // Theoretical peak: cores × clock × FLOPs/cycle
    // Assuming FMA (2 FLOPs) and some SIMD width
    // Conservative: assume scalar FMA only for now
    double clock_ghz = cpu.boost_clock_mhz / 1000.0;
    if (clock_ghz > 0) {
        // Very conservative — just scalar FMA per core
        cpu.theoretical_peak_gflops["FP64"] = cpu.compute_units * clock_ghz * 2.0;
        cpu.theoretical_peak_gflops["FP32"] = cpu.compute_units * clock_ghz * 2.0;
    }

    return {cpu};
}

} // namespace floptic
