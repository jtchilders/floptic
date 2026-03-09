#include "floptic/device_info.hpp"
#include <fstream>
#include <sstream>
#include <thread>
#include <iostream>

#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

#if defined(__x86_64__) || defined(_M_X64)
#include <cpuid.h>
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
    std::string name = get_cpu_name();
    if (name.find("Intel") != std::string::npos) return "intel";
    if (name.find("AMD") != std::string::npos) return "amd";
    return "unknown";
#else
    return "unknown";
#endif
}

// Detect SIMD capabilities via CPUID
struct SimdCaps {
    bool sse2 = false;
    bool avx = false;
    bool avx2 = false;
    bool fma3 = false;
    bool avx512f = false;
    bool avx512_fp16 = false;

    // SIMD width in bits for peak calculation
    int simd_bits() const {
        if (avx512f) return 512;
        if (avx2 || avx) return 256;
        if (sse2) return 128;
        return 64;  // scalar
    }

    // FP64 FMA ops per cycle per core
    // Assuming 2 FMA units for modern x86 (Intel since Haswell, AMD since Zen)
    int fp64_fmas_per_cycle(const std::string& vendor) const {
        int lanes = simd_bits() / 64;  // FP64 lanes per SIMD register
        int fma_units = 2;             // most modern x86 have 2 FMA units
        // AMD Zen2/Zen3: 2× 256-bit FMA units
        // Intel Haswell+: 2× 256-bit FMA units
        // Intel SKX+: 2× 512-bit FMA units
        if (avx512f && vendor == "intel") {
            fma_units = 2;  // 2× 512-bit
        } else if (avx512f && vendor == "amd") {
            // AMD Zen4+ has AVX-512 but 256-bit execution units (cracks to 2 uops)
            // Effective: same as 2× 256-bit
            lanes = 256 / 64;
            fma_units = 2;
        }
        return lanes * fma_units;
    }

    // FP32 FMA ops per cycle per core
    int fp32_fmas_per_cycle(const std::string& vendor) const {
        int lanes = simd_bits() / 32;
        int fma_units = 2;
        if (avx512f && vendor == "amd") {
            lanes = 256 / 32;
            fma_units = 2;
        }
        return lanes * fma_units;
    }
};

static SimdCaps detect_simd() {
    SimdCaps caps;

#if defined(__x86_64__) || defined(_M_X64)
    unsigned int eax, ebx, ecx, edx;

    // CPUID function 1: basic features
    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        caps.sse2 = (edx >> 26) & 1;
        caps.avx  = (ecx >> 28) & 1;
        caps.fma3 = (ecx >> 12) & 1;
    }

    // CPUID function 7, subleaf 0: extended features
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        caps.avx2    = (ebx >> 5) & 1;
        caps.avx512f = (ebx >> 16) & 1;
        // AVX-512 FP16 is in CPUID.7.0:EDX bit 23
        caps.avx512_fp16 = (edx >> 23) & 1;
    }
#elif defined(__aarch64__)
    // AArch64 always has NEON (128-bit SIMD) and FMA
    caps.fma3 = true;
    // Treat as 128-bit SIMD equivalent
#endif

    return caps;
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
    cpu.boost_clock_mhz = cpu.clock_mhz;

    // All CPUs support FP64 and FP32
    cpu.supported_precisions = {Precision::FP64, Precision::FP32};

    // Detect SIMD
    auto simd = detect_simd();

    if (simd.fma3)    cpu.features.push_back(Feature::FMA_HW);
    if (simd.avx2)    cpu.features.push_back(Feature::AVX2);
    if (simd.avx512f) cpu.features.push_back(Feature::AVX512);
    if (simd.avx512_fp16) {
        cpu.features.push_back(Feature::AVX512_FP16);
        cpu.supported_precisions.push_back(Precision::FP16);
    }

    // Report detected SIMD
    std::cerr << "  CPU SIMD: " << simd.simd_bits() << "-bit"
              << (simd.avx512f ? " (AVX-512)" : simd.avx2 ? " (AVX2)" : simd.avx ? " (AVX)" : "")
              << (simd.fma3 ? " + FMA" : "")
              << std::endl;

    // Theoretical peak calculation
    double clock_ghz = cpu.boost_clock_mhz / 1000.0;
    int cores = cpu.compute_units;
    if (clock_ghz > 0 && cores > 0) {
        // FMA = 2 FLOPs (multiply + add)
        int fp64_fmas = simd.fp64_fmas_per_cycle(cpu.vendor);
        int fp32_fmas = simd.fp32_fmas_per_cycle(cpu.vendor);

        cpu.theoretical_peak_gflops["FP64"] = cores * clock_ghz * fp64_fmas * 2.0;
        cpu.theoretical_peak_gflops["FP32"] = cores * clock_ghz * fp32_fmas * 2.0;

        std::cerr << "  CPU theoretical peak: "
                  << cpu.theoretical_peak_gflops["FP64"] << " GFLOP/s FP64, "
                  << cpu.theoretical_peak_gflops["FP32"] << " GFLOP/s FP32"
                  << std::endl;
    }

    return {cpu};
}

} // namespace floptic
