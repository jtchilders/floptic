#include <fstream>
#include <string>
#include <iostream>

namespace floptic {

// Best-effort power reading from RAPL (Intel CPUs, Linux)
// Returns power in watts, or 0.0 if unavailable
double read_cpu_power_watts() {
#ifdef __linux__
    // Try reading from powercap RAPL interface (user-readable on many systems)
    std::ifstream energy_file("/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj");
    if (energy_file.is_open()) {
        // Would need two readings and a time delta for power
        // For now, return 0 (placeholder)
        return 0.0;
    }
#endif
    return 0.0;
}

// NVML power reading is done in the CUDA kernel files directly
// since it requires linking against libnvidia-ml

} // namespace floptic
