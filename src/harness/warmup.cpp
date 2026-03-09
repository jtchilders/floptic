#include <iostream>

namespace floptic {

// Warmup is handled inline by each kernel's run() method.
// This file provides shared warmup utilities if needed.

void log_warmup(const std::string& kernel_name, int warmup_iters) {
    std::cerr << "  Warming up " << kernel_name
              << " (" << warmup_iters << " iterations)..." << std::endl;
}

} // namespace floptic
