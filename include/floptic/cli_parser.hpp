#pragma once
#include <string>
#include <vector>
#include "floptic/precision.hpp"

namespace floptic {

struct CliOptions {
    std::vector<std::string> devices;       // e.g., {"cuda:0", "cpu:0"} or {"all"}
    std::vector<Precision> precisions;      // e.g., {FP64, FP32}
    std::vector<std::string> kernel_categories; // e.g., {"scalar"} or {"all"}
    std::string kernel_name;                // specific kernel, empty = all matching
    int iterations = 100;                   // measurement trials
    int warmup = 10;                        // warmup iterations
    std::string report_format = "json";     // "json", "csv", "stdout"
    std::string output_path;                // empty = stdout
    bool list_kernels = false;
    bool show_info = false;
    bool help = false;
};

CliOptions parse_args(int argc, char* argv[]);
void print_usage(const char* progname);

} // namespace floptic
