#include "floptic/cli_parser.hpp"
#include <iostream>
#include <sstream>
#include <cstring>

namespace floptic {

static std::vector<std::string> split(const std::string& s, char delim) {
    std::vector<std::string> tokens;
    std::istringstream iss(s);
    std::string token;
    while (std::getline(iss, token, delim)) {
        if (!token.empty()) tokens.push_back(token);
    }
    return tokens;
}

void print_usage(const char* progname) {
    std::cout << "Usage: " << progname << " [options]\n"
              << "\nOptions:\n"
              << "  --device=<DEV>       Target device(s): cpu, cuda:N, all (default: all)\n"
              << "  --precision=<PREC>   Precisions: fp64, fp32, fp16, all (default: fp64,fp32)\n"
              << "  --kernels=<CAT>      Kernel categories: scalar, vector, matrix, sparse, emulated, all\n"
              << "  --kernel=<NAME>      Run specific kernel by name\n"
              << "  --iterations=<N>     Measurement trials (default: 100)\n"
              << "  --warmup=<N>         Warmup iterations (default: 10)\n"
              << "  --report=<FMT>       Output format: json, stdout (default: json)\n"
              << "  --output=<PATH>      Output file (default: stdout)\n"
              << "\nThread control:\n"
              << "  --cpu-threads=<N>    CPU threads (default: all cores)\n"
              << "  --gpu-blocks=<N>     GPU thread blocks (default: auto = blocks-per-sm × SMs)\n"
              << "  --gpu-tpb=<N>        GPU threads per block (default: 256)\n"
              << "  --gpu-bpsm=<N>       GPU blocks per SM (default: 4, used when --gpu-blocks=0)\n"
              << "\nOther:\n"
              << "  --list               List available kernels and exit\n"
              << "  --info               Print device info and exit\n"
              << "  --help               Show this help\n"
              << std::endl;
}

CliOptions parse_args(int argc, char* argv[]) {
    CliOptions opts;
    // Defaults
    opts.devices = {"all"};
    opts.precisions = {Precision::FP64, Precision::FP32};
    opts.kernel_categories = {"all"};

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            opts.help = true;
        } else if (arg == "--list") {
            opts.list_kernels = true;
        } else if (arg == "--info") {
            opts.show_info = true;
        } else if (arg.rfind("--device=", 0) == 0) {
            opts.devices = split(arg.substr(9), ',');
        } else if (arg.rfind("--precision=", 0) == 0) {
            auto strs = split(arg.substr(12), ',');
            opts.precisions.clear();
            for (auto& s : strs) {
                if (s == "all") {
                    opts.precisions = all_standard_precisions();
                    break;
                }
                opts.precisions.push_back(string_to_precision(s));
            }
        } else if (arg.rfind("--kernels=", 0) == 0) {
            opts.kernel_categories = split(arg.substr(10), ',');
        } else if (arg.rfind("--kernel=", 0) == 0) {
            opts.kernel_name = arg.substr(9);
        } else if (arg.rfind("--iterations=", 0) == 0) {
            opts.iterations = std::stoi(arg.substr(13));
        } else if (arg.rfind("--warmup=", 0) == 0) {
            opts.warmup = std::stoi(arg.substr(9));
        } else if (arg.rfind("--report=", 0) == 0) {
            opts.report_format = arg.substr(9);
        } else if (arg.rfind("--output=", 0) == 0) {
            opts.output_path = arg.substr(9);
        } else if (arg.rfind("--cpu-threads=", 0) == 0) {
            opts.cpu_threads = std::stoi(arg.substr(14));
        } else if (arg.rfind("--gpu-blocks=", 0) == 0) {
            opts.gpu_blocks = std::stoi(arg.substr(13));
        } else if (arg.rfind("--gpu-tpb=", 0) == 0) {
            opts.gpu_threads_per_block = std::stoi(arg.substr(10));
        } else if (arg.rfind("--gpu-bpsm=", 0) == 0) {
            opts.gpu_blocks_per_sm = std::stoi(arg.substr(11));
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            opts.help = true;
        }
    }

    return opts;
}

} // namespace floptic
