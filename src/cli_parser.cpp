#include "floptic/cli_parser.hpp"
#include <iostream>
#include <sstream>
#include <cstring>
#include <cstdlib>
#include <cerrno>
#include <set>
#include <string>
#include <limits>

namespace floptic {

namespace {

const char* kValidReportFormats[] = {"json", "stdout"};
const char* kValidCategories[] = {"scalar", "vector", "matrix", "memory", "all"};

std::vector<std::string> split(const std::string& s, char delim) {
    std::vector<std::string> tokens;
    std::istringstream iss(s);
    std::string token;
    while (std::getline(iss, token, delim)) {
        tokens.push_back(token);
    }
    return tokens;
}

// Strict integer parse: rejects empty strings, trailing garbage, and
// anything std::strtol can't fully consume. Returns false (leaving `out`
// untouched) on any malformed input instead of throwing.
bool parse_int_strict(const std::string& s, long& out) {
    if (s.empty()) return false;
    errno = 0;
    char* end = nullptr;
    long val = std::strtol(s.c_str(), &end, 10);
    if (end == s.c_str() || *end != '\0') return false;  // no digits or trailing junk
    if (errno == ERANGE) return false;                    // overflow/underflow
    out = val;
    return true;
}

bool is_all_digits(const std::string& s) {
    if (s.empty()) return false;
    for (char c : s) {
        if (c < '0' || c > '9') return false;
    }
    return true;
}

// Accepts: "all", "cpu", "cuda", "hip", or "<backend>:<index>" where backend
// is cpu/cuda/hip and index is a nonnegative integer.
bool is_valid_device_token(const std::string& tok) {
    if (tok == "all" || tok == "cpu" || tok == "cuda" || tok == "hip") return true;
    auto pos = tok.find(':');
    if (pos == std::string::npos) return false;
    std::string backend = tok.substr(0, pos);
    std::string idx = tok.substr(pos + 1);
    if (backend != "cpu" && backend != "cuda" && backend != "hip") return false;
    return is_all_digits(idx);
}

bool is_valid_category_token(const std::string& tok) {
    for (auto* c : kValidCategories) {
        if (tok == c) return true;
    }
    return false;
}

bool is_valid_report_format(const std::string& tok) {
    for (auto* f : kValidReportFormats) {
        if (tok == f) return true;
    }
    return false;
}

std::vector<std::string> dedup_preserve_order(const std::vector<std::string>& in) {
    std::vector<std::string> out;
    out.reserve(in.size());
    std::set<std::string> seen;
    for (auto& s : in) {
        if (seen.insert(s).second) out.push_back(s);
    }
    return out;
}

// Validates and normalizes a comma-separated list option. Returns false and
// appends to `errors` on any problem: empty value, empty element, or an
// element that fails `is_valid`. On success, applies "all dominates" then
// dedups while preserving first-seen order.
bool parse_and_validate_list(const std::string& flag_name,
                              const std::string& raw_value,
                              bool (*is_valid)(const std::string&),
                              std::vector<std::string>& out,
                              std::vector<std::string>& errors) {
    if (raw_value.empty()) {
        errors.push_back(flag_name + " requires a non-empty value");
        return false;
    }
    auto tokens = split(raw_value, ',');
    if (tokens.empty()) {
        errors.push_back(flag_name + " requires a non-empty value");
        return false;
    }
    bool ok = true;
    for (auto& t : tokens) {
        if (t.empty()) {
            errors.push_back(flag_name + ": empty element in comma-separated list");
            ok = false;
            continue;
        }
        if (!is_valid(t)) {
            errors.push_back(flag_name + ": unrecognized value '" + t + "'");
            ok = false;
        }
    }
    if (!ok) return false;

    // "all" dominates any other values in the same list.
    bool has_all = false;
    for (auto& t : tokens) {
        if (t == "all") { has_all = true; break; }
    }
    if (has_all) {
        out = {"all"};
    } else {
        out = dedup_preserve_order(tokens);
    }
    return true;
}

// Parses and validates the precision list. Precision needs its own path
// because tokens map through string_to_precision/all_standard_precisions
// rather than being kept as raw strings, and dedup must happen on the
// resulting enum values (multiple spellings can map to the same Precision).
bool parse_and_validate_precisions(const std::string& raw_value,
                                    std::vector<Precision>& out,
                                    std::vector<std::string>& errors) {
    if (raw_value.empty()) {
        errors.push_back("--precision requires a non-empty value");
        return false;
    }
    auto tokens = split(raw_value, ',');
    if (tokens.empty()) {
        errors.push_back("--precision requires a non-empty value");
        return false;
    }

    bool ok = true;
    bool has_all = false;
    std::vector<Precision> parsed;
    for (auto& t : tokens) {
        if (t.empty()) {
            errors.push_back("--precision: empty element in comma-separated list");
            ok = false;
            continue;
        }
        if (t == "all") {
            has_all = true;
            continue;
        }
        Precision p;
        if (!try_string_to_precision(t, p)) {
            errors.push_back("--precision: unrecognized value '" + t + "'");
            ok = false;
            continue;
        }
        parsed.push_back(p);
    }
    if (!ok) return false;

    if (has_all) {
        out = all_standard_precisions();
        return true;
    }

    // Dedup preserving first-seen order.
    out.clear();
    std::set<int> seen;
    for (auto p : parsed) {
        if (seen.insert(static_cast<int>(p)).second) out.push_back(p);
    }
    return true;
}

// Strict, range-checked integer option parse. Appends a concise diagnostic
// to `errors` (without throwing) for malformed or out-of-range input.
bool parse_int_option(const std::string& flag_name,
                       const std::string& raw_value,
                       long min_value,
                       const char* range_description,
                       int& out,
                       std::vector<std::string>& errors) {
    long val = 0;
    if (!parse_int_strict(raw_value, val)) {
        errors.push_back(flag_name + ": invalid integer value '" + raw_value + "'");
        return false;
    }
    if (val < min_value || val > std::numeric_limits<int>::max()) {
        errors.push_back(flag_name + ": value " + std::to_string(val) +
                          " out of range (" + range_description + ")");
        return false;
    }
    out = static_cast<int>(val);
    return true;
}

} // namespace

void print_usage(const char* progname) {
    std::cout << "Usage: " << progname << " [options]\n"
              << "\nOptions:\n"
              << "  --device=<DEV>       Target device(s): cpu, cuda:N, hip:N, all (default: all)\n"
              << "  --precision=<PREC>   Precisions: fp64, fp32, fp16, bf16, tf32, int8,\n"
              << "                       fp8e4m3, fp8e5m2, fp4, all (default: fp64,fp32)\n"
              << "  --kernels=<CAT>      Kernel categories: scalar, vector, matrix, memory, all\n"
              << "  --kernel=<NAME>      Run specific kernel by name (must match a registered kernel)\n"
              << "  --trials=<N>         Measurement trials for statistics (N > 0, default: 100)\n"
              << "  --inner-iters=<N>    Inner loop iterations per trial (N > 0, default: 100000)\n"
              << "  --warmup=<N>         Warmup iterations (N >= 0, default: 10)\n"
              << "  --report=<FMT>       Output format: json, stdout (default: json)\n"
              << "  --output=<PATH>      Output file (default: stdout for --report=stdout, otherwise none)\n"
              << "  --output-md=<PATH>   Write markdown results report to file\n"
              << "\nThread control:\n"
              << "  --cpu-threads=<N>    CPU threads (N >= 0; 0 = auto, default: 0)\n"
              << "  --gpu-blocks=<N>     GPU thread blocks (N >= 0; 0 = auto = blocks-per-sm x SMs)\n"
              << "  --gpu-tpb=<N>        GPU threads per block (N > 0, default: 256)\n"
              << "  --gpu-bpsm=<N>       GPU blocks per SM (N > 0, default: 4, used when --gpu-blocks=0)\n"
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

    std::vector<std::string> errors;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            opts.help = true;
        } else if (arg == "--list") {
            opts.list_kernels = true;
        } else if (arg == "--info") {
            opts.show_info = true;
        } else if (arg.rfind("--device=", 0) == 0) {
            std::vector<std::string> devices;
            if (parse_and_validate_list("--device", arg.substr(9), is_valid_device_token,
                                        devices, errors)) {
                opts.devices = devices;
            }
        } else if (arg.rfind("--precision=", 0) == 0) {
            std::vector<Precision> precisions;
            if (parse_and_validate_precisions(arg.substr(12), precisions, errors)) {
                opts.precisions = precisions;
            }
        } else if (arg.rfind("--kernels=", 0) == 0) {
            std::vector<std::string> categories;
            if (parse_and_validate_list("--kernels", arg.substr(10), is_valid_category_token,
                                        categories, errors)) {
                opts.kernel_categories = categories;
            }
        } else if (arg.rfind("--kernel=", 0) == 0) {
            std::string name = arg.substr(9);
            if (name.empty()) {
                errors.push_back("--kernel requires a non-empty value");
            } else {
                opts.kernel_name = name;
            }
        } else if (arg.rfind("--trials=", 0) == 0) {
            parse_int_option("--trials", arg.substr(9), 1, "must be > 0", opts.trials, errors);
        } else if (arg.rfind("--inner-iters=", 0) == 0) {
            parse_int_option("--inner-iters", arg.substr(14), 1, "must be > 0", opts.inner_iters, errors);
        } else if (arg.rfind("--warmup=", 0) == 0) {
            parse_int_option("--warmup", arg.substr(9), 0, "must be >= 0", opts.warmup, errors);
        } else if (arg.rfind("--report=", 0) == 0) {
            std::string fmt = arg.substr(9);
            if (!is_valid_report_format(fmt)) {
                errors.push_back("--report: unrecognized value '" + fmt +
                                  "' (supported: json, stdout)");
            } else {
                opts.report_format = fmt;
            }
        } else if (arg.rfind("--output=", 0) == 0) {
            opts.output_path = arg.substr(9);
        } else if (arg.rfind("--output-md=", 0) == 0) {
            opts.output_md_path = arg.substr(12);
        } else if (arg.rfind("--cpu-threads=", 0) == 0) {
            parse_int_option("--cpu-threads", arg.substr(14), 0, "must be >= 0 (0=auto)",
                              opts.cpu_threads, errors);
        } else if (arg.rfind("--gpu-blocks=", 0) == 0) {
            parse_int_option("--gpu-blocks", arg.substr(13), 0, "must be >= 0 (0=auto)",
                              opts.gpu_blocks, errors);
        } else if (arg.rfind("--gpu-tpb=", 0) == 0) {
            parse_int_option("--gpu-tpb", arg.substr(10), 1, "must be > 0",
                              opts.gpu_threads_per_block, errors);
        } else if (arg.rfind("--gpu-bpsm=", 0) == 0) {
            parse_int_option("--gpu-bpsm", arg.substr(11), 1, "must be > 0",
                              opts.gpu_blocks_per_sm, errors);
        } else {
            errors.push_back("Unknown option: " + arg);
        }
    }

    if (!errors.empty()) {
        opts.valid = false;
        std::ostringstream oss;
        for (size_t i = 0; i < errors.size(); i++) {
            if (i > 0) oss << "\n";
            oss << errors[i];
        }
        opts.error = oss.str();
    }

    return opts;
}

} // namespace floptic
