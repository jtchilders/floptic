// CLI-parsing tests that must not touch GPU discovery or kernel execution.
#include "floptic/cli_parser.hpp"
#include "check.hpp"

#include <cstring>
#include <vector>

using floptic::CliOptions;
using floptic::parse_args;

namespace {

// Build an argv-style array from a fixed argv0 plus the given args.
struct Argv {
    std::vector<std::string> storage;
    std::vector<char*> ptrs;

    explicit Argv(const std::vector<std::string>& args) {
        storage.push_back("floptic");
        for (const auto& a : args) storage.push_back(a);
        for (auto& s : storage) ptrs.push_back(s.data());
    }

    int argc() const { return static_cast<int>(ptrs.size()); }
    char** argv() { return ptrs.data(); }
};

void test_cli_defaults_parse_without_any_args() {
    Argv args({});
    CliOptions opts = parse_args(args.argc(), args.argv());

    CHECK_TRUE(!opts.help);
    CHECK_TRUE(!opts.list_kernels);
    CHECK_TRUE(!opts.show_info);
    CHECK_EQ(opts.trials, 100);
    CHECK_EQ(opts.inner_iters, 100000);
    CHECK_EQ(opts.warmup, 10);
    CHECK_EQ(opts.report_format, std::string("json"));
    CHECK_EQ(opts.devices.size(), static_cast<size_t>(1));
    CHECK_EQ(opts.devices[0], std::string("all"));
    CHECK_EQ(opts.precisions.size(), static_cast<size_t>(2));
}

void test_cli_list_flag_parses_without_device_discovery() {
    Argv args({"--list"});
    CliOptions opts = parse_args(args.argc(), args.argv());

    CHECK_TRUE(opts.list_kernels);
    CHECK_TRUE(!opts.help);
}

} // namespace

int main() {
    test_cli_defaults_parse_without_any_args();
    test_cli_list_flag_parses_without_device_discovery();
    return floptic_test::finish();
}
