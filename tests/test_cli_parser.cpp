// CLI-parsing tests that must not touch GPU discovery or kernel execution.
#include "floptic/cli_parser.hpp"
#include "check.hpp"

#include <cstring>
#include <vector>

using floptic::CliOptions;
using floptic::Precision;
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

    CHECK_TRUE(opts.valid);
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

    CHECK_TRUE(opts.valid);
    CHECK_TRUE(opts.list_kernels);
    CHECK_TRUE(!opts.help);
}

void test_cli_help_alone_is_valid_and_exit0_worthy() {
    Argv args({"--help"});
    CliOptions opts = parse_args(args.argc(), args.argv());

    CHECK_TRUE(opts.valid);
    CHECK_TRUE(opts.help);
}

// ---------------------------------------------------------------------
// Requirement 1/2: parsing returns structured failure, not throw/help-abuse.
// ---------------------------------------------------------------------

void test_unknown_option_is_structured_failure_not_help() {
    Argv args({"--bogus-flag=1"});
    CliOptions opts = parse_args(args.argc(), args.argv());

    CHECK_TRUE(!opts.valid);
    CHECK_TRUE(!opts.error.empty());
    // Must not silently overload --help to signal failure.
    CHECK_TRUE(!opts.help);
}

// ---------------------------------------------------------------------
// Requirement 3: numeric rules (malformed + out-of-range).
// ---------------------------------------------------------------------

void test_trials_must_be_positive() {
    {
        Argv args({"--trials=0"});
        CliOptions opts = parse_args(args.argc(), args.argv());
        CHECK_TRUE(!opts.valid);
    }
    {
        Argv args({"--trials=-5"});
        CliOptions opts = parse_args(args.argc(), args.argv());
        CHECK_TRUE(!opts.valid);
    }
    {
        Argv args({"--trials=7"});
        CliOptions opts = parse_args(args.argc(), args.argv());
        CHECK_TRUE(opts.valid);
        CHECK_EQ(opts.trials, 7);
    }
}

void test_inner_iters_must_be_positive() {
    Argv args({"--inner-iters=0"});
    CliOptions opts = parse_args(args.argc(), args.argv());
    CHECK_TRUE(!opts.valid);
}

void test_warmup_allows_zero_but_not_negative() {
    {
        Argv args({"--warmup=0"});
        CliOptions opts = parse_args(args.argc(), args.argv());
        CHECK_TRUE(opts.valid);
        CHECK_EQ(opts.warmup, 0);
    }
    {
        Argv args({"--warmup=-1"});
        CliOptions opts = parse_args(args.argc(), args.argv());
        CHECK_TRUE(!opts.valid);
    }
}

void test_cpu_threads_and_gpu_blocks_allow_zero_as_auto() {
    {
        Argv args({"--cpu-threads=0"});
        CliOptions opts = parse_args(args.argc(), args.argv());
        CHECK_TRUE(opts.valid);
        CHECK_EQ(opts.cpu_threads, 0);
    }
    {
        Argv args({"--cpu-threads=-1"});
        CliOptions opts = parse_args(args.argc(), args.argv());
        CHECK_TRUE(!opts.valid);
    }
    {
        Argv args({"--gpu-blocks=0"});
        CliOptions opts = parse_args(args.argc(), args.argv());
        CHECK_TRUE(opts.valid);
        CHECK_EQ(opts.gpu_blocks, 0);
    }
    {
        Argv args({"--gpu-blocks=-2"});
        CliOptions opts = parse_args(args.argc(), args.argv());
        CHECK_TRUE(!opts.valid);
    }
}

void test_gpu_tpb_and_bpsm_must_be_positive() {
    {
        Argv args({"--gpu-tpb=0"});
        CliOptions opts = parse_args(args.argc(), args.argv());
        CHECK_TRUE(!opts.valid);
    }
    {
        Argv args({"--gpu-bpsm=0"});
        CliOptions opts = parse_args(args.argc(), args.argv());
        CHECK_TRUE(!opts.valid);
    }
    {
        Argv args({"--gpu-tpb=128", "--gpu-bpsm=2"});
        CliOptions opts = parse_args(args.argc(), args.argv());
        CHECK_TRUE(opts.valid);
        CHECK_EQ(opts.gpu_threads_per_block, 128);
        CHECK_EQ(opts.gpu_blocks_per_sm, 2);
    }
}

void test_malformed_integers_are_caught_without_throwing() {
    const std::vector<std::string> bad_values = {
        "--trials=abc", "--trials=1.5", "--trials=", "--trials=1x",
        "--warmup=nan", "--cpu-threads=--3", "--gpu-tpb=99999999999999999999"
    };
    for (const auto& bad : bad_values) {
        Argv args({bad});
        CliOptions opts = parse_args(args.argc(), args.argv());
        CHECK_TRUE(!opts.valid);
    }
}

// ---------------------------------------------------------------------
// Requirement 4: reject empty list values and unknown enum-like values.
// ---------------------------------------------------------------------

void test_empty_list_values_are_rejected() {
    {
        Argv args({"--precision="});
        CliOptions opts = parse_args(args.argc(), args.argv());
        CHECK_TRUE(!opts.valid);
    }
    {
        Argv args({"--device="});
        CliOptions opts = parse_args(args.argc(), args.argv());
        CHECK_TRUE(!opts.valid);
    }
    {
        Argv args({"--kernels="});
        CliOptions opts = parse_args(args.argc(), args.argv());
        CHECK_TRUE(!opts.valid);
    }
    {
        // Empty element within an otherwise valid list.
        Argv args({"--precision=fp64,,fp32"});
        CliOptions opts = parse_args(args.argc(), args.argv());
        CHECK_TRUE(!opts.valid);
    }
}

void test_unknown_precision_is_rejected_not_mapped_to_fp64() {
    Argv args({"--precision=bogus"});
    CliOptions opts = parse_args(args.argc(), args.argv());
    CHECK_TRUE(!opts.valid);
}

void test_unknown_device_selector_is_rejected() {
    Argv args({"--device=tpu:0"});
    CliOptions opts = parse_args(args.argc(), args.argv());
    CHECK_TRUE(!opts.valid);
}

void test_unknown_kernel_category_is_rejected() {
    Argv args({"--kernels=bogus"});
    CliOptions opts = parse_args(args.argc(), args.argv());
    CHECK_TRUE(!opts.valid);
}

void test_unknown_report_format_is_rejected() {
    Argv args({"--report=csv"});
    CliOptions opts = parse_args(args.argc(), args.argv());
    CHECK_TRUE(!opts.valid);
}

void test_supported_report_formats_are_accepted() {
    {
        Argv args({"--report=json"});
        CliOptions opts = parse_args(args.argc(), args.argv());
        CHECK_TRUE(opts.valid);
        CHECK_EQ(opts.report_format, std::string("json"));
    }
    {
        Argv args({"--report=stdout"});
        CliOptions opts = parse_args(args.argc(), args.argv());
        CHECK_TRUE(opts.valid);
        CHECK_EQ(opts.report_format, std::string("stdout"));
    }
}

// ---------------------------------------------------------------------
// Requirement 5: dedup preserving first-seen order; "all" dominates.
// ---------------------------------------------------------------------

void test_device_list_dedups_preserving_first_seen_order() {
    Argv args({"--device=cpu,cuda:0,cpu,cuda:0,cuda:1"});
    CliOptions opts = parse_args(args.argc(), args.argv());
    CHECK_TRUE(opts.valid);
    CHECK_EQ(opts.devices.size(), static_cast<size_t>(3));
    CHECK_EQ(opts.devices[0], std::string("cpu"));
    CHECK_EQ(opts.devices[1], std::string("cuda:0"));
    CHECK_EQ(opts.devices[2], std::string("cuda:1"));
}

void test_device_all_dominates_other_values() {
    Argv args({"--device=cpu,all,cuda:0"});
    CliOptions opts = parse_args(args.argc(), args.argv());
    CHECK_TRUE(opts.valid);
    CHECK_EQ(opts.devices.size(), static_cast<size_t>(1));
    CHECK_EQ(opts.devices[0], std::string("all"));
}

void test_kernel_categories_dedup_and_all_dominates() {
    {
        Argv args({"--kernels=scalar,vector,scalar"});
        CliOptions opts = parse_args(args.argc(), args.argv());
        CHECK_TRUE(opts.valid);
        CHECK_EQ(opts.kernel_categories.size(), static_cast<size_t>(2));
        CHECK_EQ(opts.kernel_categories[0], std::string("scalar"));
        CHECK_EQ(opts.kernel_categories[1], std::string("vector"));
    }
    {
        Argv args({"--kernels=scalar,all,vector"});
        CliOptions opts = parse_args(args.argc(), args.argv());
        CHECK_TRUE(opts.valid);
        CHECK_EQ(opts.kernel_categories.size(), static_cast<size_t>(1));
        CHECK_EQ(opts.kernel_categories[0], std::string("all"));
    }
}

void test_precision_list_dedups_across_spellings_and_all_dominates() {
    {
        // "fp8" and "fp8e4m3" both map to FP8_E4M3 — must dedup to one entry.
        Argv args({"--precision=fp64,fp8,fp8e4m3,fp32"});
        CliOptions opts = parse_args(args.argc(), args.argv());
        CHECK_TRUE(opts.valid);
        CHECK_EQ(opts.precisions.size(), static_cast<size_t>(3));
        CHECK_TRUE(opts.precisions[0] == Precision::FP64);
        CHECK_TRUE(opts.precisions[1] == Precision::FP8_E4M3);
        CHECK_TRUE(opts.precisions[2] == Precision::FP32);
    }
    {
        Argv args({"--precision=fp64,all,fp32"});
        CliOptions opts = parse_args(args.argc(), args.argv());
        CHECK_TRUE(opts.valid);
        CHECK_TRUE(opts.precisions.size() > 2);
    }
}

// ---------------------------------------------------------------------
// --kernel=<NAME> stores the exact requested name (existence is validated
// against the live kernel registry in main.cpp, not here).
// ---------------------------------------------------------------------

void test_kernel_name_flag_is_captured_verbatim() {
    Argv args({"--kernel=scalar_fma"});
    CliOptions opts = parse_args(args.argc(), args.argv());
    CHECK_TRUE(opts.valid);
    CHECK_EQ(opts.kernel_name, std::string("scalar_fma"));
}

void test_kernel_name_flag_rejects_empty_value() {
    Argv args({"--kernel="});
    CliOptions opts = parse_args(args.argc(), args.argv());
    CHECK_TRUE(!opts.valid);
}

// ---------------------------------------------------------------------
// Multiple errors accumulate rather than stopping at the first.
// ---------------------------------------------------------------------

void test_multiple_errors_are_all_reported() {
    Argv args({"--trials=0", "--precision=bogus", "--unknown-flag"});
    CliOptions opts = parse_args(args.argc(), args.argv());
    CHECK_TRUE(!opts.valid);
    // Expect at least 3 newline-separated diagnostics.
    int newline_count = 0;
    for (char c : opts.error) if (c == '\n') newline_count++;
    CHECK_TRUE(newline_count >= 2);
}

} // namespace

int main() {
    test_cli_defaults_parse_without_any_args();
    test_cli_list_flag_parses_without_device_discovery();
    test_cli_help_alone_is_valid_and_exit0_worthy();
    test_unknown_option_is_structured_failure_not_help();
    test_trials_must_be_positive();
    test_inner_iters_must_be_positive();
    test_warmup_allows_zero_but_not_negative();
    test_cpu_threads_and_gpu_blocks_allow_zero_as_auto();
    test_gpu_tpb_and_bpsm_must_be_positive();
    test_malformed_integers_are_caught_without_throwing();
    test_empty_list_values_are_rejected();
    test_unknown_precision_is_rejected_not_mapped_to_fp64();
    test_unknown_device_selector_is_rejected();
    test_unknown_kernel_category_is_rejected();
    test_unknown_report_format_is_rejected();
    test_supported_report_formats_are_accepted();
    test_device_list_dedups_preserving_first_seen_order();
    test_device_all_dominates_other_values();
    test_kernel_categories_dedup_and_all_dominates();
    test_precision_list_dedups_across_spellings_and_all_dominates();
    test_kernel_name_flag_is_captured_verbatim();
    test_kernel_name_flag_rejects_empty_value();
    test_multiple_errors_are_all_reported();
    return floptic_test::finish();
}
