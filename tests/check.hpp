// Minimal, dependency-free test assertion helpers.
//
// Deliberately not a network-fetched framework: CI must stay deterministic
// and buildable offline. Each test translation unit defines its own main()
// that calls a handful of check functions and returns the shared exit code.
#pragma once

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>

namespace floptic_test {

inline int& failure_count() {
    static int count = 0;
    return count;
}

inline void report_failure(const std::string& expr, const std::string& file, int line,
                            const std::string& detail) {
    failure_count()++;
    std::cerr << "CHECK FAILED: " << expr << " (" << file << ":" << line << ")";
    if (!detail.empty()) {
        std::cerr << " -- " << detail;
    }
    std::cerr << "\n";
}

template <typename A, typename B>
inline void check_eq(const A& actual, const B& expected, const char* expr, const char* file,
                      int line) {
    if (!(actual == expected)) {
        std::ostringstream oss;
        oss << "actual=" << actual << " expected=" << expected;
        report_failure(expr, file, line, oss.str());
    }
}

inline void check_true(bool cond, const char* expr, const char* file, int line) {
    if (!cond) {
        report_failure(expr, file, line, "");
    }
}

// Returns process exit status: 0 if every CHECK_* call passed, 1 otherwise.
inline int finish() {
    if (failure_count() > 0) {
        std::cerr << failure_count() << " check(s) failed.\n";
        return EXIT_FAILURE;
    }
    std::cout << "All checks passed.\n";
    return EXIT_SUCCESS;
}

} // namespace floptic_test

#define CHECK_EQ(actual, expected) \
    ::floptic_test::check_eq((actual), (expected), #actual " == " #expected, __FILE__, __LINE__)

#define CHECK_TRUE(cond) \
    ::floptic_test::check_true((cond), #cond, __FILE__, __LINE__)
