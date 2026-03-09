#pragma once
#include <vector>
#include <algorithm>
#include <chrono>

namespace floptic {

// CPU timer using steady_clock
class CpuTimer {
public:
    void start() {
        start_ = std::chrono::steady_clock::now();
    }

    void stop() {
        stop_ = std::chrono::steady_clock::now();
    }

    // Elapsed time in milliseconds
    double elapsed_ms() const {
        return std::chrono::duration<double, std::milli>(stop_ - start_).count();
    }

private:
    std::chrono::steady_clock::time_point start_, stop_;
};

// Compute statistics from a vector of trial times
struct TimingStats {
    double median_ms = 0.0;
    double min_ms = 0.0;
    double max_ms = 0.0;

    static TimingStats compute(std::vector<double>& times) {
        TimingStats stats;
        if (times.empty()) return stats;

        std::sort(times.begin(), times.end());
        stats.min_ms = times.front();
        stats.max_ms = times.back();

        size_t n = times.size();
        if (n % 2 == 0)
            stats.median_ms = (times[n/2 - 1] + times[n/2]) / 2.0;
        else
            stats.median_ms = times[n/2];

        return stats;
    }
};

} // namespace floptic
