#include <iostream>
#include <cmath>

namespace floptic {

// Validation sink: prevent dead code elimination by consuming results
// The volatile write ensures the compiler cannot optimize away computation.
volatile double g_validation_sink = 0.0;

void consume_result(double value) {
    g_validation_sink = value;
}

bool validate_finite(double value, const std::string& context) {
    if (std::isnan(value) || std::isinf(value)) {
        std::cerr << "WARNING: Non-finite result in " << context
                  << ": " << value << std::endl;
        return false;
    }
    return true;
}

} // namespace floptic
