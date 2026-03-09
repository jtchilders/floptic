#pragma once
#include <memory>
#include <vector>
#include <string>
#include "floptic/kernel_base.hpp"

namespace floptic {

class KernelRegistry {
public:
    static KernelRegistry& instance();

    void register_kernel(std::unique_ptr<KernelBase> kernel);

    // Get all kernels, optionally filtered
    std::vector<KernelBase*> get_kernels(
        const std::string& category = "",
        const std::string& backend = "") const;

    // Get a specific kernel by name
    KernelBase* get_kernel(const std::string& name,
                           const std::string& backend = "") const;

    // List all registered kernel names
    std::vector<std::string> list_kernel_names() const;

private:
    KernelRegistry() = default;
    std::vector<std::unique_ptr<KernelBase>> kernels_;
};

// Macro for self-registering kernels at static init time
#define REGISTER_KERNEL(KernelClass) \
    namespace { \
    static bool _reg_##KernelClass = [] { \
        ::floptic::KernelRegistry::instance().register_kernel( \
            std::make_unique<KernelClass>()); \
        return true; \
    }(); \
    }

} // namespace floptic
