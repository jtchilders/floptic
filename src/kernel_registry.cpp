#include "floptic/kernel_registry.hpp"
#include <iostream>

namespace floptic {

KernelRegistry& KernelRegistry::instance() {
    static KernelRegistry registry;
    return registry;
}

void KernelRegistry::register_kernel(std::unique_ptr<KernelBase> kernel) {
    // Guard against duplicate registration (can happen with static init + force_link)
    for (auto& k : kernels_) {
        if (k->name() == kernel->name() && k->backend() == kernel->backend()) {
            return;  // Already registered
        }
    }
    kernels_.push_back(std::move(kernel));
}

std::vector<KernelBase*> KernelRegistry::get_kernels(
        const std::string& category,
        const std::string& backend) const {
    std::vector<KernelBase*> result;
    for (auto& k : kernels_) {
        if (!category.empty() && category != "all" && k->category() != category)
            continue;
        if (!backend.empty() && k->backend() != backend)
            continue;
        result.push_back(k.get());
    }
    return result;
}

KernelBase* KernelRegistry::get_kernel(const std::string& name,
                                        const std::string& backend) const {
    for (auto& k : kernels_) {
        if (k->name() == name) {
            if (backend.empty() || k->backend() == backend)
                return k.get();
        }
    }
    return nullptr;
}

std::vector<std::string> KernelRegistry::list_kernel_names() const {
    std::vector<std::string> names;
    for (auto& k : kernels_) {
        names.push_back(k->name() + " [" + k->backend() + "/" + k->category() + "]");
    }
    return names;
}

} // namespace floptic
