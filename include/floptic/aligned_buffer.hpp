#pragma once
#include <cstdlib>
#include <cstddef>

namespace floptic {

// Minimal RAII owner for a heap-aligned array of T.
//
// std::aligned_alloc can fail (return nullptr) just like malloc, and the
// call sites in this codebase previously used the result unconditionally —
// a failed allocation silently produced garbage reads/writes instead of a
// clean, visible failure. AlignedBuffer centralizes the allocate + validate
// + free lifecycle: construct it, check valid() before touching get(), and
// let the destructor free it (no explicit std::free at call sites, and no
// leak on early-return).
template <typename T>
class AlignedBuffer {
public:
    // alignment must be a power of two and, per the aligned_alloc contract,
    // the allocated size (count * sizeof(T), rounded up here) must be a
    // multiple of it. 64 bytes covers AVX2/AVX-512 vector widths.
    explicit AlignedBuffer(size_t count, size_t alignment = 64)
        : ptr_(nullptr), count_(count) {
        if (count_ == 0) return;
        size_t bytes = count_ * sizeof(T);
        size_t padded = ((bytes + alignment - 1) / alignment) * alignment;
        ptr_ = static_cast<T*>(std::aligned_alloc(alignment, padded));
    }

    ~AlignedBuffer() {
        std::free(ptr_);
    }

    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;

    // Allocation succeeded (or count was 0, which is a valid empty buffer).
    bool valid() const { return count_ == 0 || ptr_ != nullptr; }

    T* get() const { return ptr_; }
    size_t size() const { return count_; }

private:
    T* ptr_;
    size_t count_;
};

} // namespace floptic
