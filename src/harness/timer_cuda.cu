#ifdef FLOPTIC_HAS_CUDA
#include <cuda_runtime.h>

// CUDA event-based timer utility functions.
// Kernels use these directly rather than through a class,
// since GPU timing must wrap kernel launches.

namespace floptic {
namespace cuda {

void create_events(cudaEvent_t& start, cudaEvent_t& stop) {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
}

void destroy_events(cudaEvent_t& start, cudaEvent_t& stop) {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void record_start(cudaEvent_t& start, cudaStream_t stream = 0) {
    cudaEventRecord(start, stream);
}

void record_stop(cudaEvent_t& stop, cudaStream_t stream = 0) {
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
}

float elapsed_ms(cudaEvent_t& start, cudaEvent_t& stop) {
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    return ms;
}

} // namespace cuda
} // namespace floptic

#endif // FLOPTIC_HAS_CUDA
