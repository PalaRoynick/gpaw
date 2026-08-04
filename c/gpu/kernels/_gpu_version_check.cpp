// Compiled by setup.py before the main build to verify that GPU compiler
// and architecture flags meet our requirements.

static_assert(__cplusplus >= 201703L, "GPAW GPU extension requires C++17 or newer");

#ifdef __NVCC__
    __global__ void _gpaw_arch_check()
    {
        // __CUDA_ARCH__ is only defined during the device compilation pass
    #ifdef __CUDA_ARCH__
        static_assert(__CUDA_ARCH__ >= 700, "GPAW GPU extension for CUDA requires sm_70 or newer");
    #endif
    }

#else
    #include <hip/hip_version.h>
    static_assert(
        HIP_VERSION_MAJOR > 5 ||
        (HIP_VERSION_MAJOR == 5 && HIP_VERSION_MINOR >= 0),
        "GPAW GPU extension for AMD requires ROCm 5.0 or newer");
#endif
