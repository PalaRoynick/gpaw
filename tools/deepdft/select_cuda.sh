#!/usr/bin/env bash
# Discover a site toolkit before private toolchain activation changes PATH.

deepdft_nvcc_for_root() {
    local root=$1 sdk
    if [[ -x $root/bin/nvcc ]]; then printf '%s\n' "$root/bin/nvcc"; return; fi
    sdk=$(dirname -- "$root")
    if [[ $(basename -- "$sdk") == cuda ]]; then sdk=$(dirname -- "$sdk"); fi
    if [[ -x $sdk/compilers/bin/nvcc ]]; then
        printf '%s\n' "$sdk/compilers/bin/nvcc"
    else
        echo "No nvcc found for CUDA root: $root" >&2
        return 2
    fi
}

deepdft_cuda_has_development_files() {
    local root=$1 lib
    deepdft_nvcc_for_root "$root" >/dev/null 2>&1 || return 1
    [[ -f $root/include/cuda_runtime.h ||
       -f $root/targets/x86_64-linux/include/cuda_runtime.h ]] || return 1
    for lib in "$root/lib64" "$root/lib" "$root/targets/x86_64-linux/lib"; do
        if [[ -f $lib/libcudart.so ]]; then return 0; fi
    done
    return 1
}

deepdft_select_cuda() {
    local requested=${1-} nvcc resolved root sdk candidate release
    local -a candidates=()
    if [[ -n $requested ]]; then
        requested=$(cd -- "$requested" && pwd) || return
        nvcc=$(deepdft_nvcc_for_root "$requested") || return
    else
        nvcc=$(command -v nvcc) || return 0
    fi
    if [[ -n $nvcc ]]; then
        resolved=$(readlink -f -- "$nvcc") || return
        root=$(dirname -- "$(dirname -- "$resolved")")
        sdk=$(dirname -- "$root")
        # HPC SDK may expose a wrapper under compilers/bin, not cuda/bin.
        release=$("$nvcc" --version | sed -nE 's/.*release ([0-9]+\.[0-9]+),.*/\1/p') || return
        candidates=("$root" "$sdk/cuda/$release" "$sdk/cuda")
        for candidate in "${candidates[@]}"; do
            if deepdft_cuda_has_development_files "$candidate"; then
                (cd -- "$candidate" && pwd)
                return
            fi
        done
        echo 'nvcc was found, but its CUDA development root could not be located.' >&2
        echo 'Pass --cuda-root with bin/nvcc, CUDA headers and libcudart; cuBLAS is checked separately.' >&2
        return 2
    fi
}

deepdft_check_cuda_version() {
    local cuda=$1 version nvcc
    nvcc=$(deepdft_nvcc_for_root "$cuda") || return
    version=$("$nvcc" --version) || return
    printf '%s\n' "$version"
    if ! grep -Eq 'release 12\.[6-9],' <<< "$version"; then
        echo 'Use CUDA 12.6-12.9 for this V100/GCC 13 profile. CUDA 13 cannot target sm_70.' >&2
        echo 'Select a compatible site toolkit with --cuda-root or request --download-cuda.' >&2
        return 2
    fi
}
