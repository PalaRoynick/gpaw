#!/usr/bin/env bash
# Helpers for the private Conda toolchain; source this file from Bash.
# Preserve CUDA driver paths, GPU visibility and scheduler allocation variables.

deepdft_filter_mpi_paths() {
    local name=$1 value entry result=
    local -a entries=()
    value=${!name-}
    # LD_PRELOAD accepts both spaces and colons as separators.
    if [[ $name == LD_PRELOAD ]]; then value=${value// /:}; fi
    IFS=: read -r -a entries <<< "$value"
    for entry in "${entries[@]}"; do
        [[ -n $entry ]] || continue
        case "${entry,,}" in
            *hpcx*|*/comm_libs/*|*/openmpi/*|*/ompi/*|*/mpi/*|*/pmix/*|*/prrte/*|*/ucx/*)
                continue ;;
        esac
        result=${result:+$result:}$entry
    done
    if [[ -n $result ]]; then export "$name=$result"; else unset "$name"; fi
}

deepdft_clean_host_toolchain() {
    local name
    # The Conda activation that follows supplies its own compiler configuration.
    for name in CC CXX CPP FC F77 F90 LD LDSHARED CFLAGS CXXFLAGS CPPFLAGS \
        FFLAGS FCFLAGS LDFLAGS LD_RUN_PATH LIBRARY_PATH CPATH C_INCLUDE_PATH \
        CPLUS_INCLUDE_PATH OBJC_INCLUDE_PATH PKG_CONFIG_PATH CMAKE_PREFIX_PATH \
        PYTHONPATH PYTHONHOME; do
        unset "$name"
    done
    while IFS= read -r name; do
        case "$name" in
            OMPI_*|OPAL_*|PMIX_*|PRTE_*|HPCX_*|I_MPI_*|MPI_HOME|MPI_ROOT|UCX_*)
                unset "$name" ;;
        esac
    done < <(compgen -v)
    for name in PATH LD_LIBRARY_PATH LD_PRELOAD; do
        deepdft_filter_mpi_paths "$name"
    done
}

deepdft_prioritize_prefix() {
    local prefix=$1 name entry result
    local -a entries=()
    for name in PATH LD_LIBRARY_PATH LIBRARY_PATH; do
        result=
        IFS=: read -r -a entries <<< "${!name-}"
        for entry in "${entries[@]}"; do
            [[ -n $entry && $entry != "$prefix/bin" && $entry != "$prefix/lib" ]] || continue
            result=${result:+$result:}$entry
        done
        if [[ $name == PATH ]]; then
            export "$name=$prefix/bin${result:+:$result}"
        else
            export "$name=$prefix/lib${result:+:$result}"
        fi
    done
}
