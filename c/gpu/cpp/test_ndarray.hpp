#pragma once

#include "python_utils.h"
#include <pybind11/pybind11.h>
#include "gpu/gpu_utils.h"


GPAW_GPU_LINKAGE void bind_ndarray_tests(pybind11::module_ parent_module);
