#pragma once
#include <pybind11/pybind11.h>

namespace gpaw
{

void bind_gpu_spline_tests(pybind11::module m);

}
