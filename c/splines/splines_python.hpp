#pragma once
#include "radial_spline.hpp"
#include "pyarray_utils.hpp"
#include <stdexcept>

namespace gpaw
{

template<typename T>
static TRadialSpline<T> spline_from_numpy(T r_max, pybind11::array_t<T, pybind11::array::c_style | pybind11::array::forcecast> data)
{
    pybind11::buffer_info buf = data.request();
    if (buf.ndim != 1)
    {
        throw std::runtime_error("Input data must be 1D");
    }

    T* data_ptr = static_cast<T*>(buf.ptr);
    return TRadialSpline<T>(r_max, data_ptr, buf.size);
}

bool bind_new_splines(PyObject* module);

} // namespace gpaw
