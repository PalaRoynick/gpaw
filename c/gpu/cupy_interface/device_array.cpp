#include "device_array.hpp"
#include <cassert>
#include <stdexcept>
#include <cstddef>
#include <cstdint>

namespace gpaw
{


PyDeviceArray::PyDeviceArray() noexcept
    : data(nullptr), c_contiguous(false)
{
}

PyDeviceArray::PyDeviceArray(PyObject* array)
    : PyDeviceArray(pybind11::handle(array))
{
}

PyDeviceArray::PyDeviceArray(pybind11::handle array)
{
    if (!is_cupy_array(array))
    {
        throw std::invalid_argument("Not a CuPy ndarray");
    }
    from_cupy(array);
}

void PyDeviceArray::from_cupy(pybind11::handle array)
{
    // dtype
    pybind11::object dtype_obj = array.attr("dtype");
    pybind11::object num_obj = dtype_obj.attr("num");
    dtype = pybind11::dtype(num_obj.cast<int>());

    // Read C-contiguity flag
    pybind11::object flags = array.attr("flags");
    c_contiguous = flags.attr("c_contiguous").cast<bool>();

    // Get data pointer
    data = reinterpret_cast<void*>(
        array.attr("data").attr("ptr").cast<std::uintptr_t>()
    );

    if (!data)
    {
        // This should always be a bug
        throw std::invalid_argument("Empty Cupy array passed to C++");
    }

    // This might be one extra copy?
    auto py_shape = array.attr("shape").cast<pybind11::tuple>();
    auto py_strides = array.attr("strides").cast<pybind11::tuple>();

    assert(py_shape.size() == py_strides.size());
    shape.resize(py_shape.size());
    strides.resize(py_strides.size());

    for (size_t i = 0; i < shape.size(); ++i)
    {
        shape[i] = py_shape[i].cast<int64_t>();
        strides[i] = py_strides[i].cast<int64_t>();
    }
}

bool is_cupy_array(PyObject* obj)
{
    if (!obj)
    {
        return false;
    }

    /* Fast path: check that obj is not a Numpy array and exposes dlpack interface .
    This is very naive, but will correctly identify Cupy arrays as long as we only work with Numpy/Cupy. */
    if (!PyArray_Check(obj) && PyObject_HasAttrString(obj, "__dlpack__"))
    {
        return true;
    }

    return false;
}

} // namespace gpaw
