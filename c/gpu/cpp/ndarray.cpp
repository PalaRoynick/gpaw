#include "ndarray.hpp"

namespace gpaw::detail
{

ArrayDesc::ArrayDesc(py::handle handle)
{
    PyObject* obj = handle.ptr();
    assert(obj);

    if (is_numpy_array(obj))
    {
        py::array arr = py::reinterpret_borrow<py::array>(handle);
        data = arr.mutable_data();
        typenum = arr.dtype().num();
        ndim = static_cast<int32_t>(arr.ndim());

        shape.resize(ndim);
        strides.resize(ndim);
        for (int32_t i = 0; i < ndim; ++i)
        {
            shape[i] = arr.shape(i);
            strides[i] = arr.strides(i);
        }
    }
    else if (is_cupy_array(obj))
    {
        // dtype
        py::handle array = handle;
        pybind11::object dtype_obj = array.attr("dtype");
        pybind11::object num_obj = dtype_obj.attr("num");
        typenum = num_obj.cast<int>();

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

        ndim = py_shape.size();

        assert(py_shape.size() == py_strides.size());
        shape.resize(py_shape.size());
        strides.resize(py_strides.size());

        for (size_t i = 0; i < shape.size(); ++i)
        {
            shape[i] = py_shape[i].cast<int64_t>();
            strides[i] = py_strides[i].cast<int64_t>();
        }
    }
    else
    {
        throw std::runtime_error("Not Numpy nor Cupy array?!");
    }
}

}
