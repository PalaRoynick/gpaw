#pragma once

#include "python_utils.h"
#include <pybind11/numpy.h>

#include "spline.hpp"
#include "spline.h"
#include <string>
#include <algorithm>

namespace py = pybind11;

namespace gpaw
{

template<typename T>
void bind_spline(py::module_& m, const std::string& class_name, const std::string& class_docs)
{
    using Spline = gpaw::TSpline<T>;

    py::class_<Spline> cls(m, class_name.c_str());

    // Bind constructor via helper lambda (copy Numpy array -> std::vector).
    // This forces a copy to C-contiguous array if the input is not such.
    cls.def(py::init(
        [](T x_start, T x_end, py::array_t<T, py::array::c_style | py::array::forcecast> data)
        {
            py::buffer_info buf = data.request();
            if (buf.ndim != 1)
            {
                throw std::runtime_error("Input data must be 1D");
            }

            T* data_ptr = static_cast<T*>(buf.ptr);
            std::vector<T> data_copy(data_ptr, data_ptr + buf.size);
            return Spline(x_start, x_end, std::move(data_copy));
        })
    );

    cls.def("get_x_start", &Spline::get_x_start);
    cls.def("get_x_end", &Spline::get_x_end);
    cls.def("get_num_bins", &Spline::get_num_bins);

    // Bind vectorized form of evaluate() with Numpy input/output.
    cls.def("evaluate",
        [](const Spline& self, py::array_t<T> x)
        {
            py::buffer_info x_buf = x.request();

            // Let Numpy allocate an output array of same shape and strides
            py::array_t<T> result(x_buf.shape, x_buf.strides);
            py::buffer_info r_buf = result.request();

            const T* x_ptr = static_cast<const T*>(x_buf.ptr);
            T* r_ptr = static_cast<T*>(r_buf.ptr);

            size_t num_elements = 1;
            for (size_t dim : x_buf.shape)
            {
                num_elements *= dim;
            }

            for (size_t i = 0; i < num_elements; ++i)
            {
                r_ptr[i] = self.evaluate(x_ptr[i]);
            }
            return result;
        },
        py::arg("x"),
        R"doc(Vectorized evaluation of the spline. Output array shape will match that of the input array.
        )doc"
    );

    cls.def_static(
        "from_legacy_spline",
        [](py::object py_spline) -> Spline
        {
            if (!PyObject_TypeCheck(py_spline.ptr(), &SplineType))
            {
                throw std::runtime_error("Expected a legacy SplineObject instance (not the pure-Python Spline wrapper)");
            }

            SplineObject* spline_obj = reinterpret_cast<SplineObject*>(py_spline.ptr());
            const bmgsspline& legacy = spline_obj->spline;

            return Spline::from_legacy_spline(legacy);
        },
        py::arg("spline"),
        R"(Construct new Spline object from a legacy GPAW Spline object.)"
    );

    cls.def("get_breakpoints",
        [](const Spline& self)
        {
            const std::vector<T> x = self.get_breakpoints();
            py::array_t<T> result(x.size());

            std::copy_n(x.data(), x.size(), result.mutable_data());
            return result;
        },
        R"(Returns {x_i}.)"
    );

    cls.doc() = class_docs.c_str();
}

} // namespace gpaw
