#include "splines_python.hpp"
#include "radial_spline.hpp"
#include "spline.h"  // Legacy SplineType
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace gpaw
{

template<typename T>
void bind_spline(py::module_& m, const std::string& class_name)
{
    using Spline = gpaw::TRadialSpline<T>;

    py::class_<Spline> cls(m, class_name.c_str());

    // Bind constructor directly to the above factory function
    // See https://pybind11.readthedocs.io/en/stable/advanced/classes.html#custom-constructors
    cls.def(py::init(&spline_from_numpy<T>),
        py::arg("r_max"), py::arg("f_r"),
        R"(Create a new spline from input data. Last value in the input array should correspond to spline value at r = r_max and MUST equal 0.0.)"
    );

    cls.def("get_x_start", &Spline::get_x_start);
    cls.def("get_x_end", &Spline::get_x_end);
    cls.def("get_num_bins", &Spline::get_num_bins);

    // Bind vectorized form of evaluate() with Numpy input/output.
    auto vector_evaluate = [](const Spline& self, py::array_t<T> x)
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
        };

    cls.def("evaluate",
        vector_evaluate,
        py::arg("x"),
        R"doc(Vectorized evaluation of the spline. Output array shape will match that of the input array.
        Note that this may create a copy of the input array if its dtype does not match dtype of the spline.
        )doc"
    );

    cls.def("__call__",
        vector_evaluate,
        py::arg("x"),
        R"doc(Equivalent to evaluate(x).
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
        R"(Returns spline breakpoints {x_i}.)"
    );

    cls.doc() = R"(Implements cubic spline interpolation for radial functions, uniform spacing.
    The spline is a C2-smooth piecewise polynomial
        S(x) = S_i(x) for x_i < x <= x_{i+1},
    such that
        S_i(x) = a_i + b_i (x - x_i) + c_i (x - x_i)^2 + d_i (x - x_i)^3.
    This is intended for radial functions only, ie. x is in range [0, x_max].
    Boundary conditions are hardcoded:
       - "Clamped" boundaries, ie. first derivatives at endpoints are zero. S'(0) = S'(x_max) = 0.
       - S(x) = S(0) for x <= 0
       - S(x) = 0 for x >= x_max.)";
}

bool bind_new_splines(PyObject* module)
{
    if (!module || module == Py_None)
    {
        return false;
    }

    py::module_ m = py::reinterpret_borrow<py::module_>(module);
    if (m.is_none())
    {
        return false;
    }

    // Make submodule "new" under _gpaw. Temporary until a new LFC module is introduced
    py::module_ submodule = m.def_submodule("new", "Temporary submodule for holding new non-GPU C++ bindings");
    if (!submodule)
    {
        return false;
    }

    bind_spline<float>(submodule, "Spline32");
    bind_spline<double>(submodule, "Spline64");

    return true;
}

} // namespace gpaw
