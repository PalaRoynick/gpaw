#include "pw.h"
#include "python_utils.h"

#include "gpu/cpp/pyarray_utils.hpp"
#include "gpu/cpp/pybind_cupy_type_caster.hpp"

namespace py = pybind11;

static void _pw_insert(int nG,
                int nQ,
                double_complex* c_G,
                npy_int32* Q_G,
                double scale,
                double_complex* tmp_Q)
// Does the same as these two lines of Python:
//
//     tmp_Q[:] = 0.0
//     tmp_Q.ravel()[Q_G] = c_G * scale
{
    int Q1 = 0;
    for (int G = 0; G < nG; G++) {
        int Q2 = Q_G[G];
        for (; Q1 < Q2; Q1++)
            tmp_Q[Q1] = 0.0;
        tmp_Q[Q1++] = c_G[G] * scale;
        }
    for (; Q1 < nQ; Q1++)
        tmp_Q[Q1] = 0.0;
}


void cpp_pw_insert(py::array c_G,
                   py::array Q_G,
                   double scale,
                   py::array tmp_Q)
{

//    nG = 
//    _pw_insert(nG, nQ, c_G, Q_G, scale, tmp_Q)
}

bool bind_pw_submodule(PyObject* module)
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

    py::module_ submodule = m.def_submodule("pw", "PW specific C++ bindings for GPAW. (EXPERIMENTAL)");

    // Bind Cupy conversion testers (call from Python-side test)

    submodule.def("pw_insert",
        &cpp_pw_insert,
        py::arg("c_G"), py::arg("Q_G"), py::arg("scale"), py::arg("tmp_Q"),
        R"(Docstring for pw_insert.)"
    );

    return true;
}
