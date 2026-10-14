#include "device_arrays/device_array.hpp"
#include "device_arrays/pybind_cupy_type_caster.hpp"
#include <pybind11/stl.h>
#include "libxc_wrapper.h"
#include <xc.h>
#include <cstdint>
#include <string>
#include <optional>

namespace py = pybind11;
using gpaw::PyDeviceArray;

/** Evaluates the energy density and its first derivative for an lxcXCFunctionalObject on GPU.
* The non-const arrays are all output arguments. num_points should be the (domain-local) grid size,
* ie. size of the second dimension in arrays like rho_sg.
*/
static void evaluate_lxcfunctional_device(
    const lxcXCFunctionalObject& xc,
    int32_t functional_index,
    size_t num_points,
    const PyDeviceArray& n_sg,                              // density
    PyDeviceArray& e_g,                                     // energy
    PyDeviceArray& dedn_sg,                                 // derivative: de/dn
    const std::optional<PyDeviceArray>& sigma_xg,
    std::optional<PyDeviceArray>& dedsigma_xg,              // derivative: de/dsigma
    const std::optional<PyDeviceArray>& tau_sg,
    std::optional<PyDeviceArray>& dedtau_sg)                // derivative: de/dtau
{
    if (!xc.on_device)
    {
        throw std::runtime_error("Libxc functional has not been initialized for GPU arrays");
    }
    if (functional_index < 0 || functional_index >= 2)
    {
        throw std::invalid_argument(
            std::string("Libxc functional index out of range (should be 0 or 1): ") + std::to_string(functional_index));
    }

    const auto functional = xc.functional[functional_index];

    // Passing NULL for output derivatives is OK if they are not needed

    switch (functional->info->family)
    {
    case XC_FAMILY_LDA:
#ifdef XC_FAMILY_HYB_LDA
    case XC_FAMILY_HYB_LDA:
#endif
        xc_lda_exc_vxc(functional, num_points, n_sg.data<double>(), e_g.data<double>(), dedn_sg.data<double>());
        break;
    case XC_FAMILY_GGA:
    case XC_FAMILY_HYB_GGA:
    {
        if (!sigma_xg.has_value())
        {
            throw std::runtime_error("sigma_xg must be provided for this functional");
        }

        const double* sigma_xg_ptr = sigma_xg.value().data<double>();
        double* dedsigma_xg_ptr = dedsigma_xg.has_value() ? dedsigma_xg.value().data<double>() : nullptr;

        xc_gga_exc_vxc(functional, num_points, n_sg.data<double>(), sigma_xg_ptr, e_g.data<double>(),
            dedn_sg.data<double>(), dedsigma_xg_ptr);
        break;
    }
    case XC_FAMILY_MGGA:
    case XC_FAMILY_HYB_MGGA:
    {
        // We don't support any functionals that need Laplacians. Passing null for these is OK according to libxc docs
        const double* lapl = nullptr;
        double* lapl_derivative = nullptr;
        // More explicit check is on Python side
        assert((functional->info->flags & XC_FLAGS_NEEDS_LAPLACIAN) == 0);

        if (!sigma_xg.has_value() || !tau_sg.has_value())
        {
            throw std::runtime_error("sigma_xg and tau_sg must be provided for this functional");
        }

        const double* sigma_xg_ptr = sigma_xg.value().data<double>();
        double* dedsigma_xg_ptr = dedsigma_xg.has_value() ? dedsigma_xg.value().data<double>() : nullptr;
        const double* tau_sg_ptr = tau_sg.value().data<double>();
        double* dedtau_sg_ptr = dedtau_sg.has_value() ? dedtau_sg.value().data<double>() : nullptr;

        xc_mgga_exc_vxc(functional, num_points, n_sg.data<double>(), sigma_xg_ptr, lapl, tau_sg_ptr,
            e_g.data<double>(), dedn_sg.data<double>(), dedsigma_xg_ptr,
            lapl_derivative, dedtau_sg_ptr);
        break;
    }
    default:
        // No built-in conversion from a C++ exception to NotImplementedError. Raise it manually
        const std::string msg = "Unsupported Libxc functional family: " + std::to_string(functional->info->family);
        py::set_error(PyExc_NotImplementedError, msg.c_str());
        throw py::error_already_set();
    }
}

void bind_libxc_extras(PyObject *module)
{
    if (!module || module == Py_None)
    {
        throw std::runtime_error("Invalid module");
    }

    py::module_ m = py::reinterpret_borrow<py::module_>(module);
    if (m.is_none())
    {
        throw std::runtime_error("Invalid module");
    }

    m.def("libxc_has_gpu_support", &libxc_has_gpu_support,
        py::doc("Returns True if the libxc installation supports GPUs"));

    m.def("evaluate_lxcfunctional_device", &evaluate_lxcfunctional_device,
        py::arg("lxc_functional"), py::arg("functional_index"), py::arg("num_points"),
        py::arg("n_sg"), py::arg("e_g"), py::arg("dedn_sg"), py::arg("sigma_xg"),
        py::arg("dedsigma_xg"), py::arg("tau_sg"), py::arg("dedtau_sg"),
        py::doc(
            "Evaluates the energy density and its first derivative for a Libxc functional on GPU. "
            "All input/output arrays must be float64-valued and the Libxc functional object must have been initialized on the GPU."));

}
