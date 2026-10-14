#pragma once

#include "python_utils.h"
#include "gpaw_utils.h"
#include "pyarray_utils.hpp"

namespace gpaw
{

/* Checks if the input object is a Cupy array (cupy.ndarray).
Empty Cupy array is considered valid.
NOTE: Not quite foolproof: we assume that only Numpy and Cupy arrays are passed around
so this will return true for anything that looks like an ndarray but is NOT a Numpy array. */
bool is_cupy_array(PyObject* obj);

// See the `is_cupy_array(PyObject*)` overload for docstring
inline bool is_cupy_array(pybind11::handle obj) { return is_cupy_array(obj.ptr()); }

/* Cheap, type-erased wrapper around a Python-side GPU array (cupy.ndarray).
Does NOT take ownership or copy the data.
Defined as a hidden symbol to avoid subtle ABI issues due to exposing pybind11::dtype, see
https://pybind11.readthedocs.io/en/stable/faq.html#someclass-declared-with-greater-visibility-than-the-type-of-its-field-someclass-member-wattributes
*/
struct GPAW_HIDDEN_SYMBOL PyDeviceArray
{
    // Allow default construction for pybind11 type casting
    PyDeviceArray() noexcept;
    PyDeviceArray(PyObject* array);
    PyDeviceArray(pybind11::handle array);

    // Number of array dimensions
    size_t ndim() const { return shape.size(); }
    bool is_c_contiguous() const { return c_contiguous; }

    /** Access the data pointer as specified type. Throws a runtime_error if the dtype does not match. */
    template <typename T>
    T* data() { return data_impl<T>(); }

    /** Access the data pointer as specified type. Throws a runtime_error if the dtype does not match */
    template <typename T>
    const T* data() const { return data_impl<T>(); }

    void* data_ptr = nullptr;
    pybind11::dtype dtype;
    bool c_contiguous;
    // Use int64_t for shape/strides for compatibility with standards like dlpack
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;

private:
    void from_cupy(pybind11::handle array);
    void from_cupy(PyObject* obj) { from_cupy(pybind11::handle(obj)); }

    template <typename T>
    T* data_impl() const
    {
        using U = std::remove_cv_t<T>;
        if constexpr (!std::is_void_v<U>)
        {
            if (!dtype.is(pybind11::dtype::of<U>()))
            {
                const std::string requested_dtype_name = pybind11::dtype::of<U>().str();
                throw std::runtime_error(
                    "Trying to access data in a device array (dtype = " +
                    std::string(dtype.str()) +
                    ") as non-matching type " + requested_dtype_name);
            }
        }
        return static_cast<T*>(data_ptr);
    }

    // Let pybind11 type caster access private members
    friend class pybind11::detail::type_caster<gpaw::PyDeviceArray>;
};

} // namespace gpaw
