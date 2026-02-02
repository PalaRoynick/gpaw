#pragma once

/* Partly based on ndarray template in the Nanobind library,
* https://nanobind.readthedocs.io/en/latest/ndarray.html,
* https://github.com/wjakob/nanobind (BSD-3)
*
*/

#include "python_utils.h"
#include <type_traits>
#include <array>

#include "pyarray_utils.hpp"

namespace py = pybind11;

namespace gpaw
{

namespace device
{
// Array is in CPU memory
struct cpu {};
// Array is in GPU memory
struct gpu {};
}

// C-contiguous memory layout
struct c_contig {};
// F-contiguous memory layout (Fortran style)
struct f_contig {};

/* Use to require NDArray of specific shape. Eg: NDArray<shape<2, 3>> for a 2x3 array.
* Value of -1 leaves the size of that dimension unconstrained, eg: NDArray<shape<2, -1>>
* is a 2D array with second dimension being any size.
*/
template<int64_t... Dims> struct shape {};

/* Alternative way of specifying shape when only the number of dimensions is fixed.
* Eg: NDArray<ndim<3>> is equivalent to NDArray<shape<-1, -1, -1>>.
*/
template<int32_t N> struct ndim {};


namespace detail
{

// Sentinel for representing unused traits, eg. NDArray<> has no dtype trait so value_trait is unused.
struct unused {};

// Helpers for defining array traits

template<typename T> struct is_device : std::false_type {};
template<> struct is_device<device::cpu> : std::true_type {};
template<> struct is_device<device::gpu> : std::true_type {};

template<typename T> struct is_layout : std::false_type {};
template<> struct is_layout<c_contig> : std::true_type {};
template<> struct is_layout<f_contig> : std::true_type {};
//~

// dtype filtering
template<typename, typename = void>
struct is_numpy_scalar : std::false_type {};

/* NDArray<T> should only work for types T for which a corresponding pybind dtype exists.
However, using py::dtype::num_of<T>() here doesn't work very well because it's not SFINAE-friendly.
It always tries to call py::detail::npy_format_descriptor<T>::value which only exists for valid T.
Dunno if this could be made to work with num_of<T>().
This instead relies on SFINAE on py::detail::npy_format_descriptor<T>(). */
template<typename T>
struct is_numpy_scalar<T, std::void_t<
    decltype(py::detail::npy_format_descriptor<T>::value)
>> : std::true_type {};
//~

// Shape and ndim
template<typename T> struct shape_info;

template<int64_t... Dims>
struct shape_info<shape<Dims...>>
{
    static constexpr int32_t ndim = sizeof...(Dims);
    static constexpr std::array<int64_t, ndim> dims = { Dims... };
};

// Shape constructed from ndim<N>. Sets dims to -1 (meaning unconstrained)
template<int32_t N>
struct shape_info<ndim<N>>
{
    static constexpr int32_t ndim = N;
    static constexpr std::array<int64_t, N> dims = []{
        std::array<int64_t, N> a{};
        for (auto& v : a) v = -1;
        return a;
    }();
};

template<>
struct shape_info<unused>
{
    static constexpr int32_t ndim = -1;
    static constexpr std::array<int64_t, 0> dims = {};
};

// Predicates
template<typename T> struct is_shape_like : std::false_type {};
template<int64_t... D> struct is_shape_like<shape<D...>> : std::true_type {};
template<int32_t N> struct is_shape_like<ndim<N>> : std::true_type {};

//~

// Metaprogramming selector, used to extract specific traits from Args... using predicates from above
template<typename Default, template<typename> class Pred, typename... Ts>
struct Select;

template<typename Default, template<typename> class Pred>
struct Select<Default, Pred>
{
    using type = Default;
};

template<typename Default, template<typename> class Pred, typename T, typename... Ts>
struct Select<Default, Pred, T, Ts...>
{
    using type = std::conditional_t<
        Pred<T>::value,
        T,
        typename Select<Default, Pred, Ts...>::type>;
};

/* Used for ensuring that we only select each trait once. Eg. NDArray<float, double> is not valid.
This relies on predicates being careful with SFINAE. */
template<template<typename> class Pred, typename... Ts>
constexpr size_t count_matches()
{
    return (size_t(0) + ... + size_t(Pred<Ts>::value));
}
//~

// NDArrayTraits collects static array metadata traits, eg. dtype, data layout and shape
template<typename... Args>
struct NDArrayTraits
{
    static_assert(count_matches<is_numpy_scalar, Args...>() <= 1,
        "NDArray: only one value trait is allowed. Eg: NDArray<float, double> is not valid.");
    static_assert(count_matches<is_device, Args...>() <= 1,
        "NDArray: Only one device trait is allowed. Eg: NDArray<device::cpu, device::gpu> is not valid.");
    static_assert(count_matches<is_layout, Args...>() <=1,
        "NDArray: Only one layout trait is allowed. Eg: NDArray<c_contig, f_contig> is not valid.");
    static_assert(count_matches<is_shape_like, Args...>() <= 1,
        "NDArray: only one shape<> or ndim<> trait is allowed. You cannot give both ndim<> and shape<>.");

    using value_trait = typename Select<unused, is_numpy_scalar, Args...>::type;
    using device_trait = typename Select<unused, is_device, Args...>::type;
    using layout_trait = typename Select<unused, is_layout, Args...>::type;
    using shape_trait = typename Select<unused, is_shape_like, Args...>::type;

    static constexpr bool has_dtype = !std::is_same_v<value_trait, unused>;
    static constexpr bool is_cpu = std::is_same_v<device_trait, device::cpu>;
    static constexpr bool is_gpu = std::is_same_v<device_trait, device::gpu>;

    static constexpr int32_t ndim = shape_info<shape_trait>::ndim;
    static constexpr auto shape = shape_info<shape_trait>::dims;

    /* True if we know anything at all about the shape. Knowing ndim is sufficient,
    and knowing all or some dimension sizes is also OK. */
    static constexpr bool has_shape_info = (ndim != -1);

    static constexpr bool is_contiguous = std::is_same_v<layout_trait, c_contig> || std::is_same_v<layout_trait, f_contig>;
    // 1D contiguous arrays are both c_contig and f_contig
    static constexpr bool is_c_contig = (ndim == 1) ? is_contiguous : std::is_same_v<layout_trait, c_contig>;
    static constexpr bool is_f_contig = (ndim == 1) ? is_contiguous : std::is_same_v<layout_trait, f_contig>;
};


// Stores array metadata
struct GPAW_HIDDEN_SYMBOL ArrayDesc
{
    ArrayDesc() = default;

    // Pick array metadata from Numpy or Cupy ndarray. Suboptimal, used for tests only
    ArrayDesc(py::handle handle);

    py::dtype dtype() { return py::dtype(typenum); }

    // Use signed integers for compatibility with dlpack
    void* data = nullptr;
    int typenum = 0; // raw numpy typenum
    int32_t ndim = 0;

    // For Numpy we could go for zero-copy but that seems fragile.
    // For Cupy this is harder because the shape/stride there are Python tuples (not contiguous C-style arrays)
    // So we opt for copies.
    // TODO!! Figure out a good pattern for memory management.
    // For now, just std::vector which is not optimal for this

    std::vector<int64_t> shape;
    std::vector<int64_t> strides;
    // Offset is TODO. Libraries don't really seem to use this ATM
    static constexpr uint64_t byte_offset = 0;
};

} // namespace detail

/* Class NDArray<>: generic multidimensional array that is convertible from Python ndarrays.
Use template parameters to specify requirements like datatype, shape and whether the data is in CPU or GPU memory.
Defined as a hidden symbol to avoid subtle ABI issues due to exposing pybind11::dtype, see
https://pybind11.readthedocs.io/en/stable/faq.html#someclass-declared-with-greater-visibility-than-the-type-of-its-field-someclass-member-wattributes
*/
template<typename... Args>
class GPAW_HIDDEN_SYMBOL NDArray
{
public:
    // Must be default constructible for pybind11 type caster
    NDArray() = default;

    static constexpr detail::NDArrayTraits<Args...> traits;

    // FIXME: not optimal, py::dtype(typenum) constructor is not constexpr and may throw
    py::dtype dtype() const { return py::dtype(desc.typenum); }

    size_t ndim() const { return (size_t)desc.ndim; }
    // Returns shape in the i. dimension. Does not check bounds!
    size_t shape(size_t i) const { return desc.shape[i]; }

    // Returns stride in the i. dimension. Does not check bounds!
    int64_t stride(size_t i) const { return desc.strides[i]; }

    const int64_t* shape_ptr() const { return desc.shape.data(); }
    const int64_t* stride_ptr() const { return desc.strides.data(); }
    bool is_valid() const { return handle != nullptr; }

    // Number of elements in the array
    size_t size() const
    {
        size_t res = (size_t)is_valid();
        for (size_t i = 0; i < ndim(); ++i)
        {
            res *= shape(i);
        }
        return res;
    }

    // Size of array element in bytes
    size_t itemsize() const { return (size_t) dtype().itemsize(); }
    // Total number of bytes occupied by the array
    size_t nbytes() const { return itemsize() * size(); }

private:
    py::handle handle = nullptr;
    detail::ArrayDesc desc;

    friend struct pybind11::detail::type_caster<NDArray<Args...>>;
};

namespace detail
{

enum class Framework
{
    eNumpy, eCupy
};

/* Aggressive CPython-style accessors to array attributes.
* No ownership (borrows only), no error checking, no exceptions.
* For Numpy these are trivial wrappers around Numpy C-API.
*/
struct ArrayAccessors
{
    // ===================== NumPy =====================

    static void* data(PyArrayObject* obj) noexcept
    {
        return PyArray_DATA(obj);
    }

    static int typenum(PyArrayObject* obj) noexcept
    {
        return PyArray_TYPE(obj);
    }

    static bool is_c_contig(PyArrayObject* obj) noexcept
    {
        return PyArray_IS_C_CONTIGUOUS(obj);
    }

    static bool is_f_contig(PyArrayObject* obj) noexcept
    {
        return PyArray_IS_F_CONTIGUOUS(obj);
    }

    static int32_t ndim(PyArrayObject* obj) noexcept
    {
        return static_cast<int32_t>(PyArray_NDIM(obj));
    }

    static std::vector<int64_t> shape(PyArrayObject* obj) noexcept
    {
        const int32_t num_dims = ndim(obj);
        std::vector<int64_t> out(num_dims);
        npy_intp* shape_ptr = PyArray_DIMS(obj);
        for (int32_t i = 0; i < num_dims; ++i)
        {
            out[i] = (int64_t)shape_ptr[i];
        }
        return out;
    }

    static std::vector<int64_t> strides(PyArrayObject* obj) noexcept
    {
        const int32_t num_dims = ndim(obj);
        std::vector<int64_t> out(num_dims);
        npy_intp* stride_ptr = PyArray_STRIDES(obj);
        for (int32_t i = 0; i < num_dims; ++i)
        {
            out[i] = (int64_t)stride_ptr[i];
        }
        return out;
    }


    // ===================== CuPy =====================

    static void* data(PyObject* obj) noexcept
    {
        // obj.data.ptr (device pointer)
        PyObject* data_obj = PyObject_GetAttrString(obj, "data");
        void* ptr = PyLong_AsVoidPtr(PyObject_GetAttrString(data_obj, "ptr"));
        Py_DECREF(data_obj);
        return ptr;
    }

    static int typenum(PyObject* obj) noexcept
    {
        // obj.dtype.num
        PyObject* dtype = PyObject_GetAttrString(obj, "dtype");
        int typenum = static_cast<int>(PyLong_AsLong(PyObject_GetAttrString(dtype, "num")));
        Py_DECREF(dtype);
        return typenum;
    }

    static bool is_c_contig(PyObject* obj) noexcept
    {
        // obj.flags.c_contiguous
        PyObject* flags = PyObject_GetAttrString(obj, "flags");
        bool value = PyObject_IsTrue(PyObject_GetAttrString(flags, "c_contiguous"));
        Py_DECREF(flags);
        return value;
    }

    static bool is_f_contig(PyObject* obj) noexcept
    {
        // obj.flags.f_contiguous
        PyObject* flags = PyObject_GetAttrString(obj, "flags");
        bool value = PyObject_IsTrue(PyObject_GetAttrString(flags, "f_contiguous"));
        Py_DECREF(flags);
        return value;
    }

    static int32_t ndim(PyObject* obj) noexcept
    {
        // obj.ndim
        return static_cast<int32_t>(PyLong_AsLong(PyObject_GetAttrString(obj, "ndim")));
    }

    static std::vector<int64_t> shape(PyObject* obj) noexcept
    {
        // obj.shape (tuple)
        PyObject* shape_tuple = PyObject_GetAttrString(obj, "shape");
        const Py_ssize_t num_dims = PyTuple_GET_SIZE(shape_tuple);

        std::vector<int64_t> out(num_dims);
        for (Py_ssize_t i = 0; i < num_dims; ++i)
        {
            out[i] = (int64_t)PyLong_AsLongLong(PyTuple_GET_ITEM(shape_tuple, i));
        }

        Py_DECREF(shape_tuple);
        return out;
    }

    static std::vector<int64_t> strides(PyObject* obj) noexcept
    {
        // obj.strides (tuple)
        PyObject* strides_tuple = PyObject_GetAttrString(obj, "strides");
        const Py_ssize_t num_dims = PyTuple_GET_SIZE(strides_tuple);

        std::vector<int64_t> out(num_dims);
        for (Py_ssize_t i = 0; i < num_dims; ++i)
        {
            out[i] = (int64_t)PyLong_AsLongLong(PyTuple_GET_ITEM(strides_tuple, i));
        }

        Py_DECREF(strides_tuple);
        return out;
    }
};

/* Import an array from Numpy or Cupy. Import will fail if not all requested traits are satisfied,
in which case this returns false and the `out` desc is in an invalid state.
*/
template<Framework framework, typename... Args>
inline bool ndarray_import_impl(
    PyObject* obj,
    ArrayDesc& out) noexcept
{
    using traits = NDArrayTraits<Args...>;

    // For Numpy, cast to PyArrayObject* so that we trigger faster overloads of gpaw::Array_DATA etc.
    using pyobject_ptr_t = std::conditional_t<framework == Framework::eNumpy, PyArrayObject*, PyObject*>;
    pyobject_ptr_t arr = reinterpret_cast<pyobject_ptr_t>(obj);

    if constexpr (traits::is_cpu && framework != Framework::eNumpy)
    {
        return false;
    }
    else if constexpr (traits::is_gpu && framework != Framework::eCupy)
    {
        return false;
    }

    out.data = ArrayAccessors::data(arr);

    out.typenum = ArrayAccessors::typenum(arr);
    if constexpr (traits::has_dtype)
    {
        if (out.typenum != py::dtype::num_of<typename traits::value_trait>())
        {
            return false;
        }
    }

    if constexpr (traits::is_c_contig)
    {
        if (!ArrayAccessors::is_c_contig(arr))
        {
            return false;
        }
    }
    if constexpr (traits::is_f_contig)
    {
        if (!ArrayAccessors::is_f_contig(arr))
        {
            return false;
        }
    }

    out.ndim = static_cast<int32_t>(ArrayAccessors::ndim(arr));
    if constexpr (traits::ndim != -1)
    {
        if (out.ndim != traits::ndim)
        {
            return false;
        }
    }

    out.shape = ArrayAccessors::shape(arr);
    if constexpr (traits::has_shape_info)
    {
        for (int32_t i = 0; i < out.ndim; ++i)
        {
            if (traits::shape[i] != -1 && out.shape[i] != traits::shape[i])
            {
                return false;
            }
        }
    }

    out.strides = ArrayAccessors::strides(arr);

    return true;
}

template<typename... Args>
inline bool ndarray_import(PyObject* obj, ArrayDesc& out) noexcept
{
    bool success = true;
    if (is_numpy_array(obj))
    {
        success &= ndarray_import_impl<Framework::eNumpy, Args...>(obj, out);
    }
    else if (is_cupy_array(obj))
    {
        success &= ndarray_import_impl<Framework::eCupy, Args...>(obj, out);
    }

    if (!success)
    {
        return false;
    }
    else if (PyErr_Occurred())
    {
        // Got CPython error that is not due to mismatched array traits. Either we miscategorized the PyObject (not Numpy nor Cupy)
        // or there is a bug in ArrayAccessors, or the error actually occurred much earlier and whoever caused it forgot to handle and clear it.
        fprintf(stderr, "Unhandled Python exception occurred during gpaw::NDArray type cast. This could happen for non-Numpy or non-Cupy arrays, or could imply a bug in GPAW. This error is unrecoverable!\n");
        PyErr_PrintEx(0);
        std::terminate();
    }

    return success;
}

} // namespace detail
} // namespace gpaw


// Pybind11 type casters
namespace pybind11::detail
{

// Try to follow nanobind semantics here so that we have easier time porting to it later if needed.
// https://nanobind.readthedocs.io/en/latest/porting.html#type-casters


template<typename... Args>
struct type_caster<gpaw::NDArray<Args...>>
{
    // TODO build a better descriptor like "NDArray<float, shape<-1>, c_contig>").
    // Ideally it'd be a compile-time string (required by pybind11)
    // but that is very difficult to construct in C++17 without extensive metaprogramming.
    // We could build it at runtime if the array conversion fails, but that doesn't show up in docstrings then

    PYBIND11_TYPE_CASTER(gpaw::NDArray<Args...>, _("NDArray<>"));

    /* C++ -> Python conversion. FIXME: can't easily have this as Cupy provides no C++ interface.
    * So either we call Cupy's Python API from here, or simply don't allow conversion like this.
    * For now, don't allow it. */
    static handle cast(const gpaw::NDArray<Args...>& array, return_value_policy policy, handle parent) noexcept
    {
        assert(false && "Passing gpaw::NDArray back to Python is not implemented");
        return handle();
    }

    /* Python -> C++ conversion. Must return false on failure, which raises a TypeError on Python side.
    */
    bool load(handle src, bool implicit_convert) noexcept
    {
        PyObject* obj = src.ptr();
        assert(obj);

        if (!gpaw::detail::ndarray_import<Args...>(obj, value.desc))
        {
            return false;
        }

        value.handle = src;
        return true;
    }
};

} // namespace pybind11::detail
