#pragma once

/* Partly based on ndarray template in the Nanobind library,
* https://nanobind.readthedocs.io/en/latest/ndarray.html,
* https://github.com/wjakob/nanobind (BSD-3)
*
*/

#include "python_utils.h"
#include <type_traits>

#include "pyarray_utils.hpp"

// // Lightweight struct for describing what kind of array we are expecting to import from Python side
// struct ndarray_request
// {
//     bool is_gpu = false;
//     bool c_contig = false;
//     int typenum = 0;
//     int32_t ndim = 0;
//     int64_t* shape = nullptr;
//     ndarray_request() = default;

//     template<typename Traits> ndarray_request() noexcept
//     {
//         is_gpu = Traits::is_gpu;
//         c_contig = Traits::is_c_contig;

//         if constexpr (Traits::has_dtype)
//         {
//             typenum = py::dtype::num_of<typename Traits::value_type>();
//         }
//         else
//         {
//             typenum = 0;
//         }

//         // TODO
//         ndim = 0;
//         shape = nullptr;
//     }
// };

namespace py = pybind11;

namespace gpaw
{

namespace device
{
struct cpu {};
struct gpu {};
}

struct c_contig {};
struct f_contig {};

namespace detail
{

// Sentinel for representing unused traits, eg. NDArray<> has no dtype trait so value_type is unused.
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

// Metaprogramming selector, used to "parse" traits from Args...
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

// Used for ensuring that we only select each trait once. Eg. NDArray<float, double> is not valid.
template<template<typename> class Pred, typename... Ts>
constexpr size_t count_matches()
{
    return (size_t(0) + ... + size_t(Pred<Ts>::value));
}
//~

// NDArrayTraits collects array metadata traits, eg. dtype, data layout and shape
template<typename... Args>
struct NDArrayTraits
{
    static_assert(count_matches<is_numpy_scalar, Args...>() <= 1,
        "NDArray: only one value_type is allowed. Eg: NDArray<float, double> is not valid.");
    static_assert(count_matches<is_device, Args...>() <= 1,
        "NDArray: Only one device_type is allowed. Eg: NDArray<device::cpu, device::gpu> is not valid.");
    static_assert(count_matches<is_layout, Args...>() <=1,
        "NDArray: Only one layout specifier is allowed. Eg: NDArray<c_contig, f_contig> is not valid.");

    using value_type = typename Select<unused, is_numpy_scalar, Args...>::type;
    using device_type = typename Select<device::cpu, is_device, Args...>::type;
    using layout_type = typename Select<unused, is_layout, Args...>::type;

    static constexpr bool has_dtype = !std::is_same_v<value_type, unused>;
    static constexpr bool is_gpu = std::is_same_v<device_type, device::gpu>;
    static constexpr bool is_c_contig = std::is_same_v<layout_type, c_contig>;
};


// Stores array metadata
struct GPAW_HIDDEN_SYMBOL ArrayDesc
{
    // Use signed integers for compatibility with dlpack
    void* data = nullptr;
    int typenum = 0; // raw numpy typenum
    int32_t ndim = 0;
    int64_t* shape = nullptr;
    int64_t* strides = nullptr;
    // Offset is TODO. Libraries don't really seem to use this ATM
    static constexpr uint64_t byte_offset = 0;

private:
    // These use raw CPython API since pybind11 funcs are not necessarily noexcept
    // FIXME redundant, could remove
    void from_numpy(PyArrayObject* arr) noexcept;
    void from_cupy(PyObject* obj) noexcept;
};

} // namespace detail

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
    const int64_t* shape_ptr() const { return desc.shape; }
    const int64_t* stride_ptr() const { return desc.strides; }
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

template<typename... Args>
inline bool ndarray_import_numpy(
    PyArrayObject* obj,
    ArrayDesc& out) noexcept
{
    using traits = NDArrayTraits<Args...>;

    if constexpr (traits::is_gpu)
    {
        return false;
    }

    out.data = PyArray_DATA(obj);

    out.typenum = PyArray_TYPE(obj);
    if constexpr (traits::has_dtype)
    {
        if (out.typenum != py::dtype::num_of<typename traits::value_type>())
        {
            return false;
        }
    }

    if constexpr (traits::is_c_contig)
    {
        if (!PyArray_IS_C_CONTIGUOUS(obj))
        {
            return false;
        }
    }

    out.ndim = static_cast<int32_t>(PyArray_NDIM(obj));
    // TODO constexpr check ndim and shape
    // CAUTION: we're getting memory owned by numpy
    out.shape = PyArray_DIMS(obj);
    out.strides = PyArray_STRIDES(obj);

    return true;
}

template<typename... Args>
inline bool ndarray_import(PyObject* obj, ArrayDesc& out) noexcept
{
    if (is_numpy_array(obj))
    {
        return ndarray_import_numpy<Args...>(reinterpret_cast<PyArrayObject*>(obj), out);
    }
    else if (is_cupy_array(obj))
    {
        return false; // TODO!
    }
    return false;
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
    using ndarray = gpaw::NDArray<Args...>;
    using Traits = gpaw::detail::NDArrayTraits<Args...>;

    // TODO different text based on template args
    PYBIND11_TYPE_CASTER(gpaw::NDArray<Args...>, _("ndarray"));

    /* C++ -> Python conversion. FIXME: can't easily have this as Cupy provides no C++ interface.
    * So either we call Cupy's Python API from here, or simply don't allow conversion like this.
    * For not, don't allow it. */
    static handle cast(const gpaw::NDArray<Args...>& array, return_value_policy policy, handle parent) noexcept
    {
        assert(false && "WIP");
        return handle();
    }

    /* Python -> C++ conversion. Must return false on failure, which raises a TypeError on Python side.
    */
    bool load(handle src, bool implicit_convert) noexcept
    {
        PyObject* obj = src.ptr();
        assert(obj);

        static constexpr Traits traits;

        if (!gpaw::detail::ndarray_import<Args...>(obj, value.desc))
        {
            return false;
        }

        value.handle = src;
        return true;
    }
};

} // namespace pybind11::detail
