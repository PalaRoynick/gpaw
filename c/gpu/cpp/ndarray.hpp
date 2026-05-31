#pragma once

/* Defines NDArray<> template class with optional constraints on allowed array types.
* Inspired by a similar class in the nanobind library,
* https://nanobind.readthedocs.io/en/latest/ndarray.html,
* https://github.com/wjakob/nanobind (BSD-3).
* Ours is a less polished implementation but we try to follow their conventions for using the class.
*/

#include "python_utils.h"
#include "compile_time_strings.hpp"
#include <type_traits>
#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>

#include "pyarray_utils.hpp"
#include "array_accessors.hpp"

namespace py = pybind11;

namespace gpaw
{

template<typename T, size_t N>
class GPAW_HIDDEN_SYMBOL IndexArray
{
public:
    using value_type = T;
    static constexpr size_t size = N;

    constexpr IndexArray() = default;
    constexpr explicit IndexArray(const std::array<T, N>& values) : values_(values) {}

    constexpr T& operator[](size_t i) { return values_[i]; }
    constexpr const T& operator[](size_t i) const { return values_[i]; }

    constexpr const std::array<T, N>& values() const { return values_; }

private:
    std::array<T, N> values_{};
};

template<typename A, typename B, size_t N>
constexpr auto operator+(const IndexArray<A, N>& a, const IndexArray<B, N>& b)
{
    using R = std::common_type_t<A, B>;
    std::array<R, N> out{};
    for (size_t i = 0; i < N; i++)
    {
        out[i] = static_cast<R>(a[i]) + static_cast<R>(b[i]);
    }
    return IndexArray<R, N>(out);
}

template<typename A, typename B, size_t N>
constexpr auto operator-(const IndexArray<A, N>& a, const IndexArray<B, N>& b)
{
    using R = std::common_type_t<A, B>;
    std::array<R, N> out{};
    for (size_t i = 0; i < N; i++)
    {
        out[i] = static_cast<R>(a[i]) - static_cast<R>(b[i]);
    }
    return IndexArray<R, N>(out);
}

template<typename A, typename B, size_t N>
constexpr auto operator*(const IndexArray<A, N>& a, const IndexArray<B, N>& b)
{
    using R = std::common_type_t<A, B>;
    std::array<R, N> out{};
    for (size_t i = 0; i < N; i++)
    {
        out[i] = static_cast<R>(a[i]) * static_cast<R>(b[i]);
    }
    return IndexArray<R, N>(out);
}

template<typename A, typename B, size_t N>
constexpr auto dot(const IndexArray<A, N>& a, const IndexArray<B, N>& b)
{
    using R = std::common_type_t<A, B>;
    R out = 0;
    for (size_t i = 0; i < N; i++)
    {
        out += static_cast<R>(a[i]) * static_cast<R>(b[i]);
    }
    return out;
}


// Device trait for NDArrays
namespace device
{
// Array is in CPU memory
struct cpu { static constexpr FixedString name = "CPU"; };
// Array is in GPU memory
struct gpu { static constexpr FixedString name = "GPU"; };
}
/* NB: With C++20 concepts we could make device templating more convenient. We could eg. have
    enum class Device {cpu, gpu};
    template<typename T> concept DeviceTrait = is_device<T>::value;

Which would allow a function such as
    template<DeviceTrait device> void somefunc(NDArray<device>& arr);

Overall many things here could be cleaner with concepts.
*/


// C-contiguous memory layout
struct c_contig { static constexpr FixedString name = "order=C"; };
// F-contiguous memory layout (Fortran style)
struct f_contig { static constexpr FixedString name = "order=F"; };

/* Use to require NDArray of specific shape. Eg: NDArray<shape<2, 3>> for a 2x3 array.
* Value of -1 leaves the size of that dimension unconstrained, eg: NDArray<shape<2, -1>>
* is a 2D array with second dimension being any size.
*/
template<int64_t... Dims>
struct shape
{
    static_assert(((Dims >= 0 || Dims == -1) && ...), "Shape arguments must be positive or -1.");

    static constexpr int32_t ndim = sizeof...(Dims);
    static constexpr std::array<int64_t, ndim> dims = { Dims... };

    // FixedString("*") for -1, otherwise number
    template<int64_t D>
    static constexpr auto dim_str()
    {
        if constexpr (D < 0)
            return FixedString("*");
        else
            return uint_to_str<size_t(D)>::value;
    }

    static constexpr auto name = concat(
        FixedString("shape=("),
        concat_with_separator(FixedString(","), dim_str<Dims>()...),
        FixedString(")")
    );
};

/* Alternative way of specifying shape when only the number of dimensions is fixed.
* Eg: NDArray<ndim<3>> is equivalent to NDArray<shape<-1, -1, -1>>. */
template<int32_t N>
struct ndim {};

namespace detail
{
/* Sentinel for representing unused traits, eg. NDArray<> has no dtype trait so value_trait is unused.
Name is "" so that concat_with_separator skips in string_repr. */
struct unused { static constexpr FixedString name = ""; };

// Sentinel for when ndim is not known at compile time
inline constexpr int32_t ndim_dynamic = -1;

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
This instead relies on SFINAE on py::detail::npy_format_descriptor<T>().
Not great either since we now depend on pybind internals.
FIXME: just introduce our own dtype enum that is trivially castable to numpy typenum? */
template<typename T>
struct is_numpy_scalar<T, std::void_t<
    decltype(py::detail::npy_format_descriptor<T>::value)
>> : std::true_type {};
//~

// Shape predicates
template<typename T> struct is_shape_like : std::false_type {};
template<int64_t... D> struct is_shape_like<shape<D...>> : std::true_type {};
template<int32_t N> struct is_shape_like<ndim<N>> : std::true_type {};
//~

// Shape is either unused, given directly as shape<Dims...> or indirectly as ndim<N>. For ndim<N> we build a corresponding shape<Dims...>

template<typename T>
struct make_shape
{
    // sanity check for ourselves
    static_assert(std::is_same_v<T, unused> || is_shape_like<T>::value,
        "make_shape may only be instantiated with shape<> or ndim<>"
    );
    using type = T;  // shape<D...>
};

template<int32_t N>
struct make_shape<ndim<N>>
{
private:
    template<size_t... I>
    static shape<((void)I, int64_t(-1))...> helper(std::index_sequence<I...>);

public:
    using type = decltype(helper(std::make_index_sequence<N>{}));
};


/* Type-list find_if: scans Ts... and returns the first type T for which Pred<T>::value is true.
Pred is a metafunction predicate passed as a template template parameter.
Returns Default if no match is found. */
template<typename Default, template<typename, typename...> class Pred, typename... Ts>
struct Select;

template<typename Default, template<typename, typename...> class Pred>
struct Select<Default, Pred>
{
    using type = Default;
};

template<typename Default, template<typename, typename...> class Pred, typename T, typename... Ts>
struct Select<Default, Pred, T, Ts...>
{
    using type = std::conditional_t<
        Pred<T>::value,
        T,
        typename Select<Default, Pred, Ts...>::type>;
};

/* Used for ensuring that we only select each trait once. Eg. NDArray<float, double> is not valid.
This relies on predicates being careful with SFINAE.
*/
template<template<typename, typename...> class Pred, typename... Ts>
constexpr size_t count_matches()
{
    return (size_t(0) + ... + size_t(Pred<Ts>::value));
}
//~

// NDArrayTraits collects static array metadata traits, eg. dtype, data layout and shape, if they are known at compile time
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

    // float, double, std::complex<double>, etc. Or simply unused
    using value_trait = typename Select<unused, is_numpy_scalar, Args...>::type;
    using device_trait = typename Select<unused, is_device, Args...>::type;
    using layout_trait = typename Select<unused, is_layout, Args...>::type;
    // Always either unused or shape<Dims...>. If ndim<N> was given, convert it to shape<...>
    using shape_trait = typename make_shape<
        typename Select<unused, is_shape_like, Args...>::type
    >::type;

    static constexpr bool has_dtype = !std::is_same_v<value_trait, unused>;
    using Scalar = std::conditional_t<has_dtype, value_trait, void>;

    static constexpr bool has_device = !std::is_same_v<device_trait, unused>;
    static constexpr bool is_cpu = std::is_same_v<device_trait, device::cpu>;
    static constexpr bool is_gpu = std::is_same_v<device_trait, device::gpu>;

    /* True if we know anything at all about the shape. Knowing ndim is sufficient,
    and knowing all or some dimension sizes is also OK. */
    static constexpr bool has_shape = !std::is_same_v<shape_trait, unused>;
    static constexpr int32_t ndim = []()
    {
        if constexpr (!has_shape) return detail::ndim_dynamic;
        else return shape_trait::ndim;
    }();
    static constexpr auto static_shape = []()
    {
        if constexpr (!has_shape) return std::array<int64_t, 0>{};
        else return shape_trait::dims;
    }();

    static constexpr bool has_static_ndim = (ndim != detail::ndim_dynamic);

    static constexpr bool is_contiguous = std::is_same_v<layout_trait, c_contig> || std::is_same_v<layout_trait, f_contig>;
    // 1D contiguous arrays are both c_contig and f_contig
    static constexpr bool is_c_contig = (ndim == 1) ? is_contiguous : std::is_same_v<layout_trait, c_contig>;
    static constexpr bool is_f_contig = (ndim == 1) ? is_contiguous : std::is_same_v<layout_trait, f_contig>;

    /* Get FixedString representation of value_trait. Hard because we don't have reflection.
    Instead, this uses pybind11's dtype -> name mapping. Not great because it's considered an internal detail. */
    static constexpr auto value_type_name = []()
    {
        if constexpr(!has_dtype)
        {
            return FixedString("");
        }
        else
        {
            return FixedString(py::detail::npy_format_descriptor<value_trait>::name.text);
        }
    }();

    // FixedString string representation of all traits.
    static constexpr auto string_repr = concat_with_separator(
        FixedString(", "),
        value_type_name,
        device_trait::name,
        layout_trait::name,
        shape_trait::name
    );
};


template<int32_t N>
struct ShapeDataStatic
{
    static constexpr int32_t ndim = N;
    std::array<int64_t, N> shape;
    // Strides are given in BYTES
    std::array<int64_t, N> strides;
};

struct ShapeDataDynamic
{
    /* Upper bound on allowed number of array dimensions. Hardcoded in order to avoid heap allocs.
    * This can be increased at the cost of larger memory footprint of NDArrays whose ndim is not known at compile time. */
    static constexpr uint8_t MAX_NDIM = 8;
    int32_t ndim;

    std::array<int64_t, MAX_NDIM> shape;
    // Strides are given in BYTES
    std::array<int64_t, MAX_NDIM> strides;
};

enum class Framework
{
    eNumpy, eCupy
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
    using traits = detail::NDArrayTraits<Args...>;
    // Type of elements contained in the array. Will be void if no value trait was specified.
    using Scalar = typename traits::Scalar;

    // Must be default constructible for pybind11 type caster
    NDArray() = default;

    // FIXME: not optimal, py::dtype(typenum) constructor is not constexpr and may throw
    py::dtype dtype() const { return py::dtype(typenum); }

    constexpr int32_t ndim() const { return shape_data.ndim; }
    // Returns shape in the i. dimension. Does not check bounds!
    constexpr int64_t shape(size_t i) const { return shape_data.shape[i]; }
    // Returns stride in the i. dimension, in bytes. Does not check bounds!
    constexpr int64_t stride(size_t i) const { return shape_data.strides[i]; }

    const int64_t* shape_ptr() const { return shape_data.shape.data(); }
    const int64_t* stride_ptr() const { return shape_data.strides.data(); }
    bool is_valid() const { return !handle.is(py::handle(nullptr)); }

    // Number of elements in the array
    constexpr size_t size() const
    {
        size_t res = (size_t)is_valid();
        for (int32_t i = 0; i < ndim(); ++i)
        {
            res *= shape(i);
        }
        return res;
    }

    // Size of array element in bytes
    constexpr size_t itemsize() const { return (size_t) dtype().itemsize(); }
    // Total number of bytes occupied by the array
    constexpr size_t nbytes() const { return itemsize() * size(); }

    // Returns true if the data is on GPU
    constexpr bool on_gpu() const
    {
        if constexpr (traits::has_device) return traits::is_gpu;
        else return data_on_gpu;
    }

    /* Get read/write access to the underlying data pointer. Pointer type will match the value trait if it was specified,
    otherwise will return a void pointer. */
    Scalar* data()
    {
        return reinterpret_cast<Scalar*>(static_cast<std::byte*>(data_ptr) + byte_offset);
    }

    // Get read-only access to the underlying data pointer.
    const Scalar* data() const
    {
        return reinterpret_cast<const Scalar*>(static_cast<std::byte*>(data_ptr) + byte_offset);
    }

    template<typename S = Scalar,
             std::enable_if_t<!std::is_void_v<S>, int> = 0>
    S& at(const int64_t* indices)
    {
        static_assert(!(traits::has_device && traits::is_gpu),
            "NDArray::at() cannot be used with GPU arrays.");

        if (on_gpu())
        {
            throw std::runtime_error("NDArray::at() cannot be used with GPU arrays.");
        }

        std::byte* ptr = reinterpret_cast<std::byte*>(data());
        for (int32_t i = 0; i < ndim(); ++i)
        {
            ptr += indices[i] * stride(i);
        }
        return *reinterpret_cast<S*>(ptr);
    }

    template<typename S = Scalar,
             std::enable_if_t<!std::is_void_v<S>, int> = 0>
    const S& at(const int64_t* indices) const
    {
        static_assert(!(traits::has_device && traits::is_gpu),
            "NDArray::at() cannot be used with GPU arrays.");

        if (on_gpu())
        {
            throw std::runtime_error("NDArray::at() cannot be used with GPU arrays.");
        }

        const std::byte* ptr = reinterpret_cast<const std::byte*>(data());
        for (int32_t i = 0; i < ndim(); ++i)
        {
            ptr += indices[i] * stride(i);
        }
        return *reinterpret_cast<const S*>(ptr);
    }

    template<typename S = Scalar,
             std::enable_if_t<!std::is_void_v<S>, int> = 0>
    S& flat(size_t index)
    {
        static_assert(!(traits::has_device && traits::is_gpu),
            "NDArray::flat() cannot be used with GPU arrays.");

        if (on_gpu())
        {
            throw std::runtime_error("NDArray::flat() cannot be used with GPU arrays.");
        }
        return data()[index];
    }

    template<typename S = Scalar,
             std::enable_if_t<!std::is_void_v<S>, int> = 0>
    const S& flat(size_t index) const
    {
        static_assert(!(traits::has_device && traits::is_gpu),
            "NDArray::flat() cannot be used with GPU arrays.");

        if (on_gpu())
        {
            throw std::runtime_error("NDArray::flat() cannot be used with GPU arrays.");
        }
        return data()[index];
    }

    template<typename I, size_t N, typename S = Scalar,
             std::enable_if_t<!std::is_void_v<S>, int> = 0>
    S& operator()(const IndexArray<I, N>& indices)
    {
        if (N != static_cast<size_t>(ndim()))
        {
            throw std::invalid_argument("NDArray::operator() called with wrong number of indices.");
        }

        std::array<int64_t, N> index{};
        for (size_t i = 0; i < N; i++)
        {
            index[i] = static_cast<int64_t>(indices[i]);
        }
        return at(index.data());
    }

    template<typename I, size_t N, typename S = Scalar,
             std::enable_if_t<!std::is_void_v<S>, int> = 0>
    const S& operator()(const IndexArray<I, N>& indices) const
    {
        if (N != static_cast<size_t>(ndim()))
        {
            throw std::invalid_argument("NDArray::operator() called with wrong number of indices.");
        }

        std::array<int64_t, N> index{};
        for (size_t i = 0; i < N; i++)
        {
            index[i] = static_cast<int64_t>(indices[i]);
        }
        return at(index.data());
    }

    template<typename... Indices, typename S = Scalar,
             std::enable_if_t<!std::is_void_v<S>, int> = 0>
    S& operator()(Indices... indices)
    {
        static_assert((std::is_integral_v<Indices> && ...),
            "NDArray indices must be integral.");
        static_assert(!(traits::has_device && traits::is_gpu),
            "NDArray::operator() cannot be used with GPU arrays.");

        if (on_gpu())
        {
            throw std::runtime_error("NDArray::operator() cannot be used with GPU arrays.");
        }
        if (sizeof...(Indices) != static_cast<size_t>(ndim()))
        {
            throw std::invalid_argument("NDArray::operator() called with wrong number of indices.");
        }

        std::array<int64_t, sizeof...(Indices)> index = {
            static_cast<int64_t>(indices)...
        };
        return at(index.data());
    }

    template<typename... Indices, typename S = Scalar,
             std::enable_if_t<!std::is_void_v<S>, int> = 0>
    const S& operator()(Indices... indices) const
    {
        static_assert((std::is_integral_v<Indices> && ...),
            "NDArray indices must be integral.");
        static_assert(!(traits::has_device && traits::is_gpu),
            "NDArray::operator() cannot be used with GPU arrays.");

        if (on_gpu())
        {
            throw std::runtime_error("NDArray::operator() cannot be used with GPU arrays.");
        }
        if (sizeof...(Indices) != static_cast<size_t>(ndim()))
        {
            throw std::invalid_argument("NDArray::operator() called with wrong number of indices.");
        }

        std::array<int64_t, sizeof...(Indices)> index = {
            static_cast<int64_t>(indices)...
        };
        return at(index.data());
    }

    template<typename I, size_t N>
    void require_shape(const IndexArray<I, N>& expected, const char* message) const
    {
        if (N != static_cast<size_t>(ndim()))
        {
            throw std::invalid_argument(message);
        }
        for (size_t i = 0; i < N; i++)
        {
            if (shape(i) != static_cast<int64_t>(expected[i]))
            {
                throw std::invalid_argument(message);
            }
        }
    }

    template<typename F>
    void for_each_index(F&& f) const
    {
        static_assert(traits::ndim != detail::ndim_dynamic,
            "NDArray::for_each_index() requires compile-time ndim.");
        static_assert(!(traits::has_device && traits::is_gpu),
            "NDArray::for_each_index() cannot be used with GPU arrays.");

        if (on_gpu())
        {
            throw std::runtime_error("NDArray::for_each_index() cannot be used with GPU arrays.");
        }

        IndexArray<long, static_cast<size_t>(traits::ndim)> index;
        for_each_index_impl<0>(index, f);
    }

    /* Import an array from Numpy or Cupy. Import will fail if not all requested traits are satisfied,
    in which case this returns false and the NDArray will be in invalid state.
    NB: this is intended to be used in type-casters only. */
    bool import_from_python(PyObject* obj) noexcept;

private:
    static constexpr bool has_static_ndim = (traits::ndim != -1);
    using ShapeData = std::conditional_t<
        has_static_ndim,
        detail::ShapeDataStatic<traits::ndim>,
        detail::ShapeDataDynamic
    >;

    void* data_ptr;
    int typenum;
    bool data_on_gpu;
    ShapeData shape_data;

    // Offset is TODO. Libraries don't really seem to use this ATM
    static constexpr uint64_t byte_offset = 0;
    // Non-owning reference to the actual array PyObject
    py::handle handle = nullptr;

    template<size_t D, size_t N, typename F>
    void for_each_index_impl(IndexArray<long, N>& index, F& f) const
    {
        if constexpr (D == N)
        {
            f(index);
        }
        else
        {
            for (long i = 0; i < static_cast<long>(shape(D)); i++)
            {
                index[D] = i;
                for_each_index_impl<D + 1>(index, f);
            }
        }
    }

    template<detail::Framework framework>
    bool import_from_python_impl(PyObject* obj) noexcept;
};

template<typename ...Args>
inline bool NDArray<Args...>::import_from_python(PyObject * obj) noexcept
{
    bool success = false;
    if (is_numpy_array(obj))
    {
        success = import_from_python_impl<detail::Framework::eNumpy>(obj);
    }
    else if (is_cupy_array(obj))
    {
        success = import_from_python_impl<detail::Framework::eCupy>(obj);
    }

    if (PyErr_Occurred())
    {
        // Got CPython error that is not due to mismatched array traits. Either we miscategorized the PyObject (not Numpy nor Cupy)
        // or there is a bug in ArrayAccessors, or the error actually occurred much earlier and whoever caused it forgot to handle and clear it.
        fprintf(stderr, "Unhandled Python exception occurred during gpaw::NDArray type cast. This can happen if the passed PyObject resembles a Numpy or Cupy array but is not one, or could imply a bug in GPAW. This error is unrecoverable!\n");
        PyErr_PrintEx(0);
        std::terminate();
    }

    return success;
}

template<typename ...Args>
template<detail::Framework framework>
inline bool NDArray<Args...>::import_from_python_impl(PyObject* obj) noexcept
{
    using detail::Framework;
    if constexpr (traits::is_cpu && framework != Framework::eNumpy)
    {
        return false;
    }
    else if constexpr (traits::is_gpu && framework != Framework::eCupy)
    {
        return false;
    }

    if constexpr (!gpaw::compiled_with_gpu_support && framework == Framework::eCupy)
    {
        fprintf(stderr, "You are importing a Cupy array into GPAW C++ extension, but the extension has not been compiled with GPU support (missing -DGPAW_GPU). This is most likely an error\n");
    }

    if constexpr (traits::has_device)
    {
        data_on_gpu = traits::is_gpu;
    }
    else
    {
        data_on_gpu = (framework == Framework::eCupy);
    }

    // For Numpy, cast to PyArrayObject* so that we trigger faster overloads of array accessors
    using pyobject_ptr_t = std::conditional_t<framework == Framework::eNumpy, PyArrayObject*, PyObject*>;
    pyobject_ptr_t arr = reinterpret_cast<pyobject_ptr_t>(obj);

    typenum = ArrayAccessors::typenum(arr);
    if constexpr (traits::has_dtype)
    {
        // py::dtype::num_of() would be constexpr but requires fairly recent pybind11
        //if (out.typenum != py::dtype::num_of<typename traits::value_trait>())

        if (typenum != py::dtype::of<typename traits::value_trait>().normalized_num())
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

    const int32_t ndim = static_cast<int32_t>(ArrayAccessors::ndim(arr));
    if constexpr (traits::ndim != detail::ndim_dynamic)
    {
        if (ndim != traits::ndim)
        {
            return false;
        }
    }
    else
    {
        if (ndim > detail::ShapeDataDynamic::MAX_NDIM)
        {
            fprintf(stderr, "gpaw::NDArray only supports up to 8 dynamical array dimension. Your array has: %d. Please use the compile-time trait gpaw::ndim<%d>\n", (int)ndim, (int)ndim);
            return false;
        }
        shape_data.ndim = ndim;
    }

    ArrayAccessors::fill_shape(arr, shape_data.shape.data(), ndim);
    ArrayAccessors::fill_strides(arr, shape_data.strides.data(), ndim);

    if constexpr (traits::has_shape)
    {
        // Check that obtained shape matches the compile-time shape. Shape == -1 is wildcard and can be anything
        for (int32_t i = 0; i < traits::ndim; ++i)
        {
            if (traits::static_shape[i] != -1 && shape_data.shape[i] != traits::static_shape[i])
            {
                return false;
            }
        }
    }

    data_ptr = ArrayAccessors::data(arr);
    handle = py::handle(obj);
    return true;
}

} // namespace gpaw


// Pybind11 type casters
namespace pybind11::detail
{

template<typename T, size_t N>
struct type_caster<gpaw::IndexArray<T, N>>
{
    using Type = gpaw::IndexArray<T, N>;
    PYBIND11_TYPE_CASTER(Type, const_name("IndexArray"));

    static handle cast(const gpaw::IndexArray<T, N>& array, return_value_policy policy, handle parent) noexcept
    {
        assert(false && "Passing gpaw::IndexArray back to Python is not implemented");
        return handle();
    }

    bool load(handle src, [[maybe_unused]] bool implicit_convert) noexcept
    {
        if (!PySequence_Check(src.ptr()))
        {
            return false;
        }

        try
        {
            pybind11::sequence sequence = pybind11::reinterpret_borrow<pybind11::sequence>(src);
            if (sequence.size() != N)
            {
                return false;
            }

            std::array<T, N> values{};
            for (size_t i = 0; i < N; i++)
            {
                values[i] = sequence[i].cast<T>();
            }
            value = gpaw::IndexArray<T, N>(values);
            return true;
        }
        catch (const pybind11::error_already_set&)
        {
            if (PyErr_Occurred())
            {
                PyErr_Clear();
            }
            return false;
        }
        catch (const std::exception&)
        {
            return false;
        }
    }
};

// Try to follow nanobind semantics here so that we have easier time porting to it later if needed.
// https://nanobind.readthedocs.io/en/latest/porting.html#type-casters

template<typename... Args>
struct type_caster<gpaw::NDArray<Args...>>
{
    using traits = gpaw::detail::NDArrayTraits<Args...>;

    /* Generate a name that will appear in stubs. But:
        1. Mypy stubgen doesn't like types with [], <> etc
        2. If we bypass 1. with additional quotes like "\"Ndarray[]\"", the type will appear only in the docstring, not stub function signatures.
    Doing this properly would probably require declaration like `class NDArray(Generic[T]):` on Python side;
    dunno if this is doable with Mypy stubgen.
    With pybind11-stubgen, problem 1. still persists but 2. works as one would expect.
    But still need to define Python-side generics to make editors understand the stubs => combine with a manual stub file?
    */
    static constexpr auto ndarray_str = gpaw::concat(gpaw::FixedString("\"NDArray["),
                                                     traits::string_repr,
                                                     gpaw::FixedString("]\""));

    PYBIND11_TYPE_CASTER(gpaw::NDArray<Args...>, const_name(ndarray_str.data));

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
    bool load(handle src, [[maybe_unused]] bool implicit_convert) noexcept
    {
        PyObject* obj = src.ptr();
        assert(obj);
        return value.import_from_python(obj);
    }
};

} // namespace pybind11::detail
