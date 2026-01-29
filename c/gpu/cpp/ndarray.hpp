#pragma once

#include "python_utils.h"
#include <type_traits>

#include "pyarray_utils.hpp"


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

template<typename T>
struct is_device : std::false_type {};

template<> struct is_device<device::cpu> : std::true_type {};
template<> struct is_device<device::gpu> : std::true_type {};

template<typename T>
struct is_layout : std::false_type {};

template<> struct is_layout<c_contig> : std::true_type {};
template<> struct is_layout<f_contig> : std::true_type {};



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


struct ArrayDesc
{
    // Use signed integers for compatibility with dlpack
    void* data = nullptr;
    int32_t ndim = 0;
    int64_t* shape = nullptr;
    int64_t* strides = nullptr;
};

} // namespace detail

template<typename... Args>
class GPAW_HIDDEN_SYMBOL NDArray
{
public:
    NDArray() = default;

using value_type  = typename detail::Select<void, std::is_arithmetic, Args...>::type;
using device_type = typename detail::Select<device::cpu, detail::is_device, Args...>::type;
using layout_type = typename detail::Select<void, detail::is_layout, Args...>::type;

static constexpr bool has_dtype = !std::is_same_v<value_type, void>;
static constexpr bool is_gpu = std::is_same_v<device_type, device::gpu>;
static constexpr bool is_c_contig = std::is_same_v<layout_type, c_contig>;

public:
    void from_cupy(py::handle obj)
    {
        desc.from_cupy(obj);
    }

private:
    py::handle handle = nullptr;
    // WIP, for now just use PyDeviceArray
    //detail::ArrayDesc desc;
    PyDeviceArray desc;

    friend struct pybind11::detail::type_caster<NDArray<Args...>>;
};

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
    using value_type = typename ndarray::value_type;
    using layout_type = typename ndarray::layout_type;

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
        // nanobind first attempts to construct the full ndarray (with dlpack).
        // If it fails, the handle will be nullptr, so this function pretty much just returns (handle != nullptr)

        if (PyArray_Check(src.ptr()))
        {
            // Numpy; cast and borrow (no reference counting)
            py::array arr = reinterpret_borrow<py::array>(src);

            if constexpr (value.has_dtype)
            {
                if (!arr.dtype().is(dtype::of<value_type>()))
                {
                    return false;
                }
            }

            // how to do c_contig, f_contig without a million if-elses ?

            value.desc.dtype = arr.dtype();
            value.handle = src;
        }
        else if (gpaw::is_cupy_array(src))
        {

        }
        else
        {
            return false;
        }
        return true;
    }
};

} // namespace pybind11::detail
