#pragma once

#include "python_utils.h"

namespace gpaw
{

namespace device
{
struct cpu {};
struct gpu {};
}

struct c_style {};
struct f_style {};

namespace detail
{

template<typename T>
struct is_device : std::false_type {};

template<> struct is_device<device::cpu> : std::true_type {};
template<> struct is_device<device::gpu> : std::true_type {};

template<typename T>
struct is_layout : std::false_type {};

template<> struct is_layout<c_style> : std::true_type {};
template<> struct is_layout<f_style> : std::true_type {};


}


template<typename... Args>
class NDArray
{
public:

using valueType  = typename Select<void, std::is_arithmetic, Args...>::type;
using deviceType = typename Select<device::cpu, isDevice, Args...>::type;
using layoutType = typename Select<void, isLayout, Args...>::type;

void* handle = nullptr; // PyArrayObject*, CuPy handle, etc.

static constexpr bool hasDType  = !std::is_same_v<valueType, void>;
static constexpr bool isGPU     = std::is_same_v<deviceType, device::gpu>;
static constexpr bool isCStyle  = std::is_same_v<layoutType, c_style>;


};

} // namespace gpaw
