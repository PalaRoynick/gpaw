#pragma once

#include "gpaw_utils.h"

#include <functional>
#include <memory>
#include <type_traits>
#include <cstdint>
#include <sstream>
#include <unordered_map>

namespace gpaw
{

/* Like std::span but works in C++17.
TODO: replace with std::span when using C++20. */
template<typename T>
class Span
{
public:
    constexpr Span() noexcept : ptr_(nullptr), size_(0) {}
    constexpr Span(T* ptr, size_t size) noexcept : ptr_(ptr), size_(size) {}

    template<typename Container>
    constexpr Span(Container& c) noexcept : ptr_(c.data()), size_(c.size()) {}

    template<typename Container>
    constexpr Span(const Container& c) noexcept : ptr_(c.data()), size_(c.size()) {}

    constexpr T& operator[](size_t i) const noexcept { return ptr_[i]; }
    constexpr T* data() const noexcept { return ptr_; }
    constexpr size_t size() const noexcept { return size_; }
    constexpr bool empty() const noexcept { return size_ == 0; }
    constexpr T* begin() const noexcept { return ptr_; }
    constexpr T* end() const noexcept { return ptr_ + size_; }

private:
    T* ptr_;
    size_t size_;
};


/* Like std::source_location but works in C++17. Uses Clang/GCC extensions.
TODO: replace with std::source_location when using C++20. */
struct SourceLocation
{

    constexpr SourceLocation(
        const char* file_ = __builtin_FILE(),
        const char* func_ = __builtin_FUNCTION(),
        int line_ = __builtin_LINE()
    ) noexcept : file(file_), function(func_), line(line_)
    {
    }

    static constexpr SourceLocation current(
        const char* file_ = __builtin_FILE(),
        const char* func_ = __builtin_FUNCTION(),
        int line_ = __builtin_LINE()) noexcept
    {
        return SourceLocation(file_, func_, line_);
    }

    // Returns string `filename:line`
    std::string to_string() const
    {
        return std::string(file) + ":" + std::to_string(line);
    }

    const char* file;
    const char* function;
    uint32_t line;
};


// Use in if constexpr (...) to indicate an impossible branch
template<bool flag = false> void static_no_match() { static_assert(flag, "No constexpr match"); }

// Templated C-style malloc. NB: allocs num_elements * sizeof(T) bytes
template<typename T>
T* TMalloc(size_t num_elements)
{
    return static_cast<T*>(malloc(num_elements * sizeof(T)));
}

// Templated C-style free()
template<typename T>
void TFree(T* ptr)
{
    free(static_cast<void*>(ptr));
}

// Checks if an std::unordered_map contains a key
template<typename Key, typename Val>
inline bool map_contains(const std::unordered_map<Key, Val>& map, Key key)
{
    return map.find(key) != map.end();
}

} // namespace gpaw
