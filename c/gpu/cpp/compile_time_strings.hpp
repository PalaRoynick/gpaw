#pragma once

#include <cstddef>

// Compile-time string tools for building non-structured strings at compile time, for use as NTTP.
// Note that `constexpr std::string` won't do because it is structured and can't be used as a template parameter.

namespace gpaw
{

/* NTTP friendly compile-time string, null-terminated.
* Note that this cannot be constructed from `const char*`. Must use instead use eg.
*   const char str[] = "something";
*   FixedString fixed_str(str);  // same as FixedString("something");
*/
template<size_t N>
struct FixedString
{
    char data[N] {};
    constexpr FixedString() = default;
    constexpr FixedString(const char (&str)[N])
    {
        for (size_t i = 0; i < N; ++i)
        {
            data[i] = str[i];
        }
    }

    static constexpr size_t size() { return N; };
};


template<typename T> struct is_fixed_string : std::false_type {};
template<size_t N> struct is_fixed_string<FixedString<N>> : std::true_type {};

// Deduction guide
template<size_t N>
FixedString(const char (&)[N]) -> FixedString<N>;

// Recursive magic for integer -> FixedString conversion
template<size_t Rem, size_t... Digits>
struct uint_to_str : uint_to_str<Rem / 10, Rem % 10, Digits...> {};

// Convert unsigned integer to FixedString. Usage: FixedString s = uint_to_str<123>::value;
template<size_t... Digits>
struct uint_to_str<0, Digits...>
{
    static constexpr auto value =
        FixedString<sizeof...(Digits) + 1>{ { char('0' + Digits)..., '\0' } };
};

// Concatenate FixedStrings
template<typename... Parts>
constexpr auto concat(const Parts&... parts)
{
    static_assert((is_fixed_string<Parts>::value && ...),
        "concat only accepts FixedString arguments"
    );

    constexpr size_t size = 1 + (0 + ... + (parts.size() - 1));

    FixedString<size> out {};
    size_t pos = 0;

    auto append = [&](auto const& p)
    {
        for (size_t i = 0; i + 1 < p.size(); ++i)
        {
            out.data[pos++] = p.data[i];
        }
    };

    (append(parts), ...);
    out.data[pos] = '\0'; // always null-terminated
    return out;
};

};
