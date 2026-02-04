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

/* Concatenate any number of FixedStrings, adding a separator string between two non-empty strings. */
template<size_t SeparatorLength, typename... Parts>
constexpr auto concat_with_separator(const FixedString<SeparatorLength>& sep, const Parts&... parts)
{
    // NB: in C++20 could use "FixedString Sep" as template parameter instead.

    static_assert((is_fixed_string<Parts>::value && ...),
        "concat_with_separator only accepts FixedString arguments"
    );

    constexpr size_t nonEmpty =
        (0 + ... + (Parts::size() > 1));

    constexpr size_t size =
        1
        + (0 + ... + (Parts::size() - 1))
        + (nonEmpty > 0 ? (nonEmpty - 1) * (sep.size() - 1) : 0);

    FixedString<size> out{};
    size_t pos = 0;
    bool first = true;

    auto append = [&](auto const& p)
    {
        if constexpr (p.size() > 1)
        {
            if (!first)
            {
                for (size_t i = 0; i + 1 < sep.size(); ++i)
                    out.data[pos++] = sep.data[i];
            }
            first = false;

            for (size_t i = 0; i + 1 < p.size(); ++i)
                out.data[pos++] = p.data[i];
        }
    };

    (append(parts), ...);
    out.data[pos] = '\0';
    return out;
}


// Concatenate FixedStrings. No separators added
template<typename... Parts>
constexpr auto concat(const Parts&... parts)
{
    return concat_with_separator(FixedString(""), parts...);
}

};
