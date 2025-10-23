#pragma once

#include "gpaw_utils.h"

#ifdef __cplusplus
    #define GPAW_CPP 1
#else
    #define GPAW_CPP 0
#endif

/* Handle complex number differences in C++ vs C99. Old C-code uses custom double_complex
type, so define it here. For C++, define also some utility functions from complex.h.
Finally, old C-code assumes a borderline illegal `typedef I` for the imaginary unit,
so for now we have to include it here too (FIXME!).

Note that complex.h should NOT be included anywhere else because it can do typedefs that override std::complex.
*/
#if GPAW_CPP
    #include <complex>
    using double_complex = std::complex<double>;

    // Get real part of a complex double. Provided for compatibility with C99 code
    constexpr inline double creal(const double_complex& z) { return z.real(); }

    // Get imaginary part of a complex double. Provided for compatibility with C99 code
    constexpr inline double cimag(const double_complex& z) { return z.imag(); }

    // Computes the complex base-e exponential of z. Provided for compatibility with C99 code
    inline double_complex cexp(const double_complex& z) { return std::exp(z); }
    // NB: std::exp is not constexpr until C++26

    /* DO NOT USE IN NEW CODE!
    Shorthand for the imaginary unit, needed for compatibility with legacy C code. */
    constexpr double_complex I(0.0, 1.0);

#else
    // C99 mode
    #include <complex.h>
    typedef double complex double_complex;

    /* Numpy 2.0 undefines I (see Numpy PR 26789), so we have to put it
    * back */
    #ifndef I
        #define I _Complex_I
    #endif

#endif

/* Handle `restrict` keyword not existing in C++. Use compiler extension if
supported, otherwise simply leave GPAW_RESTRICT undefined */
#if GPAW_CPP
    #if defined(__clang__) || defined(__GNUC__)
        #define GPAW_RESTRICT __restrict__
    #elif defined(_MSC_VER)
        #define GPAW_RESTRICT __restrict
    #elif defined(_CRAYC)
        #define GPAW_RESTRICT __restrict
    #elif defined(__INTEL_COMPILER) || defined(__INTEL_LLVM_COMPILER)
        #define GPAW_RESTRICT __restrict
    #else
        // Unsupported compiler? Leave empty
        #define GPAW_RESTRICT
    #endif

#else
    // C99 mode
    #define GPAW_RESTRICT restrict
#endif