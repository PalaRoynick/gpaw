from __future__ import annotations

from math import isqrt, pi, sqrt

import numpy as np
import wignernj

from gpaw.typing import Array3D, Array4D

_gaunt: dict[int, np.ndarray] = {}
_nabla: dict[int, np.ndarray] = {}
_super_gaunt: dict[int, np.ndarray] = {}

# m of the l=1 real spherical harmonic proportional to the cartesian
# coordinate v.  The real harmonics are ordered as (y, z, x), i.e. by
# m = -1, 0, 1, so that x, y and z have m = 1, -1 and 0.
_mv = (1, -1, 0)


def _check_conventions() -> None:
    """Guard against a change of argument convention in libwignernj.

    The released library takes the angular momenta as plain integers.  A
    development version documents passing twice the angular momentum instead,
    which would silently produce wrong coefficients here.  Bail out loudly
    rather than compute nonsense.
    """
    # <Y_10 Y_10 Y_20> = 1 / sqrt(5 pi) exactly, and the harmonics must be
    # the real ones in the same phase convention as the ones used by GPAW.
    ref = 1 / sqrt(5 * pi)
    if abs(wignernj.gaunt_real(1, 0, 1, 0, 2, 0) - ref) > 1e-12:
        raise RuntimeError(
            'The installed wignernj does not use the argument convention '
            'expected by GPAW (plain angular momenta, not twice their '
            'values).  Please install a compatible version.')


_check_conventions()


def gaunt(lmax: int = 2) -> Array3D:
    r"""Gaunt coefficients

    :::

         ^      ^     -- L      ^
      Y (r)  Y (r) =  > G    Y (r)
       L      L       -- L L  L
        1      2      L   1 2
    """

    if lmax in _gaunt:
        return _gaunt[lmax]

    l2max = 2 * lmax
    Lmax = (lmax + 1)**2
    L2max = (l2max + 1)**2

    G_LLL = np.zeros((Lmax, L2max, L2max))
    for L1 in range(Lmax):
        l1 = isqrt(L1)
        m1 = L1 - l1**2 - l1
        for L2 in range(L2max):
            l2 = isqrt(L2)
            m2 = L2 - l2**2 - l2
            # Gaunt coefficients vanish unless the triangle inequality is
            # fulfilled, l1 + l2 + l3 is even, and |m3| is |m1 + m2| or
            # |m1 - m2|
            m3_m = {abs(m1 + m2), abs(m1 - m2)}
            for l3 in range(abs(l1 - l2), min(l1 + l2, l2max) + 1, 2):
                for m3 in m3_m:
                    if m3 > l3:
                        continue
                    for m in {m3, -m3}:
                        G_LLL[L1, L2, l3**2 + l3 + m] = wignernj.gaunt_real(
                            l1, m1, l2, m2, l3, m)

    _gaunt[lmax] = G_LLL
    return G_LLL


def nabla(lmax: int = 2) -> Array3D:
    r"""Create the array of derivative integrals.

    :::

      /  ^    ^   1-l' d   l'    ^
      | dr Y (r) r     --[r  Y  (r)]
      /     L           _     L'
                       dr

    The derivative of a solid harmonic is itself a solid harmonic of degree
    l'-1, so the integral is simply a Gaunt coefficient:

    :::

              ____
             /4 pi
      Y    = |----  (2 l' + 1) G           delta
       LL'v \/  3               L,L ,L'         l,l'-1
                                   v

    where L_v is the l=1 harmonic proportional to the cartesian coordinate v.
    """

    if lmax in _nabla:
        return _nabla[lmax]

    Lmax = (lmax + 1)**2
    Y_LLv = np.zeros((Lmax, Lmax, 3))
    for L2 in range(Lmax):
        l2 = isqrt(L2)
        if l2 == 0:
            continue
        m2 = L2 - l2**2 - l2
        c = sqrt(4 * pi / 3) * (2 * l2 + 1)
        l1 = l2 - 1
        for m1 in range(-l1, l1 + 1):
            for v, mv in enumerate(_mv):
                Y_LLv[l1**2 + l1 + m1, L2, v] = c * wignernj.gaunt_real(
                    l1, m1, 1, mv, l2, m2)

    _nabla[lmax] = Y_LLv
    return Y_LLv


def super_gaunt(lmax: int = 2) -> Array4D:
    r"""Product of two Gaunt coefficients.

    This gives the coefficients to contractions of three spherical harmonics:
                        __
       ˰     ˰     ˰    \   L       ˰
    Y (r) Y (r) Y (r) = /  G      Y (r)
     L     L     L      ‾‾  L L L  L
      1     2     3     L    1 2 3

    where:
              __
     L        \   L'    L
    G       = /  G     G
     L L L    ‾‾  L L   L' L
      1 2 3   L'   1 2      3
    """
    if lmax in _super_gaunt:
        return _super_gaunt[lmax]

    G1_LLL = gaunt(lmax)
    G2_LLL = gaunt(2 * lmax)

    Lmax1 = G1_LLL.shape[0]  # l=0 to l=lmax
    Lmax2 = G2_LLL.shape[0]  # l=0 to l=2*lmax
    G_LLLL = np.einsum('ijk,klm->ijlm',
                       G1_LLL[:, :Lmax1],
                       G2_LLL[:, :Lmax2])

    _super_gaunt[lmax] = G_LLLL
    return G_LLLL
