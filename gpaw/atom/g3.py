from scipy.special import genlaguerre
import numpy as np
from gpaw.atom.aeatom import AllElectronAtom
from gpaw.atom.generator2 import PAWSetupGenerator
from os import devnull


def main(symbol: str,
         projectors: str,
         rc: float,
         r0: float,
         radii,
         alpha: float):
    aea = AllElectronAtom(
        symbol,
        'LDA',
        scalar_relativistic=False,
        log=open(devnull, 'w'))
    gen = PAWSetupGenerator(aea, projectors)
    gen.construct_shape_function(rc=rc)
    gen.calculate_core_density()
    gen.find_local_potential(r0, 5)
    gen.add_waves(radii)
    vtr_r = gen.vtr_g
    for w in gen.waves_l:
        l = w.l
        ch = aea.channels[l]
        N =len(ch.e_n)
        print(l, N, w.e_n, ch.e_n)
        basis = ch.basis
        H = basis.T_bb + basis.calculate_potential_matrix(vtr_r)
        print(H.shape)
        rgd = aea.rgd
        r_r = rgd.r_g
        p_r = np.exp(-alpha * r_r**2)
        r2 = 1.5
        a = np.exp(-alpha * r2**2)
        b = -2 * alpha * r2 * a
        B = -b / 2 / r2
        A = -a - B * r2**2
        qp_r = p_r + A + B * r_r**2
        n2 = rgd.ceil(r2)
        qp_r[n2:] = 0.0
        p2_r = np.exp(-alpha * r_r**2) * (1.5 - 2 * alpha * r_r**2)
        p_b = np.linalg.lstsq(basis.basis_bg.T, p_r)[0]
        print(p_b)
        import matplotlib.pyplot as plt
        e = w.e_n[0]
        c_b = np.linalg.solve(H - e * np.eye(N), p_b)
        c_r = basis.expand(c_b)
        r = rgd.ceil(radii[l])
        c_r *= ch.phi_ng[0, r] / c_r[r]
        plt.plot(r_r, p_r)
        plt.plot(r_r, qp_r)
        plt.plot(r_r, p2_r)
        plt.plot(r_r, c_r)
        plt.plot(r_r, ch.phi_ng[0])
        plt.xlim(0, 3.0)
        plt.show()

def howf(alpha, r, l=0, n=0, rc=np.inf):
    L = genlaguerre(n, l + 0.5)
    y = alpha * r**2
    x = np.exp(-y)
    p = L(2 * y)
    z = x * p
    if np.isinf(rc):
        return z
    # Subtract a + b * r**2 so that value and derivative at r=rc is zero:
    y0 = alpha * rc**2
    x0 = np.exp(-y0)
    dx0 = -2 * alpha * rc * x0
    p0 = L(2 * y0)
    dp0 = 2 * alpha * rc * L.deriv(2 * y0)
    b = (x0 * dp0 + dx0 * p0) / (2 * rc)
    a = x0 * p0 - b * rc**2
    z[r < rc] -= a + b * r[r < rc]**2
    z[r >= rc] = 0.0
    return z


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    r = np.linspace(0, 4, 101)
    for l in range(3):
        for n in range(2):
            p = howf(3.0, r, l=l, n=n)
            plt.plot(r, p * r**l)
            p = howf(3.0, r, l=l, n=n, rc=1.5)
            plt.plot(r, p * r**l)
    plt.show()
    # main('H', '1s,P', 0.8, 0.8, [0.9], 3.0)
