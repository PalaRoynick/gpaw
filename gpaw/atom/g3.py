from scipy.special import genlaguerre
import numpy as np
from gpaw.atom.aeatom import Channel
from gpaw.atom.generator2 import PAWWaves


def pseudize(waves: PAWWaves,
             channel: Channel,
             vtr_r: np.ndarray) -> None:
    l = waves.l
    N = len(waves.e_n)
    B = len(channel.e_n)
    print(l, N, B, waves.e_n)
    print(channel.e_n)
    waves.e_n[1] = channel.e_n[1]
    basis = channel.basis
    H = basis.T_bb + basis.calculate_potential_matrix(vtr_r)
    print(H.shape)
    rgd = waves.rgd
    r_r = rgd.r_g
    alpha = 3.0
    rc = 1.5
    h_kr = [howf(alpha, r_r, l, n, rc) for n in range(N)]
    h_kb = [np.linalg.lstsq(basis.basis_bg.T, h_r)[0] for h_r in h_kr]
    for n, e in enumerate(waves.e_n):
        f_kb = []
        A_kk = []
        for h_b in h_kb:
            f_b = np.linalg.solve(H - e * np.eye(B), h_b)
            f_kb.append(f_b)
            A_kk.append([4 * np.pi * float(f_b @ h_b) for h_b in h_kb])

        B_k = np.linalg.solve(A_kk,
                              [1.0 if k == n else 0.0 for k in range(N)])
        print(n, B_k)
        f_b = B_k @ f_kb
        print([4 * np.pi * f_b @ h_b for h_b in h_kb])
        f_r = basis.expand(f_b)
        plt.plot(r_r, f_r)
        print([rgd.integrate(f_r * h_r) for h_r in h_kr])
    plt.xlim(0, 3.0)
    plt.show()
    return
    c_r = basis.expand(c_b)
    r = rgd.ceil(radii[l])
    c_r *= ch.phi_ng[0, r] / c_r[r]
    plt.plot(r_r, p_r)
    plt.plot(r_r, qp_r)
    plt.plot(r_r, p2_r)
    plt.plot(r_r, c_r)
    plt.plot(r_r, ch.phi_ng[0])


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
