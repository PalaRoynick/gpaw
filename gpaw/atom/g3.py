from scipy.special import genlaguerre
import numpy as np
from gpaw.atom.aeatom import Channel
from gpaw.atom.generator2 import PAWWaves


def pseudize(waves: PAWWaves,
             channel: Channel,
             vtr_g: np.ndarray) -> None:
    l = waves.l
    N = len(waves.e_n)
    B = len(channel.e_n)
    rgd = waves.rgd
    r_g = rgd.r_g
    print(l, N, B, waves.e_n)
    for n, f in enumerate(waves.f_n):
        if f == 0.0:
            waves.e_n[n] = channel.e_n[5]
            waves.phi_ng[n] = channel.phi_ng[5]
    basis = channel.basis
    H = basis.T_bb + basis.calculate_potential_matrix(vtr_g)
    phi_ng = waves.phi_ng = np.array(waves.phi_ng)
    phit_ng = waves.phit_ng = np.empty((N, len(r_g)))
    pt_ng = waves.pt_ng = np.empty((N, len(r_g)))
    rc = waves.rcut*1.5
    gc = rgd.round(rc)
    alpha = 3.0 * (1.5 / rc)**2
    h_kg = [howf(alpha, r_g, l, n, rc) for n in range(N)]
    h_kb = [np.linalg.lstsq(basis.basis_bg.T, h_g)[0] for h_g in h_kg]
    import matplotlib.pyplot as plt
    for n, e in enumerate(waves.e_n):
        f_kb = []
        A_kk = []
        for h_b in h_kb:
            f_b = np.linalg.solve(H - e * np.eye(B), h_b)
            f_kb.append(f_b)
            A_kk.append([4 * np.pi * float(f_b @ h_b) for h_b in h_kb])
        B_k = np.linalg.solve(A_kk,
                              [1.0 if k == n else 0.0 for k in range(N)])
        f_b = B_k @ f_kb
        f_g = basis.expand(f_b)
        x = waves.phi_ng[n][gc] / f_g[gc]
        print(n, e, B_k, x)
        print(h_kb @ f_b * 4 * np.pi)
        print([rgd.integrate(f_g * h_g) for h_g in h_kg])
        phit_ng[n] = x * f_g
        pt_ng[n] = h_kg[n] / x
        plt.plot(r_g, phit_ng[n])
        plt.plot(r_g, phi_ng[n])
    plt.xlim(0, 4)
    plt.show()

    waves.nt_g = waves.f_n @ waves.phit_ng**2 / 4 / np.pi
    waves.dS_nn = (rgd.integrate(phi_ng[:, None] * phi_ng) -
                   rgd.integrate(phit_ng[:, None] * phit_ng)) / (4 * np.pi)
    waves.Q = waves.f_n @ waves.dS_nn.diagonal()
    print(waves.dS_nn, waves.Q, waves.e_n)
    A_nn = rgd.integrate(phit_ng[:, None] * pt_ng) / (4 * np.pi)
    print(A_nn)
    waves.dH_nn = waves.dS_nn @ np.diag(waves.e_n) - A_nn
    waves.dH_nn *= 0.5
    waves.dH_nn += waves.dH_nn.T.copy()
    pt_ng[:] = np.linalg.inv(A_nn.T) @ pt_ng


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
