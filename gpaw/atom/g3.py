from scipy.special import genlaguerre
import numpy as np
from gpaw.atom.aeatom import Channel
from gpaw.atom.generator2 import PAWWaves


def pseudize(waves: PAWWaves,
             channel: Channel,
             vtr_g: np.ndarray) -> None:
    l = waves.l
    N = len(waves.e_n)
    rgd = waves.rgd
    r_g = rgd.r_g
    print(l, N, waves.e_n)
    # for n, f in enumerate(waves.f_n):
    #     if f == 111110.0:
    #         waves.e_n[n] = channel.e_n[5]
    #         waves.phi_ng[n] = channel.phi_ng[5]
    phi_ng = waves.phi_ng = np.array(waves.phi_ng)
    phit_ng = waves.phit_ng = np.empty((N, len(r_g)))
    pt_ng = waves.pt_ng = np.empty((N, len(r_g)))
    rc = waves.rcut
    gc = rgd.ceil(rc)
    alpha = 3.0 * (1.5 / rc)**2
    h0_g = howf(alpha, r_g, l, 0, rc)
    e = waves.e_n[0]
    phit_g, c = solve(channel, rgd, vtr_g, gc, e, h0_g, phi_ng[0])
    print(c)
    h0_g *= c
    phit_ng[0] = phit_g
    pt_ng[0] = h0_g
    if N == 2:
        h1_g = h0_g * r_g**2
        p0 = rgd.integrate(phit_g * h0_g)
        p1 = rgd.integrate(phit_g * h1_g)
        h1_g = h0_g - p0 / p1 * h1_g
        e = waves.e_n[1]
        phit0_g, c0 = solve(channel, rgd, vtr_g, gc, e, h0_g, phi_ng[1])
        phit1_g, c1 = solve(channel, rgd, vtr_g, gc, e, h1_g, phi_ng[1])
        h0_g *= c0
        h1_g *= c1
        p0 = rgd.integrate(pt_ng[0] * phit0_g)
        p1 = rgd.integrate(pt_ng[0] * phit1_g)
        x = p1 / (p1 - p0)
        phit_ng[1] = x * phit0_g + (1 - x) * phit1_g
        h1_g = x * h0_g + (1 - x) * h1_g
        pt_ng[1] = h1_g
    elif N > 2:
        1 / 0
    waves.nt_g = waves.f_n @ phit_ng**2 / 4 / np.pi
    waves.dS_nn = (rgd.integrate(phi_ng[:, None] * phi_ng) -
                   rgd.integrate(phit_ng[:, None] * phit_ng)) / (4 * np.pi)
    waves.Q = waves.f_n @ waves.dS_nn.diagonal()
    print(waves.dS_nn, waves.Q, waves.e_n)
    A_nn = rgd.integrate(phit_ng[:, None] * pt_ng) / (4 * np.pi)
    print(A_nn)
    waves.dH_nn = waves.dS_nn @ np.diag(waves.e_n) - A_nn
    waves.dH_nn *= 0.5
    waves.dH_nn += waves.dH_nn.T.copy()
    print(waves.dH_nn)
    pt_ng[:] = np.linalg.inv(A_nn.T) @ pt_ng


def solve(channel, rgd, vtr_g, gc, e, h_g, phi_g):
    u0_g = rgd.zeros()
    u1_g = rgd.zeros()
    channel.integrate_outwards(
        u0_g, rgd, vtr_g, gc + 3, e,
        scalar_relativistic=False)[1]
    channel.integrate_outwards(
        u1_g, rgd, vtr_g, gc + 3, e,
        scalar_relativistic=False, pt_g=h_g)[1]
    r_g = rgd.r_g
    c0, c1 = np.linalg.solve(
        [[u0_g[gc], u1_g[gc]],
         [u0_g[gc + 1], u1_g[gc + 1]]],
        phi_g[gc:gc + 2] * r_g[gc:gc + 2])
    phit_g = c0 * u0_g + c1 * u1_g
    phit_g[1:] /= r_g[1:]
    phit_g[0] = phit_g[1]
    phit_g[gc:] = phi_g[gc:]
    return phit_g, c1


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


PARAMS = {
    'H': ('1s,s,P', 0.9),
    'He': ('1s,s,P', 1.5),
    'Li': ('2s,s,2p,D', 2.1),
    'Be': ('2s,s,2p,D', 1.5),
    'B': ('2s,s,2p,p,d,F', 1.2),
    'C': ('2s,s,2p,p,d,F', 1.2),
    'Ne': ('2s,s,2p,p,d,F', 1.8),
    'Na': ('2s,3s,2p,3p,d,F', 2.3),
    'Al': ('3s,s,3p,p,d,F', 2.1),
    'P': ('3s,s,3p,p,d,F', 1.7),
    'K': ('3s,4s,3p,4p,d,d,F', 2.1),
    'Sc': ('3s,4s,3p,4p,3d,d,F', 2.3),
    'Cu': ('4s,s,4p,p,3d,d,F', 2.1),
    'As': ('4s,s,4p,p,d,F', 2.0)}


def create(symbol):
    from ase.data import chemical_symbols, atomic_numbers
    from gpaw.atom.generator2 import generate
    Z = atomic_numbers[symbol]
    s = symbol
    while s not in PARAMS:
        Z -= 1
        s = chemical_symbols[Z]
    projectors, rc = PARAMS[s]
    gen = generate(
        symbol,
        None,
        projectors,
        [rc],
        rc * 0.9,
        None,
        5,
        xc='PBE',
        pseudize=('harm', -1))
    gen.check_all()
    setup = gen.make_paw_setup()
    setup.write_xml()


def create_all():
    import os
    from pathlib import Path
    from ase.data import chemical_symbols
    for symbol in chemical_symbols[1:37]:
        Path(symbol).mkdir()
        os.chdir(symbol)
        create(symbol)
        os.chdir('..')


def eos(name, symbol):
    from gpaw.utilities.acwf import run_eos_calculation
    from gpaw import setup_paths
    setup_paths[:1] = ['.']
    run_eos_calculation(name, symbol)


def workflow():
    from pathlib import Path
    from myqueue.workflow import run
    from gpaw.utilities.acwf import acwf_structures
    symbol = Path().resolve().name
    for name in acwf_structures:
        run(function=eos, args=[name, symbol], cores=24, name=name)


def hmm():
    import matplotlib.pyplot as plt
    r = np.linspace(0, 4, 101)
    for l in range(3):
        for n in range(2):
            p = howf(3.0, r, l=l, n=n)
            plt.plot(r, p * r**l)
            p = howf(3.0, r, l=l, n=n, rc=1.5)
            plt.plot(r, p * r**l)
    plt.show()


if __name__ == '__main__':
    create_all()
