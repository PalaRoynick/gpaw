from scipy.special import genlaguerre
import numpy as np
from gpaw.atom.aeatom import Channel
from gpaw.atom.generator2 import PAWWaves


def pseudiz0(waves: PAWWaves,
             channel: Channel,
             vtr_g: np.ndarray) -> None:
    l = waves.l
    N = len(waves.e_n)
    rgd = waves.rgd
    r_g = rgd.r_g
    phi_ng = waves.phi_ng = np.array(waves.phi_ng)
    phit_ng = waves.phit_ng = np.empty((N, len(r_g)))
    pt_ng = waves.pt_ng = np.empty((N, len(r_g)))
    rc = waves.rcut
    gc = rgd.ceil(rc)
    alpha = 2.0 * (1.5 / rc)**2
    h0_g = howf(alpha, r_g, l, 0, rc)
    e = waves.e_n[0]
    phit_g, c = solve(channel, rgd, vtr_g, gc, e, h0_g, phi_ng[0])
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
    A_nn = rgd.integrate(phit_ng[:, None] * pt_ng) / (4 * np.pi)
    waves.dH_nn = waves.dS_nn @ np.diag(waves.e_n) - A_nn
    waves.dH_nn *= 0.5
    waves.dH_nn += waves.dH_nn.T.copy()
    pt_ng[:] = np.linalg.inv(A_nn.T) @ pt_ng


# gpaw -T dataset Be -p -zharm,6 -P 2s,2p,D -lspd -r 2.2
def pseudize(waves: PAWWaves,
             channel: Channel,
             vtr_g: np.ndarray) -> None:
    l = waves.l
    N = len(waves.e_n)
    rgd = waves.rgd
    r_g = rgd.r_g
    phi_ng = waves.phi_ng = np.array(waves.phi_ng)
    phit_ng = waves.phit_ng = np.empty((N, len(r_g)))
    pt_ng = waves.pt_ng = np.empty((N, len(r_g)))
    rc = waves.rcut
    gc = rgd.ceil(rc)
    e = waves.e_n[0]
    phit_g, pt_g = first_partial_wave(rgd, phi_ng[0], rc, l, vtr_g, e)
    phit_ng[0] = phit_g
    pt_ng[0] = pt_g
    if N == 2:
        e = waves.e_n[1]
        ha_g = pt_g.copy()
        hb_g = pt_g * r_g**2
        phit2a_g, c2a = solve(channel, rgd, vtr_g, gc, e, ha_g, phi_ng[1])
        phit2b_g, c2b = solve(channel, rgd, vtr_g, gc, e, hb_g, phi_ng[1])
        ha_g *= c2a
        hb_g *= c2b
        p0 = rgd.integrate(pt_ng[0] * phit2a_g)
        p1 = rgd.integrate(pt_ng[0] * phit2b_g)
        x = p1 / (p1 - p0)
        phit_ng[1] = x * phit2a_g + (1 - x) * phit2b_g
        h1_g = x * ha_g + (1 - x) * hb_g
        pt_ng[1] = h1_g
    elif N > 2:
        1 / 0
    waves.nt_g = waves.f_n @ phit_ng**2 / 4 / np.pi
    waves.dS_nn = (rgd.integrate(phi_ng[:, None] * phi_ng) -
                   rgd.integrate(phit_ng[:, None] * phit_ng)) / (4 * np.pi)
    waves.Q = waves.f_n @ waves.dS_nn.diagonal()
    A_nn = rgd.integrate(phit_ng[:, None] * pt_ng) / (4 * np.pi)
    waves.dH_nn = waves.dS_nn @ np.diag(waves.e_n) - A_nn
    waves.dH_nn *= 0.5
    waves.dH_nn += waves.dH_nn.T.copy()
    print(l, N, A_nn, waves.dH_nn, waves.dS_nn)
    pt_ng[:] = np.linalg.inv(A_nn.T) @ pt_ng


def first_partial_wave(rgd, phi_g, rcut, l, vtr_g, eig):
    print(l, rcut, eig)
    gcut = rgd.floor(rcut)
    r_g = rgd.r_g
    aphi_g = abs(phi_g)
    aphi_g[1:] /= r_g[1:]**l
    aphi_g[0] = aphi_g[1]
    print(aphi_g.min())
    logphit_g, _ = rgd.pseudize(np.log(aphi_g), gcut, 0, 5)
    sign = np.sign(phi_g[gcut])
    a_g = sign * np.exp(logphit_g)
    phit_g = a_g * r_g**l
    dadg_g, d2adg2_g = rgd.zeros(2)
    # a_g[1:] = phit_g[1:] / r_g[1:]**l
    # a_g[0] = c0
    dadg_g[1:-1] = 0.5 * (a_g[2:] - a_g[:-2])
    d2adg2_g[1:-1] = a_g[2:] - 2 * a_g[1:-1] + a_g[:-2]
    # print(a_g[:5])
    # print(dadg_g[:5])
    # print(d2adg2_g[:5])
    if d2adg2_g[1] * sign > 0.0:
        print('BAD CURVATURE:', l)
    q_g = (vtr_g - eig * r_g) * phit_g
    dgdr_g = 1 / rgd.dr_g
    d2gdr2_g = rgd.d2gdr2()
    q_g -= 0.5 * r_g**l * (
        (2 * (l + 1) * dgdr_g + r_g * d2gdr2_g) * dadg_g +
        r_g * d2adg2_g * dgdr_g**2)
    q_g[gcut:] = 0
    rgd.cut(q_g, rcut)
    q_g[1:] /= r_g[1:]
    if l == 0:
        q_g[0] = q_g[1]
    if l == 111111110:
        rgd.plot(phit_g)
        rgd.plot(q_g, show=1)
    return phit_g, q_g


def second_partial_wave(rgd, phi_g, rcut, l, vtr_g, eig):
    gcut = rgd.floor(rcut)
    phit_g, c0 = rgd.pseudize(phi_g, gcut, l, 6)
    a_g, dadg_g, d2adg2_g = rgd.zeros(3)
    r_g = rgd.r_g
    a_g[1:] = phit_g[1:] / r_g[1:]**l
    a_g[0] = c0
    dadg_g[1:-1] = 0.5 * (a_g[2:] - a_g[:-2])
    d2adg2_g[1:-1] = a_g[2:] - 2 * a_g[1:-1] + a_g[:-2]
    q_g = (vtr_g - eig * r_g) * phit_g
    dgdr_g = 1 / rgd.dr_g
    d2gdr2_g = rgd.d2gdr2()
    q_g -= 0.5 * r_g**l * (
        (2 * (l + 1) * dgdr_g + r_g * d2gdr2_g) * dadg_g +
        r_g * d2adg2_g * dgdr_g**2)
    q_g[gcut:] = 0
    rgd.cut(q_g, rcut)
    q_g[1:] /= r_g[1:]
    if l == 0:
        q_g[0] = q_g[1]
    return phit_g, q_g


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
    'H': ('1s,s,p', 0.9),
    'He': ('1s,s,p', 1.5),
    'Li': ('2s,s,2p,D', 3.0),
    'Li.3': ('1s,2s,2p,D', 1.6),
    'Be': ('2s,s,2p,D', 2.7),
    'B': ('2s,s,2p,p,D', 2.0),
    'C': ('2s,1.0s,2p,p,d', 1.6),
    'C.a': ('2s,1.0s,2p,p,d,F', 1.6),
    'C.b': ('2s,1.0s,2p,p,D', 1.6),
    'C.c': ('2s,1.0s,2p,D', 1.6),
    'N': ('2s,s,2p,p,d', 1.6),
    'O': ('2s,s,2p,p,d', 1.6),
    'F': ('2s,s,2p,p,d', 1.6),
    'Ne': ('2s,s,2p,p,d,F', 1.8),
    'Na': ('2s,3s,2p,3p,d,F', 2.3),
    'Al': ('3s,s,3p,p,d,F', 2.1),
    'P': ('3s,s,3p,p,d,F', 1.7),
    'K': ('3s,4s,3p,4p,d,d,F', 2.1),
    'Sc': ('3s,4s,3p,4p,3d,d,F', 2.3),
    'Cu': ('4s,s,4p,p,3d,d,F', 2.1),
    'As': ('4s,s,4p,p,d,F', 2.0)}


def create(symbol):
    from gpaw.atom.generator2 import generate
    projectors, rc = PARAMS[symbol]
    gen = generate(
        symbol.split('.')[0],
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
    from contextlib import chdir
    from pathlib import Path
    for symbol in PARAMS:
        dir = Path(symbol)
        s = symbol.split('.')[0]
        if (dir / f'{s}.PBE').is_file():
            continue
        dir.mkdir(exist_ok=True)
        with chdir(dir):
            create(symbol)


def eos(structure, symbol):
    from gpaw.utilities.acwf import work
    from gpaw import setup_paths
    setup_paths[:1] = ['.']
    work(structure, symbol, backwards_compatible=False)


def workflow():
    from pathlib import Path
    # from myqueue.workflow import run
    from gpaw.utilities.acwf import acwf_structures
    symbol = Path().resolve().name.split('.')[0]
    for name in acwf_structures:
        # run(function=eos, args=[name, symbol], cores=24, name=name)
        eos(name, symbol)
    ecuts(symbol)


def ecuts(paw_pot_name):
    from gpaw.utilities.acwf import reference_structure
    from gpaw.dft import DFT, PW, MonkhorstPack, Parameters
    from gpaw import setup_paths
    from pathlib import Path
    setup_paths[:1] = ['.']
    symbol = paw_pot_name.split('.')[0]
    atoms = reference_structure(symbol, 'FCC')
    dft = None
    results = []
    for ecut in range(300, 1100, 100):
        if dft is None:
            dft = DFT(
                atoms,
                xc='PBE',
                mode=PW(ecut),
                txt=f'{symbol}.txt',
                kpts=MonkhorstPack(density=3.0),
                experimental={'backwards_compatible': False})
            dft.converge()
        else:
            params = Parameters(**dict(dft.params.todict(), mode=PW(ecut)))
            dft = dft.new(atoms, params, log=dft.log)
            dft.converge()
        e = dft.calculate_energy()
        results.append((ecut, e))
    if dft.comm.rank == 0:
        Path(f'{symbol}-ecuts.json').write(
            '[' +
            ',\n '.join(f'[{ecut:6.1f}, {e}]' for ecut, e in results) +
            ']\n')
    return results


def get_ecut(ecuts, energies, de):
    y = abs(energies - energies[-1])
    i = len(energies) - 2
    while y[i] < de:
        i -= 1
    return (ecuts[i] * (de - y[i + 1]) +
            ecuts[i + 1] * (y[i] - de)) / (y[i] - y[i + 1])


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
    # convergence_check('H')
