import numpy as np
import pytest
from ase import Atoms
from ase.units import Bohr, Ha

from gpaw import GPAW, FermiDirac
from gpaw.new.constraints import SpinDirectionConstraint
from gpaw.new.sjm import SJM
from gpaw.solvation.sjm import SJM as OldSJM


@pytest.mark.parametrize('mode', ['pw', 'fd'])
@pytest.mark.parametrize('gpaw_new', [False, True])
def test_h(gpaw_new, mode, in_tmp_dir):
    if mode == 'pw' and not gpaw_new:
        pytest.skip('PW-mode not implemented for old GPAW')

    a = 1.4
    atoms = Atoms('H', cell=[a, a, 11.0], pbc=(1, 1, 0))
    atoms.positions[0, 2] = 4.0

    k = 2
    params = dict(
        mode=mode,
        kpts=(k, k, 1),
        occupations=FermiDirac(0.2))
    sjm = {'target_potential': 7.5,
           'excess_electrons': -0.0045,
           'jelliumregion': {'top': 9.0, 'bottom': 7.0},
           'tol': 0.01}
    solvation = dict(cavity=NoCavity(), dielectric=Vacuum(), interactions=[])

    if gpaw_new:
        atoms.calc = GPAW(extensions=[SJM(**sjm, **solvation)], **params)
        energy = atoms.get_potential_energy()
        pot = -atoms.calc.get_fermi_level()
    else:
        atoms.calc = OldSJM(**solvation, sj=sjm, **params)
        energy = atoms.get_potential_energy()
        pot = atoms.calc.get_electrode_potential()

    assert pot == pytest.approx(7.5, abs=0.01)

    if gpaw_new:
        check_grand_canonical_results(atoms, energy)

    atoms.write('h.traj')
    if gpaw_new:
        atoms.calc.write('h.gpw')
        calc = GPAW('h.gpw', object_hooks={'extensions': lambda exts: []})
        # The grand-potential energies must survive a gpw round-trip:
        for name in ['energy', 'free_energy', 'excess_electrons',
                     'electrode_potential']:
            assert calc.results[name] == pytest.approx(
                atoms.calc.results[name], abs=1e-10)

    if 0:
        v = atoms.calc.get_electrostatic_potential()
        import matplotlib.pyplot as plt
        plt.plot(np.linspace(0, 11, v.shape[2], 0), v[0, 0])
        plt.show()


def check_grand_canonical_results(atoms, energy):
    """The grand potential - not the canonical energy - goes to ASE."""
    calc = atoms.calc
    sjm = calc.dft.sjm
    results = calc.results

    assert results['excess_electrons'] == pytest.approx(sjm.charge)
    assert results['electrode_potential'] == pytest.approx(
        sjm.get_electrode_potential())

    # Omega = E - N mu, with mu = -workfunction = fermi level:
    mu_N = sjm.charge * calc.get_fermi_level()
    canonical = calc.dft.energies.total_extrapolated * Ha
    canonical_free = calc.dft.energies.total_free * Ha
    assert energy == pytest.approx(canonical - mu_N)
    assert results['energy'] == pytest.approx(canonical - mu_N)
    assert results['free_energy'] == pytest.approx(canonical_free - mu_N)
    # The excess charge is small but non-zero, so the two must differ:
    assert abs(energy - canonical) > 1e-6

    sjm.write_sjm_traces(props=('potential', 'background_charge'))
    z, v = np.loadtxt('sjm_traces/potential.txt').T
    assert len(z) == len(v) and 0.0 <= z[0] < z[-1] < atoms.cell[2, 2]
    # The potential trace is referenced to the Fermi level, so it equals the
    # work function in the vacuum where the Poisson solver zeroes it.  Both
    # modes zero it at the last z index -- the top of the cell, above the
    # jellium -- so this is one-sided:
    pot = -calc.get_fermi_level()
    assert abs(v[-1] - pot) < 0.05
    z, rho = np.loadtxt('sjm_traces/background_charge.txt').T
    # The counter charge (in e/Bohr^3) integrates to minus the excess
    # electrons; z and the cell are in Angstrom:
    dV = (z[1] - z[0]) * abs(np.linalg.det(atoms.cell[:2, :2])) / Bohr**3
    assert (rho.sum() * dV) == pytest.approx(-sjm.charge, abs=1e-6)


def test_implemented_properties():
    """The extra SJM properties are declared by the extension itself."""
    extra = ['excess_electrons', 'electrode_potential']
    solvation = dict(cavity=NoCavity(), dielectric=Vacuum(), interactions=[])

    # ASE may ask before anything has been calculated:
    calc = GPAW(mode='fd', txt=None,
                extensions=[SJM(target_potential=7.5, **solvation)])
    assert all(name in calc.implemented_properties for name in extra)

    calc = GPAW(mode='fd', txt=None)
    assert not any(name in calc.implemented_properties for name in extra)

    # An extension that adds no properties of its own - and that is an
    # Extension rather than an ExtensionInput:
    calc = GPAW(mode='fd', txt=None,
                extensions=[SpinDirectionConstraint({0: [1., 0., 0.]}, 2.0)])
    assert not any(name in calc.implemented_properties for name in extra)


class NoCavity:
    depends_on_el_density = False

    def set_grid_descriptor(self, gd):
        pass

    def allocate(self):
        pass

    def update_atoms(self, atoms, log):
        pass

    def update(self, atoms, density):
        pass

    def communicate_vol_surf(self, comm):
        pass

    def summary(self, log):
        pass

    def todict(self):
        return {}


class Vacuum:
    def set_grid_descriptor(self, gd):
        self.eps_gradeps = [gd.zeros() for _ in range(4)]
        self.eps_gradeps[0][:] = 1.0

    def allocate(self):
        pass

    def todict(self):
        return {}


if __name__ == '__main__':
    import sys
    test_h(int(sys.argv[1]), sys.argv[2], 1)
