import numpy as np
from ase import Atoms
from ase.units import Bohr

from gpaw.core import PWDesc, pw_to_grid
from gpaw.dft import ExtensionInput
from gpaw.extensions import Extension
from gpaw.fd_operators import Gradient
from gpaw.new.builder import DFTComponentsBuilder
from gpaw.new.c import add_to_density
from gpaw.new.poisson import PoissonSolver, PoissonSolverWrapper
from gpaw.solvation.cavity import Cavity
from gpaw.solvation.dielectric import Dielectric
from gpaw.solvation.interactions import Interaction
from gpaw.solvation.poisson import WeightedFDPoissonSolver


class Solvation(ExtensionInput):
    name = 'solvation'

    def __init__(self, cavity, dielectric, interactions=None,
                 dipolelayer=False, zero_vacuum=False):
        self.cavity = Cavity.from_dict(cavity)
        self.dielectric = Dielectric.from_dict(dielectric)
        self.interactions = [Interaction.from_dict(i)
                             for i in interactions or []]
        self.dipolelayer = dipolelayer
        self.zero_vacuum = zero_vacuum

    def todict(self):
        return {'cavity': self.cavity.todict(),
                'dielectric': self.dielectric.todict(),
                'interactions': [
                    {'name': i.__class__.__name__, **i.todict()}
                    for i in self.interactions],
                'dipolelayer': self.dipolelayer,
                'zero_vacuum': self.zero_vacuum}

    def build(self, builder: DFTComponentsBuilder):
        if builder.mode == 'pw':
            bad = [type(o).__name__ for o in (self.cavity, *self.interactions)
                   if o.depends_on_el_density]
            if bad:
                raise NotImplementedError(
                    'Plane-wave solvation does not currently support '
                    f'density-dependent {", ".join(bad)}; use fd mode.')
        elif self.dipolelayer or self.zero_vacuum:
            raise NotImplementedError(
                'dipolelayer and zero_vacuum are only implemented for '
                "plane-wave solvation (mode='pw'), not fd mode.")
        return SolvationExtension(
            cavity=self.cavity,
            dielectric=self.dielectric,
            interactions=self.interactions,
            dipolelayer=self.dipolelayer,
            zero_vacuum=self.zero_vacuum,
            setups=builder.setups,
            grid=builder.fine_grid,
            relpos_ac=builder.relpos_ac,
            log=builder.log,
            comm=builder.communicators['w'])


class SolvationExtension(Extension):
    name = 'solvation'

    def __init__(self,
                 *,
                 cavity,
                 dielectric,
                 interactions=None,
                 dipolelayer=False,
                 zero_vacuum=False,
                 setups, grid, relpos_ac, log, comm):
        self.cavity = cavity
        self.dielectric = dielectric
        self.interactions = interactions or []
        self.dipolelayer = dipolelayer
        self.zero_vacuum = zero_vacuum
        finegd = grid._gd
        self.grid = grid
        self.comm = comm
        self.cavity.set_grid_descriptor(finegd)
        self.dielectric.set_grid_descriptor(finegd)
        for ia in self.interactions:
            ia.set_grid_descriptor(finegd)
        self.cavity.allocate()
        self.dielectric.allocate()
        for ia in self.interactions:
            ia.allocate()
        self.atoms = Atoms([setup.symbol for setup in setups],
                           scaled_positions=relpos_ac,
                           cell=grid.cell * Bohr,
                           pbc=grid.pbc)
        self._atoms_moved = True  # force initial cavity build
        self.cavity_changed = True
        self.cavity.update_atoms(self.atoms, log)
        for ia in self.interactions:
            ia.update_atoms(self.atoms, log)
        self.grad_v = [Gradient(grid, v, 1.0, n=3) for v in range(3)]
        self.vt_ia_r = grid.empty()  # self.finegd.zeros()
        self.interaction_energies = {}

    def move_atoms(self, relpos_ac):
        """Point the cavity at the new geometry after the atoms move."""
        self.atoms.set_scaled_positions(relpos_ac)
        self._atoms_moved = True

    def get_energy_contributions(self) -> dict[str, float]:
        """Per-interaction energies, keyed by e.g. 'solvation_surf'."""
        return {f'solvation_{subscript}': e
                for subscript, e in self.interaction_energies.items()}

    def create_poisson_solver(self,
                              grid,
                              pw,
                              *,
                              charge,
                              xp) -> PoissonSolver:
        if isinstance(pw, PWDesc):
            from gpaw.new.pw.poisson import ConjugateGradientPoissonSolver
            return ConjugateGradientPoissonSolver(
                pw, grid, self.dielectric,
                dipolelayer=self.dipolelayer,
                zero_vacuum=self.zero_vacuum)

        psolver = WeightedFDPoissonSolver()
        psolver.set_dielectric(self.dielectric)
        psolver.set_grid_descriptor(grid._gd)
        return PoissonSolverWrapper(psolver)

    def update1pw(self, nt_g, kin_en_using_band=True):
        if self.cavity.depends_on_el_density:
            nt_r = pw_to_grid(nt_g, self.grid)
        else:
            # Density is unused, so skip the IFFT computation.
            nt_r = None
        self.update1(nt_r, kin_en_using_band)

    def update1(self, nt_r, kin_en_using_band=True):
        density = DensityWrapper(nt_r)
        atoms = self.atoms if self._atoms_moved else None
        self.cavity_changed = self.cavity.update(atoms, density)
        if self.cavity_changed:
            self.cavity.update_vol_surf()
            self.dielectric.update(self.cavity)

    def update2pw(self, nt_g, vHt_h, vt_sr):
        if self.cavity.depends_on_el_density:
            nt_r = pw_to_grid(nt_g, self.grid)
            vHt_r = pw_to_grid(vHt_h, self.grid)
        else:
            # Density/potential unused, so skip the IFFT computation.
            nt_r = None
            vHt_r = None
        self.update2(nt_r, vHt_r, vt_sr)

    def update2(self, nt_r, vHt_r, vt_sr):
        if self.cavity.depends_on_el_density:
            del_g_del_n_g = self.cavity.del_g_del_n_g
            del_eps_del_g_g = self.dielectric.del_eps_del_g_g
            Veps = -1 / (8 * np.pi) * del_eps_del_g_g * del_g_del_n_g
            Veps *= grad_squared(vHt_r, self.grad_v).data
            for vt_r in vt_sr.data:
                vt_r += Veps

        density = DensityWrapper(nt_r)
        ia_changed = [
            ia.update(
                self.atoms if self._atoms_moved else None,
                density,
                self.cavity if self.cavity_changed else None)
            for ia in self.interactions]
        if any(ia_changed):
            self.vt_ia_r.data.fill(.0)
            for ia in self.interactions:
                if ia.depends_on_el_density:
                    self.vt_ia_r.data += ia.delta_E_delta_n_g
                if self.cavity.depends_on_el_density:
                    self.vt_ia_r.data += (ia.delta_E_delta_g_g *
                                          self.cavity.del_g_del_n_g)
        if len(self.interactions) > 0:
            for vt_r in vt_sr.data:
                vt_r += self.vt_ia_r.data
        Eias = np.array([ia.E for ia in self.interactions])
        self.grid.comm.sum(Eias)
        self.interaction_energies = {ia.subscript: E
                                     for ia, E in zip(self.interactions, Eias)}

        self.cavity.communicate_vol_surf(self.comm)

        self._atoms_moved = False

    def force_contribution(self, nt_r, vHt_r):
        if isinstance(vHt_r.desc, PWDesc):
            # vHt_r (and nt_r) arrive as PWArray in plane-wave mode and are
            # reconstructed onto the fine grid.
            vHt_r = pw_to_grid(vHt_r, self.grid)
            nt_r = None  # not implemented for density-dependent cavities

        F_av = np.zeros((len(self.atoms), 3))
        add_el_force_correction(
            nt_r, vHt_r, self.grad_v, self.cavity, self.dielectric, F_av)

        density = DensityWrapper(nt_r)

        for ia in self.interactions:
            if self.cavity.depends_on_atomic_positions:
                delta_E_delta_g_r = self.grid.from_data(
                    ia.delta_E_delta_g_g)
                for a, F_v in enumerate(F_av):
                    del_g_del_r_vg = self.grid.from_data(
                        self.cavity.get_del_r_vg(a, density))
                    F_v -= delta_E_delta_g_r.integrate(del_g_del_r_vg,
                                                       skip_sum=True)

            if ia.depends_on_atomic_positions:
                for a, F_v in enumerate(F_av):
                    del_E_del_r_vr = self.grid.from_data(
                        ia.get_del_r_vg(a, density))
                    F_v -= del_E_del_r_vr.integrate(skip_sum=True)

        self.grid.comm.sum(F_av)
        return F_av


def add_el_force_correction(nt_r, vHt_r, grad_v, cavity, dielectric, F_av):
    if not cavity.depends_on_atomic_positions:
        return

    fixed_r = grad_squared(vHt_r, grad_v)  # XXX grad_vHt_g inexact in bmgs
    fixed_r.data *= 1 / (8 * np.pi) * dielectric.del_eps_del_g_g

    density = DensityWrapper(nt_r)

    for a, F_v in enumerate(F_av):
        del_g_del_r_vr = fixed_r.desc.from_data(
            cavity.get_del_r_vg(a, density))
        F_v += fixed_r.integrate(del_g_del_r_vr, skip_sum=True)


class DensityWrapper:
    def __init__(self, nt_r):
        self.nt_g = nt_r.data if nt_r is not None else None


def grad_squared(a_r, grad_v):
    tmp_r = a_r.new()
    b_r = a_r.desc.zeros()
    for grad in grad_v:
        grad(a_r, tmp_r)
        add_to_density(1, tmp_r.data, b_r.data)
    return b_r
