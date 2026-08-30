from __future__ import annotations

import os
import textwrap
import warnings

import numpy as np
from ase.units import Bohr, Ha

from gpaw import ConvergenceError
from gpaw.core import PWArray, PWDesc, UGArray
from gpaw.mpi import broadcast_float
from gpaw.new.builder import DFTComponentsBuilder
from gpaw.extensions import Extension
from gpaw.extensions.jellium import (FixedPotentialJelliumExtension,
                                     JelliumExtension)
from gpaw.new.builder import DFTComponentsBuilder
from gpaw.new.poisson import PoissonSolverWrapper
from gpaw.new.pw.poisson import PWPoissonSolver
from gpaw.new.solvation import Solvation, SolvationExtension

#: Default top of the jellium region: 1 Å below the top of the unit cell.
DEFAULT_TOP = -1.0
#: Default bottom of the jellium region: 3 Å above the highest atom.
DEFAULT_BOTTOM = -3.0


class PotentialConvergenceError(ConvergenceError):
    """Raised if the electrode potential did not equilibrate."""


class JelliumRegion:
    """Shape of the jellium counter-charge region.

    The region is a slab perpendicular to the z axis, specified by at most
    two of ``top``, ``bottom`` and ``thickness`` (the third one follows).

    Parameters
    ----------
    top:
        Upper boundary of the region.  A positive value is the z coordinate
        in Å; a negative value is the distance below the upper cell
        boundary.  Default: -1 Å.
    bottom:
        Lower boundary of the region.  A positive value is the z coordinate
        in Å; a negative value is the distance above the highest atom, in
        which case the region follows the atoms during a relaxation unless
        ``fix_bottom`` is used.  Default: -3 Å.
    thickness:
        Thickness of the region in Å.  Can only be given together with one
        of ``top`` and ``bottom``.
    fix_bottom:
        Freeze a relative (negative) lower limit at the value it has for
        the first geometry, so that it does not adapt to a change in the
        atomic positions during a relaxation.
    """

    def __init__(self,
                 top: float | None = None,  # Å
                 bottom: float | str | None = None,  # Å
                 thickness: float | None = None,  # Å
                 fix_bottom: bool = False):
        if bottom == 'cavity_like':
            raise NotImplementedError(
                'A cavity-like jellium region is not implemented in this '
                'version of SJM.  Use gpaw.solvation.sjm instead.')

        specified = [top is not None,
                     bottom is not None,
                     thickness is not None]
        if all(specified):
            raise ValueError(
                'The jellium region has been overspecified: give at most two '
                "of 'top', 'bottom' and 'thickness'.")
        if not any(specified):
            top, bottom = DEFAULT_TOP, DEFAULT_BOTTOM
        elif specified == [True, False, False]:
            bottom = DEFAULT_BOTTOM
        elif specified == [False, True, False]:
            top = DEFAULT_TOP
        elif specified == [False, False, True]:
            bottom = DEFAULT_BOTTOM

        self.top = top
        self.bottom = bottom
        self.thickness = thickness
        self.fix_bottom = fix_bottom
        self._frozen_bottom: float | None = None

    def __repr__(self):
        return (f'JelliumRegion(top={self.top}, bottom={self.bottom}, '
                f'thickness={self.thickness}, fix_bottom={self.fix_bottom})')

    @property
    def follows_atoms(self) -> bool:
        """Does the region move when the atoms move?"""
        return (self.bottom is not None and self.bottom < 0.0 and
                not self.fix_bottom)

    def limits(self,
               height: float,  # Å
               zmax: float,  # Å
               ) -> tuple[float, float]:
        """Lower and upper boundary in Å.

        ``height`` is the height of the unit cell and ``zmax`` the z
        coordinate of the highest atom.
        """
        top, bottom, thickness = self.top, self.bottom, self.thickness

        if top is not None and top < 0.0:
            top = height + top
        if bottom is not None and bottom < 0.0:
            if self._frozen_bottom is not None:
                bottom = self._frozen_bottom
            else:
                bottom = zmax - bottom
                if self.fix_bottom:
                    self._frozen_bottom = bottom

        if top is None:
            top = bottom + thickness
        if bottom is None:
            bottom = top - thickness

        if top > height:
            raise ValueError(
                f'The upper limit of the jellium region ({top:.3f} Å) lies '
                f'outside of the unit cell ({height:.3f} Å).  If you did not '
                'set it manually, increase your unit cell size or translate '
                'the atomic system down along the z axis.')
        if bottom >= top:
            raise ValueError(
                f'Your jellium region has a bottom at {bottom:.3f} Å, which '
                f'is not below the top at {top:.3f} Å.')
        return bottom, top

    def mask(self, grid, relpos_ac) -> tuple[UGArray, float, float]:
        """Build the 0/1 mask of the region, plus its limits in Å."""
        height = grid.cell_cv[2, 2] * Bohr
        bottom, top = self.limits(height, relpos_ac[:, 2].max() * height)
        z_r = grid.xyz()[:, :, :, 2]
        mask_r = grid.empty()
        mask_r.data[:] = np.logical_and(z_r > bottom / Bohr,
                                        z_r < top / Bohr)
        return mask_r, bottom, top


class SJM(Solvation):
    """Solvated jellium method.

    Electronically grand-canonical calculations, typically for simulating
    electrochemical interfaces.  A jellium counter charge is placed in the
    implicit solvent above the slab; the number of excess electrons can
    either be fixed (constant-charge mode) or adjusted until the electrode
    potential matches ``target_potential`` (constant-potential mode).

    Parameters
    ----------
    cavity, dielectric, interactions:
        Solvation model, see :class:`gpaw.new.solvation.Solvation`.
    jelliumregion:
        Shape of the counter-charge region: a dict with the keys ``'top'``,
        ``'bottom'``, ``'thickness'`` and ``'fix_bottom'``, see
        :class:`JelliumRegion`.
    target_potential:
        Electrode potential to equilibrate to, in V on the vacuum scale
        (expressed as a work function).  ``None`` (the default) gives a
        constant-charge calculation based on ``excess_electrons``.
    excess_electrons:
        Number of excess electrons in the atomic system, and (with opposite
        sign) in the jellium region.  Used as the initial guess in
        constant-potential mode.
    tol:
        Tolerance for the deviation of the target potential, in V.
    max_iters:
        Maximum number of attempts to equilibrate the potential, in
        constant-potential mode.  A :class:`PotentialConvergenceError` is
        raised if the potential is not reached.
    grand_output:
        Write the grand-potential energy (instead of the canonical energy)
        into the results dictionary handed to ASE, and thereby into
        trajectory files.  Default: True.
    """

    name = 'sjm'

    def __init__(self,
                 *,
                 cavity,
                 dielectric,
                 interactions,
                 jelliumregion: dict | None = None,
                 target_potential: float | None = None,  # eV
                 excess_electrons: float = 0.0,
                 tol: float = 0.001,  # eV
                 max_iters: int = 10,
                 grand_output: bool = True):
        super().__init__(cavity, dielectric, interactions,
                         dipolelayer=True, zero_vacuum=True)
        self.jelliumregion = jelliumregion or {}
        self.target_potential = target_potential
        self.excess_electrons = excess_electrons
        self.tol = tol
        self.max_iters = max_iters
        self.grand_output = grand_output

    def build(self, builder: DFTComponentsBuilder) -> SJMExtension:
        solvation = super().build(builder)
        log = builder.log
        self._check_convergence_criteria(builder.params.convergence, log)

        region = JelliumRegion(**self.jelliumregion)
        mask_r, z1, z2 = region.mask(builder.fine_grid, builder.relpos_ac)

        if self.target_potential is None:
            jellium = JelliumExtension(
                mask_r,
                charge=self.excess_electrons,
                pw=builder.interpolation_desc)
        else:
            jellium = FixedPotentialJelliumExtension(
                mask_r,
                workfunction_target=self.target_potential,
                excess_electrons_guess=self.excess_electrons,
                tolerance=self.tol,
                pw=builder.interpolation_desc)

        log('Solvated jellium method (SJM):')
        log(f'  jellium region: {z1:.3f} to {z2:.3f}  # Å')
        if self.target_potential is None:
            log('  mode: constant charge')
            log(f'  excess electrons: {self.excess_electrons:+.5f}')
        else:
            log('  mode: constant potential')
            log(f'  target potential: {self.target_potential:.5f}  # V')
            log(f'  tolerance: {self.tol:.5f}  # V')
            log(f'  max iterations: {self.max_iters}')
            log('  initial guess of excess electrons: '
                f'{self.excess_electrons:+.5f}')

        return SJMExtension(solvation, jellium, region,
                            target_potential=self.target_potential,
                            tol=self.tol,
                            max_iters=self.max_iters,
                            grand_output=self.grand_output,
                            log=log)

    def _check_convergence_criteria(self, convergence, log) -> None:
        if self.target_potential is None:
            return
        wf_tol = (convergence or {}).get('work function')
        if wf_tol is not None and wf_tol >= self.tol:
            msg = ('It appears that your work function convergence criterion '
                   f'({wf_tol:g}) is higher than your desired potential '
                   f'tolerance ({self.tol:g}).  This may lead to issues with '
                   'potential convergence.')
            warnings.warn(msg)
            log(textwrap.fill('Warning: ' + msg) + '\n')

    def todict(self):
        dct = super().todict()
        dct.pop('dipolelayer', None)
        dct.pop('zero_vacuum', None)
        dct.update(
            jelliumregion=self.jelliumregion,
            target_potential=self.target_potential,
            excess_electrons=self.excess_electrons,
            tol=self.tol,
            max_iters=self.max_iters,
            grand_output=self.grand_output)
        return dct


class SJMExtension(Extension):
    name = 'sjm'

    def __init__(self,
                 solvation: SolvationExtension,
                 jellium: JelliumExtension,
                 region: JelliumRegion | None = None,
                 *,
                 target_potential: float | None = None,  # eV
                 tol: float = 0.001,  # eV
                 max_iters: int = 10,
                 grand_output: bool = True,
                 log=None):
        self.solvation = solvation
        self.jellium = jellium
        self.region = region or JelliumRegion()
        self.dielectric = solvation.dielectric
        self.charge = jellium.charge
        self.target_potential = target_potential
        self.tol = tol
        self.max_iters = max_iters
        self.grand_output = grand_output
        self.log = log or (lambda *args, **kwargs: None)

        self.niters = 0
        # Set by post_converged():
        self.fermi_level = np.nan  # Ha
        self.omega_free = np.nan  # Ha
        self.omega_extrapolated = np.nan  # Ha
        self._dft = None

    def __str__(self):
        txt = ('solvated jellium method:\n'
               f'  excess electrons: {self.charge:+.5f}\n')
        if self.target_potential is not None:
            txt += (f'  target potential: {self.target_potential:.5f}  # V\n'
                    f'  tolerance: {self.tol:.5f}  # V\n')
        return txt

    def create_poisson_solver(self, grid, pw, charge, xp):
        if isinstance(pw, PWDesc):
            from gpaw.new.pw.poisson import ConjugateGradientPoissonSolver
            return ConjugateGradientPoissonSolver(
                pw, grid, self.dielectric, dipolelayer=True, zero_vacuum=True)
            # from gpaw.new.sjm import SJMPWPoissonSolver
            # return SJMPWPoissonSolver(pw, environment.dielectric, grid)

        ps = self.solvation.create_poisson_solver(
            grid, pw, charge=charge, xp=xp).solver
        return SJMPoissonSolver(ps, self.solvation.dielectric)

    def post_scf_convergence(self,
                             ibzwfs,
                             nelectrons,
                             occ_calc,
                             mixer,
                             log) -> bool:
        converged = self.jellium.post_scf_convergence(
            ibzwfs, nelectrons, occ_calc, mixer, log)
        self.charge = self.jellium.charge
        if converged:
            return True

        self.niters += 1
        if self.target_potential is not None and self.niters >= self.max_iters:
            raise PotentialConvergenceError(
                self._convergence_error_message(log))
        return False

    def _convergence_error_message(self, log) -> str:
        msg = (f'The electrode potential could not be equilibrated to '
               f'{self.target_potential:.3f} +/- {self.tol:.3f} V in '
               f'{self.max_iters:d} attempts.  This may indicate that your '
               'work function is noisier than your potential tolerance; you '
               'may try tightening the SCF convergence criteria.  The excess '
               'electrons and the resulting potentials are listed below; '
               'plotting them could give you insight into the problem.')
        msg = textwrap.fill(msg) + '\n'
        for charge, fermi_level in getattr(self.jellium, 'history', []):
            msg += f'{charge:+.6f} {-fermi_level * Ha:.6f}\n'
        log(msg, flush=True)
        return msg

    def get_electrode_potential(self) -> float:
        """Electrode potential in V, relative to the vacuum.

        The Poisson solver zeroes the electrostatic potential in the vacuum
        region, so the work function -- and with it the electrode potential
        -- is simply minus the Fermi level.
        """
        return -self.fermi_level * Ha

    def post_converged(self, dft) -> None:
        """Assemble grand-potential energies and write them to log/results."""
        self._dft = dft
        log = dft.log
        (fermi_level,) = dft.ibzwfs.fermi_levels
        self.fermi_level = broadcast_float(float(fermi_level), dft.comm)
        potential = self.get_electrode_potential()

        # Legendre transform: Omega = E - N mu, with mu = -workfunction.
        # Take the canonical energies from dft.energies rather than from
        # dft.results, so that this can be called again after the results
        # have already been replaced by the grand-potential energies.
        e_free = broadcast_float(dft.energies.total_free, dft.comm)
        e_extrapolated = broadcast_float(dft.energies.total_extrapolated,
                                         dft.comm)
        mu_N = self.fermi_level * self.charge
        self.omega_free = e_free - mu_N
        self.omega_extrapolated = e_extrapolated - mu_N

        log('Legendre-transformed energies '
            '(grand potential, Omega = E - N mu):')
        log(f'  N (excess electrons): {self.charge:+14.6f}')
        log(f'  mu (-work function):  {-potential:+14.6f}  # eV')
        log(f'  (Grand) free energy:  {self.omega_free * Ha:+14.6f}')
        log(f'  (Grand) extrapolated: {self.omega_extrapolated * Ha:+14.6f}')
        log()

        dft.results['excess_electrons'] = self.charge
        dft.results['electrode_potential'] = potential / Ha
        if self.grand_output:
            dft.results['free_energy'] = self.omega_free
            dft.results['energy'] = self.omega_extrapolated
            log('Grand-potential energy was written into results.\n')
        else:
            dft.results['free_energy'] = e_free
            dft.results['energy'] = e_extrapolated
            log('Canonical energy was written into results.\n')
        log.fd.flush()

    def write_sjm_traces(self, path='sjm_traces', style='z',
                         props=('potential', 'cavity', 'background_charge')):
        """Write traces of the quantities in `props` to disk.

        The traces are stored within the specified `path`.  The default is to
        save them as vertical traces (style ``'z'``), but they can also be
        saved as cube files (``style='cube'``).

        Note that ``'potential'`` is the electrostatic potential referenced
        to the Fermi level, in eV, and that ``'background_charge'`` is the
        jellium counter-charge density in e/Bohr^3 -- it integrates to minus
        the number of excess electrons.
        """
        dft = self._dft
        if dft is None:
            raise RuntimeError('No results yet - run a calculation before '
                               'writing SJM traces.')
        if style not in {'z', 'cube'}:
            raise ValueError(f'Unknown style: {style!r}')

        comm = self.solvation.comm  # world
        if comm.rank == 0:
            os.makedirs(path, exist_ok=True)
        comm.barrier()

        for prop in props:
            a_r = self._trace_data(prop, dft).gather(broadcast=True)
            if comm.rank == 0:
                if style == 'z':
                    _write_trace_in_z(a_r, os.path.join(path, prop + '.txt'))
                else:
                    import ase.io
                    ase.io.write(os.path.join(path, prop + '.cube'),
                                 dft.atoms, data=a_r.data)
        comm.barrier()

    def _trace_data(self, prop: str, dft) -> UGArray:
        """One of the quantities that write_sjm_traces() can write out."""
        grid = self.solvation.grid
        if prop == 'potential':
            vHt_r = self._electrostatic_potential(dft)
            vHt_r.data[:] = (vHt_r.data - self.fermi_level) * Ha
            return vHt_r
        if prop == 'background_charge':
            charge_r = self.jellium.mask_r.new()
            charge_r.data[:] = -self.charge * self.jellium.mask_r.data
            return charge_r
        if prop == 'cavity':
            try:
                g_g = self.solvation.cavity.g_g
            except AttributeError:
                raise ValueError('This cavity has no g_g attribute - '
                                 'cannot write a cavity trace.')
            return grid.from_data(g_g)
        raise ValueError(f'Unknown property: {prop!r}')

    def _electrostatic_potential(self, dft) -> UGArray:
        """Electrostatic potential on the fine (solvation) grid, in Hartree."""
        grid = self.solvation.grid
        vHt_x = dft.potential.vHt_x
        if isinstance(vHt_x, UGArray):
            return vHt_x.copy()
        vHt_g = vHt_x.gather()
        vHt_r = grid.empty()
        vHt_r.scatter_from(vHt_g.ifft(grid=grid.new(comm=None))
                           if vHt_g is not None else None)
        return vHt_r

    def set(self, **kwargs) -> None:
        """Change SJM parameters without throwing away the wave functions.

        Recognized keywords are ``target_potential``, ``excess_electrons``,
        ``tol``, ``max_iters``, ``grand_output`` and ``jelliumregion``, with
        the same meaning as for :class:`SJM`.  This is the replacement for
        the legacy ``calc.set(sj={...})``; reach it as ``calc.dft.sjm.set()``.

        The density and the wave functions are kept, so the next
        ``get_potential_energy()`` continues the SCF loop from the current
        solution instead of starting over.  When the target potential is
        changed and the slope of potential vs. charge is known, a first step
        in the number of excess electrons is taken right away -- unless
        ``excess_electrons`` is given explicitly, which then wins.
        """
        known = {'target_potential', 'excess_electrons', 'tol', 'max_iters',
                 'grand_output', 'jelliumregion'}
        unknown = set(kwargs) - known
        if unknown:
            raise ValueError(
                f'Unexpected SJM keyword(s): {", ".join(sorted(unknown))}.  '
                f'Allowed are: {", ".join(sorted(known))}.')

        log = self.log
        rerun = False  # does the SCF loop have to run again?

        if 'grand_output' in kwargs:
            self.grand_output = kwargs['grand_output']
            if self._dft is not None and self._dft.results:
                # Cheap: just re-do the Legendre transform.
                self.post_converged(self._dft)
        if 'max_iters' in kwargs:
            self.max_iters = kwargs['max_iters']
        if 'tol' in kwargs:
            self.tol = kwargs['tol']
            if isinstance(self.jellium, FixedPotentialJelliumExtension):
                self.jellium.tolerance = self.tol / Ha
                if abs(self.get_electrode_potential() -
                       self.target_potential) > self.tol:
                    rerun = True

        if 'jelliumregion' in kwargs:
            self.region = JelliumRegion(**(kwargs['jelliumregion'] or {}))
            mask_r, z1, z2 = self.region.mask(self.jellium.mask_r.desc,
                                              self._relpos_ac())
            self.jellium.set_mask(mask_r)
            log(f'Jellium region changed to {z1:.3f} to {z2:.3f}  # Å')
            rerun = True

        if 'target_potential' in kwargs:
            target = kwargs['target_potential']
            if (target is None) != (self.target_potential is None):
                self._switch_mode(target)
            elif target is not None:
                self.jellium.workfunction_target = target / Ha
            self.target_potential = target
            if target is None:
                log('Switched to constant-charge mode with '
                    f'{self.charge:+.5f} excess electrons.')
            else:
                log(f'Target potential changed to {target:.5f} +/- '
                    f'{self.tol:.5f} V.')
                if 'excess_electrons' not in kwargs:
                    self._step_towards(target)
            rerun = True

        if 'excess_electrons' in kwargs:
            self._set_charge(kwargs['excess_electrons'])
            log(f'Excess electrons changed to {self.charge:+.5f}.')
            rerun = True

        if rerun and self._dft is not None:
            # Empty results make ASECalculator.iconverge() run the SCF loop
            # again; the density and the wave functions are kept.
            self._dft.results.clear()
            self.niters = 0

    def _relpos_ac(self):
        if self._dft is None:
            raise RuntimeError('No results yet - run a calculation before '
                               'changing the jellium region.')
        return self._dft.relpos_ac

    def _set_charge(self, charge: float) -> None:
        """Set the number of excess electrons."""
        self.jellium.charge = charge
        self.charge = charge

    def _switch_mode(self, target_potential: float | None) -> None:
        """Switch between constant-charge and constant-potential mode.

        The (already normalized) mask is re-used, so this is cheap and does
        not touch the density or the wave functions.
        """
        mask_r = self.jellium.mask_r
        pw = self.jellium.pw
        if target_potential is None:
            self.jellium = JelliumExtension(
                mask_r, charge=self.charge, pw=pw)
        else:
            self.jellium = FixedPotentialJelliumExtension(
                mask_r,
                workfunction_target=target_potential,
                excess_electrons_guess=self.charge,
                tolerance=self.tol,
                pw=pw)

    def _slope(self) -> float | None:
        """Estimated d(potential)/d(charge) in V per electron."""
        history = getattr(self.jellium, 'history', [])
        if len(history) < 2:
            return None
        (c2, fl2), (c1, fl1) = history[-2:]
        if c1 == c2:
            return None
        return -(fl1 - fl2) * Ha / (c1 - c2)

    def _step_towards(self, target_potential: float) -> None:
        """Guess the charge needed for the new target potential."""
        slope = self._slope()
        if slope is None or np.isnan(self.fermi_level):
            return
        charge = self.charge + ((target_potential -
                                 self.get_electrode_potential()) / slope)
        self._set_charge(charge)
        self.log(f'Excess electrons changed to {self.charge:+.5f} based on a '
                 f'slope of {slope:.4f} V/electron.')

    def move_atoms(self, relpos_ac) -> None:
        # Each new geometry gets a fresh budget of equilibration attempts:
        self.niters = 0
        if not self.region.follows_atoms:
            return
        grid = self.jellium.mask_r.desc
        mask_r, z1, z2 = self.region.mask(grid, relpos_ac)
        self.jellium.set_mask(mask_r)
        self.log(f'Jellium region moved to {z1:.3f} to {z2:.3f}  # Å')

    def update1(self, nt_r):
        self.solvation.update1(nt_r)
        self.jellium.update1(nt_r)

    def update1pw(self, nt_g):
        grid = self.jellium.mask_r.desc
        nt_r = grid.empty()
        nt_r.scatter_from(nt_g.ifft(grid=grid.new(comm=None))
                          if nt_g is not None else None)
        self.solvation.update1(nt_r)
        self.jellium.update1pw(nt_g)

    def update2(self, nt_r, vHt_r, vt_sr) -> float:
        return self.solvation.update2(nt_r, vHt_r, vt_sr)


def _write_trace_in_z(a_r: UGArray, filename: str) -> None:
    """Write a quantity as a function of the z coordinate only.

    ``a_r`` must be a serial (gathered) array.
    """
    a_z = a_r.data.mean(axis=(0, 1))
    z_z = a_r.desc.xyz()[0, 0, :, 2] * Bohr
    with open(filename, 'w') as fd:
        for z, val in zip(z_z, a_z):
            fd.write(f'{z:f} {val:1.8f}\n')


class SJMPoissonSolver(PoissonSolverWrapper):
    def __init__(self, solver, dielectric):
        super().__init__(solver)

    def solve(self,
              vHt_r,
              rhot_r) -> float:
        self.solver.solve(vHt_r.data, rhot_r.data)
        eps_r = vHt_r.desc.from_data(self.solver.dielectric.eps_gradeps[0])
        eps0_r = eps_r.gather()
        vHt0_r = vHt_r.gather()
        if eps0_r is not None:
            saw_tooth_z = modified_saw_tooth(eps0_r)
            s1, s2 = saw_tooth_z[[2, 10]]
            v1, v2 = vHt0_r.data[:, :, [2, 10]].mean(axis=(0, 1))
            vHt0_r.data -= (v2 - v1) / (s2 - s1) * saw_tooth_z[np.newaxis,
                                                               np.newaxis]
            vHt0_r.data -= vHt0_r.data[:, :, -1].mean()
        vHt_r.scatter_from(vHt0_r)
        return np.nan


def modified_saw_tooth(eps_r: UGArray) -> np.ndarray:
    a_z = 1.0 / eps_r.data.mean(axis=(0, 1))
    saw_tooth_z = np.add.accumulate(a_z)
    saw_tooth_z -= 0.5 * a_z  # +0.5 from z=0.0 ???
    return saw_tooth_z


class SJMPWPoissonSolver(PWPoissonSolver):
    def __init__(self, pw, dielectric):
        super().__init__(pw)
        self.dielectric = dielectric
        self.saw_tooth_g = saw_tooth(pw, 0.25)

    def solve(self, vHt_g, rhot_g):
        energy = super().solve(vHt_g, rhot_g)
        dipole = rhot_g.moment()[2]
        slope = 4 * np.pi * dipole / rhot_g.desc.volume
        vHt_g.data += slope * self.saw_tooth_g.data
        # Shift potential so that it is zero above the slab:
        shift = 0.5 * slope * rhot_g.desc.cell_cv[2, 2]
        v0 = vHt_g.boundary_value(2)
        vHt_g.data[0] -= shift + v0
        return energy


def saw_tooth_sympy():
    """Fourier-transform."""
    from sympy import Symbol, integrate, sin, var
    z = var('z')
    G = Symbol('G', positive=True)
    b = Symbol('b', positive=True)
    m = integrate(sin(G * z) * z, (z, 0, b))
    print(m)  # -b*cos(G*b)/G + sin(G*b)/G**2


def saw_tooth(pw: PWDesc, width: float = 0.5) -> PWArray:
    """Saw-tooth in reciprocal space with a slope of 1."""
    assert np.allclose(pw.cell_cv[:2, 2], 0.0)
    assert np.allclose(pw.cell_cv[2, :2], 0.0)

    m0_g, m1_g = pw.indices_cG[:2, pw.ng1:pw.ng2] == 0
    mask_g = m0_g & m1_g
    Gz_i = pw.G_plus_k_Gv[mask_g, 2]
    if pw.comm.rank == 0.0:
        assert Gz_i[0] == 0.0
        Gz_i[0] = 1.0
    L = pw.cell_cv[2, 2]
    b = L / 2
    st_i = -(np.sin(b * Gz_i) / Gz_i -
             b * np.cos(b * Gz_i)) / Gz_i * (2j / L)
    if pw.comm.rank == 0.0:
        st_i[0] = 0.0

    # Make the saw-tooth more smooth (fold with Gaussian):
    alpha = width**-2
    st_i *= np.exp(-Gz_i**2 / (4 * alpha))

    # Shift by half the cell height:
    st_i *= np.exp(1j * Gz_i * b)

    st_g = pw.zeros()
    st_g.data[mask_g] = st_i
    return st_g
