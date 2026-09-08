from __future__ import annotations

import os
import textwrap
import warnings
from typing import Any

import numpy as np
from ase.units import Bohr, Ha, _e

from gpaw import ConvergenceError
from gpaw.core import PWArray, PWDesc, UGArray, pw_to_grid
from gpaw.mpi import broadcast_float
from gpaw.extensions import Extension
from gpaw.new.builder import DFTComponentsBuilder
from gpaw.extensions.electrocatalysis.potentiostat import (
    DirectOptimizer, JelliumCharging, JelliumRegion, Potentiostat,
    WorkFunctionProbe, build_jellium_region, regress_slope)
from gpaw.new.poisson import PoissonSolverWrapper
from gpaw.new.pw.poisson import PWPoissonSolver
from gpaw.new.solvation import Solvation, SolvationExtension

_KEYWORDS = frozenset({'target_potential', 'excess_electrons', 'tol',
                       'max_iters', 'grand_output', 'jelliumregion',
                       'always_adjust', 'slope', 'mixer',
                       'slope_regression_depth'})

#: Grid points closer than this to the lower cell boundary are left out of
#: the slope-fitting window: the finite-difference solution is least accurate
#: there because of the zero boundary condition.
SKIP_NEAR_BOUNDARY = 0.5  # Å

#: Smallest usable slope-fitting window.
MIN_WINDOW_POINTS = 4

#: A grid point counts as solvated once the dielectric exceeds this.
EPS_VACUUM_TOL = 1e-3

#: A grid point counts as occupied once ``|rho|`` exceeds this fraction of its
#: largest xy-averaged value.  It has to be a small fraction: the fit must sit
#: in genuine vacuum, where the field does not depend on the density.  Let it
#: reach into the tail of the slab and the measured slope starts to track the
#: density, which feeds back into the next iteration and stalls the SCF.
RHO_VACUUM_TOL = 1e-5


class PotentialConvergenceError(ConvergenceError):
    """Raised if the electrode potential did not equilibrate."""


def _check_keywords(kwargs: dict) -> None:
    """Raise on anything that is not an SJM parameter."""
    unknown = set(kwargs) - _KEYWORDS
    if unknown:
        raise ValueError(
            f'Unexpected SJM keyword(s): {", ".join(sorted(unknown))}.  '
            f'Allowed are: {", ".join(sorted(_KEYWORDS))}.')


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
        Shape of the counter-charge region, as a dict.  The default is a
        slab (:class:`SlabRegion`), taking the keys ``'top'``, ``'bottom'``,
        ``'thickness'`` and ``'fix_bottom'``.  A ``'shape'`` key selects a
        different kind of region; see :func:`build_jellium_region`.
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
    always_adjust:
        Take one more step in the number of excess electrons even when the
        potential is already within ``tol``.  The step is not re-converged --
        it is applied to the *next* geometry -- so it costs no extra SCF
        cycles.  Together with a loose ``tol`` this lets a relaxation optimize
        the structure and the potential simultaneously.  Default: False.
    slope:
        Initial guess of the slope of the potential with respect to the number
        of excess electrons, in V/electron.  It is only used until there have
        been enough attempts to regress a slope from; ``None`` guesses one
        from the apparent capacitance of a plate capacitor of the cell's
        area.
    mixer:
        Damping of the slope estimate, because regressing a slope can be
        noisy: ``new = mixer * old + (1 - mixer) * regressed``.  Set to 0 for
        no damping.  It also sets how fast the step in excess electrons may
        grow: a step may exceed the previous one by the fraction
        ``1 - mixer`` of the excess the slope asks for.  Default: 0.2.
    slope_regression_depth:
        Number of previous attempts that the slope is regressed over.
        Default: 4.
    """

    name = 'sjm'
    properties = ['excess_electrons', 'electrode_potential']

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
                 grand_output: bool = True,
                 always_adjust: bool = False,
                 slope: float | None = None,  # V/electron
                 mixer: float = 0.2,
                 slope_regression_depth: int = 4):
        super().__init__(cavity, dielectric, interactions,
                         dipolelayer=True, zero_vacuum=True)
        self.jelliumregion = jelliumregion or {}
        self.target_potential = target_potential
        self.excess_electrons = excess_electrons
        self.tol = tol
        self.max_iters = max_iters
        self.grand_output = grand_output
        self.always_adjust = always_adjust
        self.slope = slope
        self.mixer = mixer
        self.slope_regression_depth = slope_regression_depth
        # Set by build(), so that set() can forward to the live extension:
        self._extension: SJMExtension | None = None

    def build(self, builder: DFTComponentsBuilder) -> SJMExtension:
        solvation = super().build(builder)
        log = builder.log
        self._check_convergence_criteria(builder.params.convergence, log)
        self._check_always_adjust(log)

        region = build_jellium_region(self.jelliumregion,
                                      cavity=solvation.cavity)
        mask_r, z1, z2 = region.mask(builder.fine_grid, builder.relpos_ac)

        charging = JelliumCharging(mask_r,
                                   pw=builder.interpolation_desc,
                                   charge=self.excess_electrons,
                                   region=region)
        if self.target_potential is None:
            optimizer = None
        else:
            optimizer = DirectOptimizer(
                slope=self.slope,
                mixer=self.mixer,
                slope_regression_depth=self.slope_regression_depth)
        potentiostat = Potentiostat(charging=charging,
                                    probe=WorkFunctionProbe(),
                                    target_potential=self.target_potential,
                                    optimizer=optimizer,
                                    tol=self.tol)

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
            if self.slope is not None:
                log(f'  initial guess of slope: {self.slope:.4f}  '
                    '# V/electron')
            log(f'  slope damping (mixer): {self.mixer:.2f}')
            log('  slope regressed over the last '
                f'{self.slope_regression_depth:d} attempts')
            if self.always_adjust:
                log('  always adjust: the charge is stepped on to the next '
                    'geometry')

        self._extension = SJMExtension(solvation, potentiostat,
                                       target_potential=self.target_potential,
                                       tol=self.tol,
                                       max_iters=self.max_iters,
                                       grand_output=self.grand_output,
                                       always_adjust=self.always_adjust,
                                       slope=self.slope,
                                       mixer=self.mixer,
                                       slope_regression_depth=(
                                           self.slope_regression_depth),
                                       log=log)
        return self._extension

    def set(self, **kwargs) -> None:
        """Change SJM parameters, before or after the first calculation.

        Takes the same keywords as :meth:`SJMExtension.set`.  Before the
        calculation has been built there is nothing to keep alive, so the
        parameters are simply stored for :meth:`build` to pick up;
        afterwards the call is forwarded to the extension, which keeps the
        density and the wave functions.  Reach this as ``calc.sjm.set(...)``.

        Note that assigning to the extension instead --
        ``calc.sjm.charge = ...`` -- does *not* work: the charge has to reach
        the jellium as well, and only ``set()`` empties the results, which is
        what makes the next ``get_potential_energy()`` run the SCF loop again.
        """
        _check_keywords(kwargs)
        for key, value in kwargs.items():
            if key == 'jelliumregion':
                value = value or {}  # as in __init__
            setattr(self, key, value)
        if self._extension is not None:
            self._extension.set(**kwargs)

    def _check_always_adjust(self, log) -> None:
        if self.always_adjust and self.target_potential is None:
            msg = ('always_adjust has no effect in constant-charge mode: '
                   'there is no target potential to step the number of '
                   'excess electrons towards.')
            warnings.warn(msg)
            log(textwrap.fill('Warning: ' + msg) + '\n')

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
            grand_output=self.grand_output,
            always_adjust=self.always_adjust,
            slope=self.slope,
            mixer=self.mixer,
            slope_regression_depth=self.slope_regression_depth)
        return dct


class SJMExtension(Extension):
    name = 'sjm'

    def __init__(self,
                 solvation: SolvationExtension,
                 potentiostat: Potentiostat,
                 *,
                 target_potential: float | None = None,  # eV
                 tol: float = 0.001,  # eV
                 max_iters: int = 10,
                 grand_output: bool = True,
                 always_adjust: bool = False,
                 slope: float | None = None,  # V/electron
                 # Neither of these is used here -- they are only handed on
                 # to the optimizer, so ``None`` means "not specified" and
                 # lets DirectOptimizer supply its own default rather than
                 # repeating it:
                 mixer: float | None = None,
                 slope_regression_depth: int | None = None,
                 log=None):
        self.solvation = solvation
        self.potentiostat = potentiostat
        self.dielectric = solvation.dielectric
        self.target_potential = target_potential
        self.tol = tol
        self.max_iters = max_iters
        self.grand_output = grand_output
        self.always_adjust = always_adjust
        # Only used while there is no optimizer (constant-charge mode) --
        # otherwise the optimizer keeps the estimate, see _slope():
        self.slope = slope
        self.mixer = mixer
        self.slope_regression_depth = slope_regression_depth
        self.log = log or (lambda *args, **kwargs: None)

        self.niters = 0
        # (Charge, Fermi-level) of every attempt of the current
        # equilibration, reported in one block by post_converged():
        self.attempts: list[tuple[float, float]] = []
        # Charge staged by always_adjust for the next geometry:
        self._pending_charge: float | None = None
        # Set by post_converged():
        self.fermi_level = np.nan  # Ha
        self.omega_free = np.nan  # Ha
        self.omega_extrapolated = np.nan  # Ha
        self._dft = None

    @property
    def charge(self) -> float:
        """Number of excess electrons (kept by the potentiostat)."""
        return self.potentiostat.charge

    @charge.setter
    def charge(self, q: float) -> None:
        self.potentiostat.charge = q

    @property
    def _charging(self) -> JelliumCharging:
        """The charging scheme, narrowed: SJM always builds a jellium one."""
        charging = self.potentiostat.charging
        assert isinstance(charging, JelliumCharging)
        return charging

    @property
    def _optimizer(self) -> DirectOptimizer | None:
        """The optimizer, narrowed to the one concrete kind SJM builds."""
        optimizer = self.potentiostat.optimizer
        assert optimizer is None or isinstance(optimizer, DirectOptimizer)
        return optimizer

    @property
    def region(self) -> JelliumRegion:
        """Shape of the jellium counter-charge region."""
        return self._charging.region

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
        if not self.potentiostat.constant_potential:
            return True

        # Grab the values belonging to this attempt before the potentiostat
        # steps the charge on to the next one:
        charge = self.charge
        fermi_level = ibzwfs.fermi_level

        converged = self.potentiostat.equilibrate(
            ibzwfs, nelectrons, occ_calc, mixer, log)

        # Only record here -- the equilibration is reported in one block by
        # post_converged(), so that it does not interleave with the SCF
        # table once per attempt.
        self.niters += 1
        self.attempts.append((charge, fermi_level))

        if converged:
            if self.always_adjust:
                # Step the charge on anyway, but do not re-converge: the new
                # value is picked up by the next geometry (move_atoms()), so
                # that the potential can equilibrate alongside the structure.
                # Reported by _log_equilibration(), not here, to keep it out
                # of the SCF table.
                self._pending_charge = self.potentiostat.propose_next(ibzwfs)
            return True

        if self.niters >= self.max_iters:
            raise PotentialConvergenceError(
                self._convergence_error_message(log))
        return False

    def _log_equilibration(self, log) -> None:
        """Report the equilibration: where it ended up, and how it got there.

        Only the attempt that ended the equilibration is reported -- the
        intermediate ones are summarized by the regressed slope.
        """
        if not self.potentiostat.constant_potential or not self.attempts:
            return

        charge, fermi_level = self.attempts[-1]
        log(f'Potential found to be {-fermi_level * Ha:.5f} V '
            f'(with {charge:+.5f} excess electrons, '
            f'attempt {len(self.attempts):d}/{self.max_iters:d}).')

        slope = self._slope()
        if slope is not None and self._optimizer is not None:
            log(f'Slope estimated from the last '
                f'{len(self._optimizer.history):d} attempts is '
                f'{slope:.4f} V/electron,')
            # Å^2 -> cm^2 is 1e16 and F -> muF is 1e6:
            capacitance = -_e * 1e22 / (self._area() * slope)
            log(f'or apparent capacitance of {capacitance:.4f} muF/cm^2')

        log('Potential is within tolerance. Equilibrated.')
        if self._pending_charge is not None:
            log('always_adjust: the next geometry starts from '
                f'{self._pending_charge:+.5f} excess electrons.')
        log(flush=True)

    def _area(self) -> float:
        """Area of the unit cell in the xy plane, in Å^2."""
        return self.potentiostat.charging.region_area

    def _convergence_error_message(self, log) -> str:
        msg = (f'The electrode potential could not be equilibrated to '
               f'{self.target_potential:.3f} +/- {self.tol:.3f} V in '
               f'{self.max_iters:d} attempts.  This may indicate that your '
               'work function is noisier than your potential tolerance; you '
               'may try tightening the SCF convergence criteria.  The excess '
               'electrons and the resulting potentials are listed below; '
               'plotting them could give you insight into the problem.')
        msg = textwrap.fill(msg) + '\n'
        for charge, fermi_level in self.attempts:
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

        self._log_equilibration(log)

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

        # The next calculation (the next step of a relaxation, say) gets a
        # fresh equilibration budget and a fresh record to report:
        self.attempts = []
        self.niters = 0

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

        A ``'z'`` trace has one row per grid point of the full mesh, so traces
        from an fd and a plane-wave run of the same cell line up row by row.
        A real-space grid with a zero boundary in z does not store the z = 0
        point; the row written for it repeats the first stored value and is
        padding rather than a computed value.
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
            mask_r = self._charging.mask_r
            charge_r = mask_r.new()
            charge_r.data[:] = -self.charge * mask_r.data
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
        ``tol``, ``max_iters``, ``grand_output``, ``jelliumregion``,
        ``always_adjust``, ``slope``, ``mixer`` and
        ``slope_regression_depth``, with the same meaning as for
        :class:`SJM`.  This is the replacement for
        the legacy ``calc.set(sj={...})``; reach it as ``calc.sjm.set()``.

        The density and the wave functions are kept, so the next
        ``get_potential_energy()`` continues the SCF loop from the current
        solution instead of starting over.  When the target potential is
        changed and the slope of potential vs. charge is known, a first step
        in the number of excess electrons is taken right away -- unless
        ``excess_electrons`` is given explicitly, which then wins.

        Assigning to the attributes instead -- ``sjm.charge = ...`` -- does
        *not* work: the charge has to reach the jellium as well, and only
        ``set()`` empties the results, which is what makes the next
        ``get_potential_energy()`` run the SCF loop again.
        """
        _check_keywords(kwargs)

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
            if self.target_potential is not None:
                self.potentiostat.tol = self.tol
                if abs(self.get_electrode_potential() -
                       self.target_potential) > self.tol:
                    rerun = True

        if 'always_adjust' in kwargs:
            self.always_adjust = kwargs['always_adjust']
            if not self.always_adjust:
                self._pending_charge = None
            elif kwargs.get('target_potential',
                            self.target_potential) is None:
                warnings.warn('always_adjust has no effect in '
                              'constant-charge mode.')
        if 'mixer' in kwargs:
            self.mixer = kwargs['mixer']
            opt = self._optimizer
            if opt is not None:
                opt.mixer = self.mixer
        if 'slope_regression_depth' in kwargs:
            self.slope_regression_depth = kwargs['slope_regression_depth']
            opt = self._optimizer
            if opt is not None:
                opt.slope_regression_depth = self.slope_regression_depth
        if 'slope' in kwargs:
            # Whatever the user says replaces the estimate, so the attempts
            # behind the old one go too:
            self.slope = kwargs['slope']
            opt = self._optimizer
            if opt is not None:
                opt.reset()
                opt.slope = self.slope
            if self.slope is None:
                log('Slope estimate reset; it will be guessed again.')
            else:
                log(f'Slope set to {self.slope:.4f} V/electron.')

        if 'jelliumregion' in kwargs:
            charging = self._charging
            charging.region = build_jellium_region(
                kwargs['jelliumregion'], cavity=self.solvation.cavity)
            mask_r, z1, z2 = charging.region.mask(charging.mask_r.desc,
                                                  self._relpos_ac())
            charging.set_region(mask_r)
            log(f'Jellium region changed to {z1:.3f} to {z2:.3f}  # Å')
            rerun = True

        if 'target_potential' in kwargs:
            target = kwargs['target_potential']
            if (target is None) != (self.target_potential is None):
                self._switch_mode(target)
            elif target is not None:
                # Same mode as before, so the optimizer already exists:
                self.potentiostat.target_potential = target
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
            self.attempts = []

    def _relpos_ac(self):
        if self._dft is None:
            raise RuntimeError('No results yet - run a calculation before '
                               'changing the jellium region.')
        return self._dft.relpos_ac

    def _set_charge(self, charge: float) -> None:
        """Set the number of excess electrons."""
        self.charge = charge
        # An explicit charge wins over one staged by always_adjust:
        self._pending_charge = None

    def _switch_mode(self, target_potential: float | None) -> None:
        """Switch between constant-charge and constant-potential mode.

        The charging scheme (and its normalized mask) is left in place, so
        this is cheap and does not touch the density or the wave functions;
        only the optimizer and the target are swapped.
        """
        # Carry the slope estimate across the switch:
        self.slope = self._slope()
        self.potentiostat.target_potential = target_potential
        # A tol set while in constant-charge mode was kept on this class
        # only (the 'tol' handler skips the potentiostat then); adopt it now.
        self.potentiostat.tol = self.tol
        if target_potential is None:
            self.potentiostat.optimizer = None
        else:
            # Leave out what was never specified, so that DirectOptimizer's
            # own defaults apply instead of being shadowed by a repeat:
            self.potentiostat.optimizer = DirectOptimizer(
                slope=self.slope,
                **self._optimizer_defaults())

    def _optimizer_defaults(self) -> dict[str, Any]:
        """The optimizer keywords that were actually specified.

        Neither of these is used by this class -- it only hands them on.
        Leaving out the ones that were never specified is what keeps their
        defaults in DirectOptimizer alone, instead of repeating them here
        where they would silently drift apart.
        """
        specified: dict[str, Any] = {}
        if self.mixer is not None:
            specified['mixer'] = self.mixer
        if self.slope_regression_depth is not None:
            specified['slope_regression_depth'] = self.slope_regression_depth
        return specified

    def _slope(self) -> float | None:
        """Estimated d(potential)/d(charge) in V per electron.

        This is the damped estimate that the optimizer steps along, so that
        set() and the log report what is actually used.
        """
        opt = self._optimizer
        if opt is not None:
            if opt.slope is not None:
                return opt.slope
            return regress_slope(opt.history)
        return self.slope

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
        # The wrapped solvation extension is not itself in the extension
        # list, so its cavity only follows the atoms if we forward the move:
        self.solvation.move_atoms(relpos_ac)
        charge = self._pending_charge
        if charge is not None:
            # always_adjust: the step taken after the last geometry had
            # already equilibrated (see post_scf_convergence).
            self._set_charge(charge)
            self.log(f'Excess electrons adjusted to {self.charge:+.5f} for '
                     'the new geometry (always_adjust).')
        # Each new geometry gets a fresh budget of equilibration attempts,
        # and a fresh record to regress the slope from -- the slope estimate
        # itself is carried over:
        self.niters = 0
        self.attempts = []
        opt = self._optimizer
        if opt is not None:
            opt.forget_attempts()
        grid = self._charging.mask_r.desc
        zs = self._charging.move(relpos_ac, grid)
        if zs is not None:
            self.log(f'Jellium region moved to {zs[0]:.3f} to '
                     f'{zs[1]:.3f}  # Å')

    def update1(self, nt_r):
        self.solvation.update1(nt_r)
        self.potentiostat.add_charge(nt_r)

    def update1pw(self, nt_g):
        nt_r = pw_to_grid(nt_g, self._charging.mask_r.desc)
        self.solvation.update1(nt_r)
        self.potentiostat.add_charge_pw(nt_g)

    def update2(self, nt_r, vHt_r, vt_sr) -> None:
        self.solvation.update2(nt_r, vHt_r, vt_sr)

    def update2pw(self, nt_g, vHt_h, vt_sr) -> None:
        self.solvation.update2pw(nt_g, vHt_h, vt_sr)

    def get_energy_contributions(self) -> dict[str, float]:
        return self.solvation.get_energy_contributions()

    def force_contribution(self, nt_r, vHt_r):
        # The wrapped solvation extension is not in the extension list, so
        # its force correction only reaches the total if we forward it.
        # Without this the forces are inconsistent with the energy and a
        # relaxation walks uphill.  The jellium itself is a fixed external
        # background and exerts no force on the atoms.
        return self.solvation.force_contribution(nt_r, vHt_r)


def _write_trace_in_z(a_r: UGArray, filename: str) -> None:
    """Write a quantity as a function of the z coordinate only.

    ``a_r`` must be a serial (gathered) array.

    A real-space grid with a zero boundary in z does not store the z = 0
    point, so its trace would be one row shorter than the same trace from a
    plane-wave run, where the fine grid is an FFT grid and always carries the
    full mesh.  Pad the missing point with a copy of the first stored value,
    so that traces of the same cell have the same number of rows and the same
    z column in every mode.
    """
    a_z = a_r.data.mean(axis=(0, 1))
    z_z = a_r.desc.xyz()[0, 0, :, 2] * Bohr
    if a_r.desc.zerobc_c[2]:
        a_z = np.concatenate([a_z[:1], a_z])
        z_z = np.concatenate([[0.0], z_z])
    with open(filename, 'w') as out:
        for z, val in zip(z_z, a_z):
            out.write(f'{z:f} {val:1.8f}\n')


class SJMPoissonSolver(PoissonSolverWrapper):
    def __init__(self, solver, dielectric):
        super().__init__(solver)

    def solve(self,
              vHt_r,
              rhot_r) -> float:
        self.solver.solve(vHt_r.data, rhot_r.data)
        eps_r = vHt_r.desc.from_data(self.solver.dielectric.eps_gradeps[0])
        eps0_r = eps_r.gather()
        rhot0_r = rhot_r.gather()
        vHt0_r = vHt_r.gather()
        if eps0_r is not None:
            saw_tooth_z = modified_saw_tooth(eps0_r)
            v_z = vHt0_r.data.mean(axis=(0, 1))
            window = vacuum_window(eps0_r, rhot0_r)
            # Cancel the slope of the potential at the lower boundary by
            # subtracting a scaled saw-tooth.  Both slopes are measured over
            # the same window, so any constant offset in the saw-tooth (see
            # modified_saw_tooth) drops out of the ratio.
            slope_v = np.polyfit(window, v_z[window], 1)[0]
            slope_s = np.polyfit(window, saw_tooth_z[window], 1)[0]
            vHt0_r.data -= (slope_v / slope_s) * saw_tooth_z[np.newaxis,
                                                             np.newaxis]
            vHt0_r.data -= vHt0_r.data[:, :, -1].mean()
        vHt_r.scatter_from(vHt0_r)
        return np.nan


def vacuum_window(eps_r: UGArray, rhot_r: UGArray) -> np.ndarray:
    """Grid indices to measure the boundary slope over.

    The window is the run of points above the lower cell boundary that is
    genuinely empty: the dielectric is still that of vacuum, so the saw-tooth
    of :func:`modified_saw_tooth` is linear there, *and* the charge density
    has not yet picked up, so the potential is too.  The density is what ends
    the window in practice -- the slab sits below the solvent and has
    ``eps == 1`` itself, so the dielectric alone would let the window run
    right through it.

    Defining the window physically rather than through fixed indices keeps
    the fit over the same distance whatever the grid spacing is.

    ``eps_r`` and ``rhot_r`` must be serial (gathered) arrays.
    """
    eps_z = eps_r.data.mean(axis=(0, 1))
    rho_z = abs(rhot_r.data.mean(axis=(0, 1)))
    # Take the heights from the descriptor rather than from an index times a
    # spacing: with a zero boundary condition in z -- which is what FD mode
    # uses for a slab -- the first row of the array sits at z = h, not z = 0.
    z_z = eps_r.desc.xyz()[0, 0, :, 2] * Bohr
    start = int(np.searchsorted(z_z, SKIP_NEAR_BOUNDARY))
    # Stop below the first point that feels either the solvent or the slab:
    rho_max = rho_z.max()
    occupied_z = eps_z > 1.0 + EPS_VACUUM_TOL
    if rho_max > 0.0:
        occupied_z |= rho_z > RHO_VACUUM_TOL * rho_max
    above_z = np.nonzero(occupied_z[1:])[0]
    stop = 1 + above_z[0] if len(above_z) else len(eps_z)

    if stop - start < MIN_WINDOW_POINTS:
        # Too little vacuum to also skip the boundary layer.  Move the window
        # down towards the boundary rather than up into the slab: a fit close
        # to the boundary is only less accurate, whereas a fit that reaches
        # into the density couples the correction to it (see RHO_VACUUM_TOL).
        start = max(1, stop - MIN_WINDOW_POINTS)
    if stop - start < 2:
        # Degenerate cell with no vacuum at all -- np.polyfit needs two
        # points, and there is nothing better to offer it.
        stop = min(len(eps_z), start + 2)
    return np.arange(start, stop)


def modified_saw_tooth(eps_r: UGArray) -> np.ndarray:
    a_z = 1.0 / eps_r.data.mean(axis=(0, 1))
    saw_tooth_z = np.add.accumulate(a_z)
    # The half-step below and the missing factor of h are both constant
    # offsets/scalings.  They cancel in the slope ratio taken by
    # SJMPoissonSolver.solve(), and the constant they leave in the potential
    # is removed when it is re-zeroed at the top of the cell, so neither
    # affects the result.
    saw_tooth_z -= 0.5 * a_z
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
