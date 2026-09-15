from __future__ import annotations

import inspect
import os
import textwrap
import warnings

import numpy as np
from ase.units import Bohr, Ha, _e

from gpaw.core import PWDesc, UGArray
from gpaw.mpi import broadcast_float
from gpaw.extensions import Extension
from gpaw.new.builder import DFTComponentsBuilder
from gpaw.extensions.electrocatalysis.potentiostat import (
    DirectOptimizer, JelliumCharging, JelliumRegion, Potentiostat,
    WorkFunctionProbe, build_jellium_region)
# Raised by SJM calculations, so importable from here as well:
from gpaw.extensions.electrocatalysis.potentiostat import (  # noqa: F401
    PotentialConvergenceError)
from gpaw.new.poisson import PoissonSolverWrapper
from gpaw.new.solvation import Solvation, SolvationExtension

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


def _check_keywords(kwargs: dict) -> None:
    """Raise on anything that is not an SJM parameter."""
    unknown = set(kwargs) - set(_KEYWORDS)
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
    max_potential_stepsize:
        If a step moves the potential by more than this many volts *and*
        away from the target, it is rejected: the change in excess electrons
        is halved and the SCF re-run, without the rejected point counting
        against ``max_iters`` or feeding the slope regression.  This keeps a
        bad slope estimate from throwing the charge out of the linear
        region.  Set to ``None`` to turn the rejection/halving off
        entirely.  Default: 2 V.
    max_reruns:
        How many times in a row one step may be halved this way before a
        :class:`PotentialConvergenceError` is raised.  Default: 10.
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
                 max_potential_stepsize: int | None = 2,  # V
                 max_reruns: int = 10,
                 grand_output: bool = True,
                 always_adjust: bool = False,
                 slope: float | None = None,  # V/electron
                 mixer: float = 0.2,
                 slope_regression_depth: int = 4):
        # dipolelayer and zero_vacuum are chosen by build(), per mode:
        super().__init__(cavity, dielectric, interactions)
        self.jelliumregion = jelliumregion or {}
        self.target_potential = target_potential
        self.excess_electrons = excess_electrons
        self.tol = tol
        self.max_iters = max_iters
        self.max_potential_stepsize = max_potential_stepsize
        self.max_reruns = max_reruns
        self.grand_output = grand_output
        self.always_adjust = always_adjust
        self.slope = slope
        self.mixer = mixer
        self.slope_regression_depth = slope_regression_depth
        # Set by build(), so that change() can forward to the live extension:
        self._extension: SJMExtension | None = None

    def build(self, builder: DFTComponentsBuilder) -> SJMExtension:
        # Solvation only implements the dipole layer in plane-wave mode.  In
        # fd mode SJMPoissonSolver corrects the dipole itself, so it must not
        # be requested from Solvation, which would refuse it.
        self.dipolelayer = self.zero_vacuum = builder.mode == 'pw'
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
        # The optimizer is built in constant-charge mode too: it keeps the
        # slope settings, and the estimate, for a later switch of mode.
        optimizer = DirectOptimizer(
            slope=self.slope,
            mixer=self.mixer,
            slope_regression_depth=self.slope_regression_depth)
        # From here on the potentiostat holds the settings; SJMExtension
        # reads them back from it.
        potentiostat = Potentiostat(
            charging=charging,
            probe=WorkFunctionProbe(),
            target_potential=self.target_potential,
            optimizer=optimizer,
            tol=self.tol,
            max_potential_stepsize=self.max_potential_stepsize,
            max_iters=self.max_iters,
            max_reruns=self.max_reruns,
            always_adjust=self.always_adjust,
            grand_output=self.grand_output)

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
            if self.max_potential_stepsize is not None:
                log(f'  max potential stepsize: {self.max_potential_stepsize}'
                    f' V (halved up to {self.max_reruns} times if exceeded)')
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

        self._extension = SJMExtension(solvation, potentiostat, log=log)
        return self._extension

    def change(self, **kwargs) -> None:
        """Change SJM parameters, before or after the first calculation.

        Takes the same keywords as :meth:`SJMExtension.change`.  Before the
        calculation has been built there is nothing to keep alive, so the
        parameters are simply stored for :meth:`build` to pick up;
        afterwards the call is forwarded to the extension, which keeps
        the density and the wave functions.  Reach this as
        ``calc.sjm.change(...)``.

        Note that assigning to the extension instead --
        ``calc.sjm.charge = ...`` -- does *not* work: the charge has to reach
        the jellium as well, and only ``change()`` empties the results, which
        is what makes the next ``get_potential_energy()`` run the SCF loop
        again.
        """
        _check_keywords(kwargs)
        for key, value in kwargs.items():
            if key == 'jelliumregion':
                value = value or {}  # as in __init__
            setattr(self, key, value)
        if self._extension is not None:
            self._extension.change(**kwargs)

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
        dct.update({key: getattr(self, key) for key in _KEYWORDS})
        return dct


#: The SJM parameters: the keywords of SJM.__init__ other than those of the
#: solvation model.  This is the one place they are listed -- change(),
#: todict() and the keyword check all read them from here.
_KEYWORDS = tuple(name for name in inspect.signature(SJM.__init__).parameters
                  if name not in {'self', 'cavity', 'dielectric',
                                  'interactions'})


class SJMExtension(Extension):
    """SJM inside a running calculation.

    The settings of the potential control, and the state of the
    equilibration, are kept by :attr:`potentiostat`; this class forwards the
    GPAW hooks to it and to the solvation, and does the reporting.
    """

    name = 'sjm'

    def __init__(self,
                 solvation: SolvationExtension,
                 potentiostat: Potentiostat,
                 *,
                 log=None):
        self.solvation = solvation
        self.potentiostat = potentiostat
        self.log = log or (lambda *args, **kwargs: None)
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
    def target_potential(self) -> float | None:
        """Target electrode potential in V (kept by the potentiostat)."""
        return self.potentiostat.target_potential

    @property
    def tol(self) -> float:
        """Potential tolerance in V (kept by the potentiostat)."""
        return self.potentiostat.tol

    @property
    def _charging(self) -> JelliumCharging:
        """The charging scheme, narrowed: SJM always builds a jellium one."""
        charging = self.potentiostat.charging
        assert isinstance(charging, JelliumCharging)
        return charging

    @property
    def _optimizer(self) -> DirectOptimizer:
        """The optimizer, narrowed to the one concrete kind SJM builds."""
        optimizer = self.potentiostat.optimizer
        assert isinstance(optimizer, DirectOptimizer)
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
        ps = self.solvation.create_poisson_solver(
            grid, pw, charge=charge, xp=xp)
        if isinstance(pw, PWDesc):
            # SJM.build() has asked the solvation for the dipole layer and
            # the zeroed vacuum, so its plane-wave solver is already SJM's.
            return ps
        return SJMPoissonSolver(ps.solver)

    def post_scf_convergence(self,
                             ibzwfs,
                             nelectrons,
                             occ_calc,
                             mixer,
                             log) -> bool:
        # The attempts are only recorded there -- the equilibration is
        # reported in one block by post_converged(), so that it does not
        # interleave with the SCF table once per attempt.
        return self.potentiostat.post_scf_convergence(
            ibzwfs, nelectrons, occ_calc, mixer, log)

    def _log_equilibration(self, log) -> None:
        """Report the equilibration: where it ended up, and how it got there.

        Only the attempt that ended the equilibration is reported -- the
        intermediate ones are summarized by the regressed slope.
        """
        pot = self.potentiostat
        if not pot.constant_potential or not pot.attempts:
            return

        charge, potential = pot.attempts[-1]
        log(f'Potential found to be {potential:.5f} V '
            f'(with {charge:+.5f} excess electrons, '
            f'attempt {pot.niters:d}/{pot.max_iters:d}).')

        slope = pot.slope_estimate()
        if slope is not None:
            log(f'Slope estimated from the last '
                f'{len(self._optimizer.history):d} attempts is '
                f'{slope:.4f} V/electron,')
            # Å^2 -> cm^2 is 1e16 and F -> muF is 1e6:
            capacitance = -_e * 1e22 / (self._area() * slope)
            log(f'or apparent capacitance of {capacitance:.4f} muF/cm^2')

        log('Potential is within tolerance. Equilibrated.')
        if pot.pending_charge is not None:
            log('always_adjust: the next geometry starts from '
                f'{pot.pending_charge:+.5f} excess electrons.')
        log(flush=True)

    def _area(self) -> float:
        """Area of the unit cell in the xy plane, in Å^2."""
        return self.potentiostat.charging.region_area

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
        if self.potentiostat.grand_output:
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
        self.potentiostat.reset_attempts()

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
            except AttributeError as exc:
                raise ValueError('This cavity has no g_g attribute - '
                                 'cannot write a cavity trace.') from exc
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

    def change(self, **kwargs) -> None:
        """Change SJM parameters without throwing away the wave functions.

        Takes the parameters of :class:`SJM` (all but the solvation model),
        with the same meaning.  This is the replacement for
        the legacy ``calc.set(sj={...})``; reach it as ``calc.sjm.change()``.

        The density and the wave functions are kept, so the next
        ``get_potential_energy()`` continues the SCF loop from the current
        solution instead of starting over.  When the target potential is
        changed and the slope of potential vs. charge is known, a first step
        in the number of excess electrons is taken right away -- unless
        ``excess_electrons`` is given explicitly, which then wins.

        Assigning to the attributes instead -- ``sjm.charge = ...`` -- does
        *not* work: the charge has to reach the jellium as well, and only
        ``change()`` empties the results, which is what makes the next
        ``get_potential_energy()`` run the SCF loop again.
        """
        _check_keywords(kwargs)

        log = self.log
        pot = self.potentiostat
        opt = self._optimizer
        rerun = False  # does the SCF loop have to run again?

        if 'grand_output' in kwargs:
            pot.grand_output = kwargs['grand_output']
            if self._dft is not None and self._dft.results:
                # Cheap: just re-do the Legendre transform.
                self.post_converged(self._dft)
        for key in ['max_iters', 'max_reruns', 'max_potential_stepsize']:
            if key in kwargs:
                setattr(pot, key, kwargs[key])
        if 'tol' in kwargs:
            pot.tol = kwargs['tol']
            if (pot.target_potential is not None and
                    abs(self.get_electrode_potential() -
                        pot.target_potential) > pot.tol):
                rerun = True

        if 'always_adjust' in kwargs:
            pot.always_adjust = kwargs['always_adjust']
            if not pot.always_adjust:
                pot.pending_charge = None
            elif kwargs.get('target_potential',
                            pot.target_potential) is None:
                warnings.warn('always_adjust has no effect in '
                              'constant-charge mode.')
        if 'mixer' in kwargs:
            opt.mixer = kwargs['mixer']
        if 'slope_regression_depth' in kwargs:
            opt.slope_regression_depth = kwargs['slope_regression_depth']
        if 'slope' in kwargs:
            # Whatever the user says replaces the estimate, so the attempts
            # behind the old one go too:
            opt.reset()
            opt.slope = kwargs['slope']
            if opt.slope is None:
                log('Slope estimate reset; it will be guessed again.')
            else:
                log(f'Slope set to {opt.slope:.4f} V/electron.')

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
            pot.set_target_potential(target)
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
            pot.set_charge(kwargs['excess_electrons'])
            log(f'Excess electrons changed to {self.charge:+.5f}.')
            rerun = True

        if rerun and self._dft is not None:
            # Empty results make ASECalculator.iconverge() run the SCF loop
            # again; the density and the wave functions are kept.
            self._dft.results.clear()
            pot.reset_attempts()

    def _relpos_ac(self):
        if self._dft is None:
            raise RuntimeError('No results yet - run a calculation before '
                               'changing the jellium region.')
        return self._dft.relpos_ac

    def _step_towards(self, target_potential: float) -> None:
        """Guess the charge needed for the new target potential."""
        slope = self.potentiostat.slope_estimate()
        if slope is None or np.isnan(self.fermi_level):
            return
        charge = self.charge + ((target_potential -
                                 self.get_electrode_potential()) / slope)
        self.potentiostat.set_charge(charge)
        self.log(f'Excess electrons changed to {self.charge:+.5f} based on a '
                 f'slope of {slope:.4f} V/electron.')

    def move_atoms(self, relpos_ac) -> None:
        # The wrapped solvation extension is not itself in the extension
        # list, so its cavity only follows the atoms if we forward the move:
        self.solvation.move_atoms(relpos_ac)
        # A fresh budget of equilibration attempts for the new geometry,
        # starting from the charge staged by always_adjust, if any:
        if self.potentiostat.new_geometry() is not None:
            self.log(f'Excess electrons adjusted to {self.charge:+.5f} for '
                     'the new geometry (always_adjust).')
        grid = self._charging.mask_r.desc
        zs = self._charging.move(relpos_ac, grid)
        if zs is not None:
            self.log(f'Jellium region moved to {zs[0]:.3f} to '
                     f'{zs[1]:.3f}  # Å')

    def update1(self, nt_r):
        self.solvation.update1(nt_r)
        self.potentiostat.add_charge(nt_r)

    def update1pw(self, nt_g):
        self.solvation.update1pw(nt_g)
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
    with open(filename, 'w', encoding='utf-8') as out:
        for z, val in zip(z_z, a_z):
            out.write(f'{z:f} {val:1.8f}\n')


class SJMPoissonSolver(PoissonSolverWrapper):
    """Finite-difference Poisson solver with a fitted dipole correction.

    Unlike the plane-wave solver -- ``ConjugateGradientPoissonSolver``,
    built by :meth:`SJMExtension.create_poisson_solver` -- this one does
    not drive the boundary slope to zero.  It reaches about 1e-6 a.u., and
    refining the grid does not improve that, because the residual is set by
    a fit rather than by the grid.

    A zero boundary condition in z pins both vacua to ``v == 0``, so a slab
    with a dipole is forced to ramp: the ramp is the exact solution of the
    wrong boundary-value problem, not a numerical artifact.  :meth:`solve`
    removes it afterwards by fitting the slope over :func:`vacuum_window` and
    subtracting a scaled :func:`modified_saw_tooth` -- a one-dimensional shape
    built from the xy-averaged ``1 / eps``, not a solution of the actual
    operator.  Plane waves have no such ramp to remove, being fully periodic,
    which is why ``_correct_dipole`` there can cancel analytically instead.

    Three ways of closing the gap have been measured and none worked.  An
    analytic coefficient from the density fails because the 19-point stencil
    does not obey the continuum identity it relies on (5.8e-5 against the
    fit's 5e-8).  Subtracting a solved reference layer
    flattens to 1e-7 but injects a real field where the layer meets the
    density -- the Cu(100) force gap went 0.071 to 6.678 eV/Ang, invisible to
    the vacuum-flatness tests -- and moving the layer to the cell edge leaves
    no vacuum to fit in.  Cancelling the dipole at the source, as
    :meth:`gpaw.dipole_correction.DipoleCorrection.fdsolve` does, is the
    structurally right answer, but its analytic potential holds only where
    ``eps == 1`` and one of its lobes lands in the solvent.
    """

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
        warnings.warn(
            'Not enough vacuum below the slab to fit the dipole correction '
            f'clear of the first {SKIP_NEAR_BOUNDARY} Ang of the cell, so '
            'it is being fitted next to the zero boundary instead.  The '
            'fit is noisy there and the SCF may fail to converge.  Add '
            'vacuum below the slab.')
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
