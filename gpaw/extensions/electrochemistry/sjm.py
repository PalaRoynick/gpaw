from __future__ import annotations

import inspect
import os
import textwrap
import warnings
from types import SimpleNamespace
from typing import Any

import numpy as np
from ase.units import Bohr, Ha

from gpaw.core import UGArray
from gpaw.dft import ExtensionInput
from gpaw.extensions import Extension
from gpaw.new.builder import DFTComponentsBuilder
from gpaw.extensions.electrochemistry.potentiostat import (
    DirectOptimizer, JelliumCharging, Potentiostat, build_jellium_region)
from gpaw.solvation.extension import Solvation


always_adjust_warning = 'always_adjust has no effect in constant-charge mode'


def _check_keywords(kwargs: dict) -> None:
    """Raise on anything that is not an SJM parameter."""
    unknown = set(kwargs) - set(_KEYWORDS)
    if unknown:
        raise ValueError(
            f'Unexpected SJM keyword(s): {", ".join(sorted(unknown))}.  '
            f'Allowed are: {", ".join(sorted(_KEYWORDS))}.')


class SJM(ExtensionInput):
    """Solvated jellium method.

    Electronically grand-canonical calculations, typically for simulating
    electrochemical interfaces.  A jellium counter charge is placed in the
    implicit solvent above the slab; the number of excess electrons can
    either be fixed (constant-charge mode) or adjusted until the electrode
    potential matches ``target_potential`` (constant-potential mode).

    An SJM object, and the solvent objects given to it, belong to one
    calculator: each calculation keeps its state in them.  So give every
    calculator its own -- every image of a NEB, for instance.

    Parameters
    ----------
    cavity, dielectric, interactions:
        Solvation model, held as ``sjm.solvation``; see
        :class:`gpaw.solvation.Solvation`.
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
        Default: 0.005 V.
    max_iters:
        Maximum number of attempts to equilibrate the potential, in
        constant-potential mode.  A :class:`PotentialConvergenceError` is
        raised if the potential is not reached.  Default: 10.
    max_potential_stepsize:
        If a step moves the potential by more than this many volts *and*
        away from the target, it is rejected: the change in excess electrons
        is halved and the SCF re-run, without the rejected point counting
        against ``max_iters`` or feeding the slope regression.  This keeps a
        bad slope estimate from throwing the charge out of the linear
        region.  Set to ``None`` to turn the rejection/halving off
        entirely.  Default: 2 V.
    max_reruns:
        How many times in a row one step may be halved because
        ``max_potential_stepsize`` was exceeded before a
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

    def __init__(self,
                 *,
                 cavity,
                 dielectric,
                 interactions,
                 jelliumregion: dict | None = None,
                 target_potential: float | None = None,  # V
                 excess_electrons: float = 0.0,
                 tol: float = 0.005,  # V
                 max_iters: int = 10,
                 max_potential_stepsize: int | None = 2,  # V
                 max_reruns: int = 10,
                 grand_output: bool = True,
                 always_adjust: bool = False,
                 slope: float | None = None,  # V/electron
                 mixer: float = 0.2,
                 slope_regression_depth: int = 4):
        # The solvation model, with the electrode potential referenced to
        # the vacuum at the top of the cell:
        self.solvation = Solvation(cavity, dielectric, interactions,
                                   dipolelayer={'zero_vacuum': True})
        # The parameters as given (and changed to).  They are what build()
        # starts a calculation from and what todict() records; reading them
        # as attributes gives the live values instead, see __getattr__.
        self._params: dict[str, Any] = dict(
            jelliumregion=jelliumregion or {},
            target_potential=target_potential,
            excess_electrons=excess_electrons,
            tol=tol,
            max_iters=max_iters,
            max_potential_stepsize=max_potential_stepsize,
            max_reruns=max_reruns,
            grand_output=grand_output,
            always_adjust=always_adjust,
            slope=slope,
            mixer=mixer,
            slope_regression_depth=slope_regression_depth)
        # Set by build(): the running potential control.
        self._potentiostat: Potentiostat | None = None

    # Once the calculation has been built, the parameters are read off it,
    # so that, e.g., excess_electrons follows the equilibration; before
    # that, they are the ones given.  __getattr__ only runs when normal
    # lookup fails, and the parameters are never set as attributes.
    def __getattr__(self, name):
        params = self.__dict__.get('_params')  # no recursion when copying
        if params is not None and name in _KEYWORDS:
            pot = self.__dict__.get('_potentiostat')
            if pot is None:
                return params[name]
            if name == 'jelliumregion':
                return pot.charging.region
            for owner in (pot, pot.optimizer):
                if hasattr(owner, name):
                    return getattr(owner, name)
        raise AttributeError(
            f'{type(self).__name__!r} object has no attribute {name!r}')

    def __setattr__(self, name, value):
        """Refuse to set a parameter: change() is the way to set one."""
        if name in _KEYWORDS:
            raise AttributeError(
                f'{name!r} cannot be assigned: use change({name}=...) '
                'instead, which reaches the calculation as well.')
        super().__setattr__(name, value)

    def __dir__(self):
        """List the parameters too, for help() and completion."""
        return [*super().__dir__(), *_KEYWORDS]

    def build(self, builder: DFTComponentsBuilder) -> list[Extension]:
        solvation = self.solvation.build(builder)
        log = builder.log
        p = SimpleNamespace(**self._params)  # as given
        warns = []
        wf_tol = (builder.params.convergence or {}).get('work function')
        if (p.target_potential is not None and wf_tol is not None and
                wf_tol >= p.tol):
            warns.append(
                'It appears that your work function convergence criterion '
                f'({wf_tol:g}) is higher than your desired potential '
                f'tolerance ({p.tol:g}).  This may lead to issues with '
                'potential convergence.')
        if p.always_adjust and p.target_potential is None:
            warns.append(always_adjust_warning)
        for msg in warns:
            warnings.warn(msg)
            log(textwrap.fill('Warning: ' + msg) + '\n')

        region = build_jellium_region(p.jelliumregion,
                                      cavity=solvation.cavity)
        mask_r, z1, z2 = region.mask(builder.fine_grid, builder.relpos_ac)

        charging = JelliumCharging(mask_r,
                                   pw=builder.interpolation_desc,
                                   excess_electrons=p.excess_electrons,
                                   region=region)
        # The optimizer is built in constant-charge mode too: it keeps the
        # slope settings, and the estimate, for a later switch of mode.
        optimizer = DirectOptimizer(
            slope=p.slope,
            mixer=p.mixer,
            slope_regression_depth=p.slope_regression_depth)
        # From here on the potentiostat holds the settings; __getattr__
        # reads them back from it.
        potentiostat = Potentiostat(
            charging=charging,
            target_potential=p.target_potential,
            optimizer=optimizer,
            tol=p.tol,
            max_potential_stepsize=p.max_potential_stepsize,
            max_iters=p.max_iters,
            max_reruns=p.max_reruns,
            always_adjust=p.always_adjust,
            grand_output=p.grand_output,
            log=log)

        log('Solvated jellium method (SJM):')
        log(f'  jellium region: {z1:.3f} to {z2:.3f}  # Å')
        if p.target_potential is None:
            log('  mode: constant charge')
            log(f'  excess electrons: {p.excess_electrons:+.5f}')
        else:
            log('  mode: constant potential')
            log(f'  target potential: {p.target_potential:.5f}  # V')
            log(f'  tolerance: {p.tol:.5f}  # V')
            log(f'  max iterations: {p.max_iters}')
            if p.max_potential_stepsize is not None:
                log(f'  max potential stepsize: {p.max_potential_stepsize}'
                    f' V (halved up to {p.max_reruns} times if exceeded)')
            log('  initial guess of excess electrons: '
                f'{p.excess_electrons:+.5f}')
            if p.slope is not None:
                log(f'  initial guess of slope: {p.slope:.4f}  '
                    '# V/electron')
            log(f'  slope damping (mixer): {p.mixer:.2f}')
            log('  slope regressed over the last '
                f'{p.slope_regression_depth:d} attempts')
            if p.always_adjust:
                log('  always adjust: the charge is stepped on to the next '
                    'geometry')

        self._potentiostat = potentiostat
        return [solvation, potentiostat]

    def change(self, **kwargs) -> None:
        """Change SJM parameters, before or after the first calculation.

        Takes the parameters of :class:`SJM` (all but the solvation model),
        with the same meaning.  This is the replacement for the legacy
        ``calc.set(sj={...})``.  Before the calculation has been built the
        parameters are simply stored for ``build()`` to pick up; afterwards
        they reach the running calculation too, and the density and the
        wave functions are kept, so the next ``get_potential_energy()``
        continues the SCF loop from the current solution instead of starting
        over.  So keep the SJM object in a variable and change the parameters
        through it, before and after the first calculation alike::

            sjm = SJM(...)
            atoms.calc = GPAW(..., extensions=[sjm])
            atoms.get_potential_energy()
            sjm.change(target_potential=4.3)
            atoms.get_potential_energy()

        When the target potential is changed and the slope of potential vs.
        charge is known, a first step in the number of excess electrons is
        taken right away -- unless ``excess_electrons`` is given explicitly,
        which then wins.

        The parameters can all be read back by their keyword names --
        ``sjm.target_potential``, ``sjm.tol``, ``sjm.slope``, ... -- but not
        assigned to: ``sjm.excess_electrons = ...`` raises, because the
        excess electrons have to reach the jellium as well, and only
        ``change()`` empties the results, which is what makes the next
        ``get_potential_energy()`` run the SCF loop again.
        """
        _check_keywords(kwargs)
        for key, value in kwargs.items():
            if key == 'jelliumregion':
                value = value or {}  # as in __init__
            self._params[key] = value
        pot = self._potentiostat
        if pot is None:
            return
        dft, log, opt = pot.dft, pot.log, pot.optimizer
        assert isinstance(opt, DirectOptimizer)  # the only kind SJM builds
        rerun = False  # does the SCF loop have to run again?

        if 'grand_output' in kwargs:
            pot.grand_output = kwargs['grand_output']
            if dft is not None and dft.results:
                # Only which energy is reported changes:
                pot.write_results(dft.results, log)
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
                pot.pending_excess_electrons = None
            elif kwargs.get('target_potential',
                            pot.target_potential) is None:
                warnings.warn(always_adjust_warning)
        for key in ['mixer', 'slope_regression_depth']:
            if key in kwargs:
                setattr(opt, key, kwargs[key])
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
            if dft is None:
                raise RuntimeError('No results yet - run a calculation '
                                   'before changing the jellium region.')
            charging = pot.charging
            charging.region = build_jellium_region(
                kwargs['jelliumregion'], cavity=dft.solvation.cavity)
            mask_r, z1, z2 = charging.region.mask(charging.mask_r.desc,
                                                  dft.relpos_ac)
            charging.set_region(mask_r)
            log(f'Jellium region changed to {z1:.3f} to {z2:.3f}  # Å')
            rerun = True

        if 'target_potential' in kwargs:
            target = kwargs['target_potential']
            pot.set_target_potential(target)
            if target is None:
                log('Switched to constant-charge mode with '
                    f'{self.excess_electrons:+.5f} excess electrons.')
            else:
                log(f'Target potential changed to {target:.5f} +/- '
                    f'{self.tol:.5f} V.')
                slope = opt.slope
                if ('excess_electrons' not in kwargs and slope is not None
                        and not np.isnan(pot.fermi_level)):
                    # Guess the charge needed for the new target potential:
                    pot.set_excess_electrons(
                        self.excess_electrons +
                        (target - self.get_electrode_potential()) / slope)
                    log(f'Excess electrons changed to '
                        f'{self.excess_electrons:+.5f} based on a slope of '
                        f'{slope:.4f} V/electron.')
            rerun = True

        if 'excess_electrons' in kwargs:
            pot.set_excess_electrons(kwargs['excess_electrons'])
            log(f'Excess electrons changed to {self.excess_electrons:+.5f}.')
            rerun = True

        if rerun and dft is not None:
            dft.results.clear()
            density = dft.density
            dft.ibzwfs.calculate_occs(
                dft.scf_loop.occ_calc,
                density.nvalence - density.charge + dft.pot_calc.charge)
            density.update(dft.ibzwfs)
            dft.potential, dft.energies, _ = dft.pot_calc.calculate(
                density, dft.ibzwfs, dft.potential.vHt_x)
            pot.attempts = []

    def _running(self, what: str) -> Potentiostat:
        if self._potentiostat is None:
            raise RuntimeError(f'No calculation yet - run one before {what}.')
        return self._potentiostat

    def get_electrode_potential(self) -> float:
        """Electrode potential of the calculation, in V.

        It is the potential of the last converged SCF loop, on the vacuum
        scale: the Poisson solver zeroes the electrostatic potential in the
        vacuum region, so the work function -- and with it the electrode
        potential -- is simply minus the Fermi level.  It is the same number
        as ``calc.results['electrode_potential']``, and NaN until an SCF
        loop has converged.

        Raises ``RuntimeError`` before the calculation has been built.
        """
        return self._running(
            'asking for the electrode potential').electrode_potential

    def write_sjm_traces(self, path='sjm_traces', style='z',
                         props=('potential', 'cavity', 'background_charge')):
        """Write traces of the quantities in `props` to disk.

        The electrostatic potential (``'potential'``), the solvent cavity
        (``'cavity'``) and the jellium counter charge
        (``'background_charge'``) of the calculation are written into the
        directory `path`, one file per quantity: ``<prop>.txt`` for
        ``style='z'`` (the default), averaged over x and y as a function of
        z, and ``<prop>.cube`` on the full 3-D grid for ``style='cube'``::

            atoms.get_potential_energy()
            sjm.write_sjm_traces()  # sjm_traces/*.txt
            sjm.write_sjm_traces(path='traces-3d', style='cube')

        Note that ``'potential'`` is the electrostatic potential referenced
        to the Fermi level, in eV, and that ``'background_charge'`` is the
        jellium counter-charge density in e/Bohr^3 -- it integrates to minus
        the number of excess electrons.

        A ``'z'`` trace has one row per grid point of the full mesh, so traces
        from an fd and a plane-wave run of the same cell line up row by row.
        A real-space grid with a zero boundary in z does not store the z = 0
        point; the row written for it repeats the first stored value and is
        padding rather than a computed value.

        This needs a calculation to have been run, and raises
        ``RuntimeError`` before.  It is collective: call it on all ranks.
        """
        pot = self._running('writing SJM traces')
        dft = pot.dft
        if dft is None:
            raise RuntimeError('No results yet - run a calculation before '
                               'writing SJM traces.')
        if style not in {'z', 'cube'}:
            raise ValueError(f'Unknown style: {style!r}')

        comm = dft.solvation.comm  # world
        if comm.rank == 0:
            os.makedirs(path, exist_ok=True)
        comm.barrier()

        for prop in props:
            a_r = self._trace_data(prop, pot, dft).gather(broadcast=True)
            if style == 'z':
                if comm.rank == 0:
                    _write_trace_in_z(a_r, os.path.join(path, prop + '.txt'))
            else:
                import ase.io
                # ase.io.write() carries ASE's @parallel_function wrapper: it
                # runs on the master rank and then broadcasts the outcome to
                # the whole world.  Every rank therefore has to call it -- ASE
                # does the rank-0 guarding itself.  Calling it inside a guard
                # of our own leaves that broadcast unmatched, and the stale
                # message is then picked up by the next collective, which is
                # synchronize_atoms() at the start of the next calculation.
                ase.io.write(os.path.join(path, prop + '.cube'),
                             dft.atoms, data=a_r.data)
        comm.barrier()

    def _trace_data(self, prop: str, pot: Potentiostat, dft) -> UGArray:
        """One of the quantities that write_sjm_traces() can write out."""
        grid = dft.solvation.grid
        if prop == 'potential':
            vHt_x = dft.potential.vHt_x  # Hartree
            if isinstance(vHt_x, UGArray):
                vHt_r = vHt_x.copy()
            else:
                vHt_g = vHt_x.gather()
                vHt_r = grid.empty()
                vHt_r.scatter_from(vHt_g.ifft(grid=grid.new(comm=None))
                                   if vHt_g is not None else None)
            vHt_r.data[:] = (vHt_r.data - pot.fermi_level) * Ha
            return vHt_r
        if prop == 'background_charge':
            mask_r = pot.charging.mask_r
            charge_r = mask_r.new()
            charge_r.data[:] = -self.excess_electrons * mask_r.data
            return charge_r
        if prop == 'cavity':
            try:
                g_g = dft.solvation.cavity.g_g
            except AttributeError as exc:
                raise ValueError('This cavity has no g_g attribute - '
                                 'cannot write a cavity trace.') from exc
            return grid.from_data(g_g)
        raise ValueError(f'Unknown property: {prop!r}')

    def todict(self):
        dct = self.solvation.todict()
        del dct['dipolelayer']  # always the same for SJM
        dct.update({key: self._params[key] for key in _KEYWORDS})
        return dct


#: The SJM parameters: the keywords of SJM.__init__ other than those of the
#: solvation model.  This is the one place they are listed -- change(),
#: todict() and the keyword check all read them from here.
_KEYWORDS = tuple(name for name in inspect.signature(SJM.__init__).parameters
                  if name not in {'self', 'cavity', 'dielectric',
                                  'interactions'})


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
