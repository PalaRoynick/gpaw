from __future__ import annotations

import inspect
import os
import textwrap
import warnings

import numpy as np
from ase.units import Bohr, Ha

from gpaw.core import UGArray
from gpaw.dft import ExtensionInput
from gpaw.extensions import Extension
from gpaw.new.builder import DFTComponentsBuilder
from gpaw.extensions.electrochemistry.potentiostat import (
    DirectOptimizer, JelliumCharging, Potentiostat, _from_dict,
    build_jellium_region, default_parts)
from gpaw.solvation import get_HW14_water_kwargs
from gpaw.solvation.dipole_layer import GaussianDerivativeDipole
from gpaw.solvation.extension import Solvation


always_adjust_warning = 'always_adjust has no effect in constant-charge mode'

#: The keys that change() takes: all but the parts themselves.
changeable = ('always_adjust', 'grand_output',
              *inspect.signature(JelliumCharging).parameters,
              *inspect.signature(DirectOptimizer).parameters)


class SJM(ExtensionInput):
    """Solvated jellium method.

    Electronically grand-canonical calculations, typically for simulating
    electrochemical interfaces.  A jellium counter charge is placed in the
    implicit solvent above the slab; the number of excess electrons can
    either be fixed (constant-charge mode) or adjusted until the electrode
    potential matches ``target_potential`` (constant-potential mode).

    An SJM object, and the objects given to it, belong to one calculator:
    each calculation keeps its state in them.  So give every calculator its
    own -- every image of a NEB, for instance.

    Parameters
    ----------
    solvation:
        The solvation model, held as ``sjm.solvation``: a
        :class:`gpaw.solvation.Solvation` (or a subclass of it), or a dict
        of its keyword arguments (``cavity``, ``dielectric``,
        ``interactions`` and ``dipolelayer``).  ``None`` (the default) gives
        water as in :func:`gpaw.solvation.get_HW14_water_kwargs`, with its
        effective potential made an
        :class:`~gpaw.solvation.sjm.SJMPower12Potential`, which keeps the
        solvent from behind the electrode.  Without a dipole layer it gets
        ``GaussianDerivativeDipole(zero_vacuum=True)``, which puts the
        electrostatic potential to zero in the vacuum at the top of the cell.
        The electrode potential is read off that vacuum level, so a layer of
        your own works with either ``zero_vacuum``.
    potentiostat:
        The potential control, held as ``sjm.potentiostat``: a
        :class:`Potentiostat` (or a subclass of it), or a dict of the keys
        below.  ``None`` (the default) gives a constant-charge calculation
        with no excess electrons.

    The keys of the ``potentiostat`` dict:

    target_potential:
        Electrode potential to equilibrate to, in V on the vacuum scale
        (expressed as a work function).  ``None`` (the default) gives a
        constant-charge calculation based on ``excess_electrons``.
    excess_electrons:
        Number of excess electrons in the atomic system, and (with opposite
        sign) in the jellium region.  Used as the initial guess in
        constant-potential mode.
    jelliumregion:
        Shape of the counter-charge region, as a dict.  The default is a
        slab (:class:`SlabRegion`), taking the keys ``'top'``, ``'bottom'``,
        ``'thickness'`` and ``'fix_bottom'``.  A ``'shape'`` key selects a
        different kind of region; see :func:`build_jellium_region`.
    tol:
        Tolerance for the deviation of the target potential, in V.
        Default: 0.005 V.
    max_iters, max_potential_stepsize, max_reruns:
        Limits of the equilibration; see :class:`PotentialOptimizer`.
        Defaults: 10, 2 V and 10.
    grand_output:
        Write the grand-potential energy (instead of the canonical energy)
        into the results dictionary handed to ASE, and thereby into
        trajectory files.  Default: True.
    always_adjust:
        Take one more step in the number of excess electrons even when the
        potential is already within ``tol``.  The step is not re-converged --
        it is applied to the *next* geometry -- so it costs no extra SCF
        cycles.  Together with a loose ``tol`` this lets a relaxation optimize
        the structure and the potential simultaneously.  Without
        ``target_potential`` it does nothing, and warns.  Default: False.
    slope:
        Initial guess of d(potential)/d(excess electrons), in V/electron;
        slopes regressed later are mixed into it by ``mixer``.  ``None``, or
        a value that is not negative, gives the slope of a 10 muF/cm^2 plate
        capacitor of the cell's xy area.
    mixer:
        Damping of the slope estimate, because regressing a slope can be
        noisy: ``new = mixer * old + (1 - mixer) * regressed``.  Set to 0 for
        no damping.  It also sets how fast the step in excess electrons may
        grow: a step may exceed the previous one by the fraction
        ``1 - mixer`` of the excess the slope asks for.  Default: 0.2.
    slope_regression_depth:
        Number of previous attempts that the slope is regressed over.
        Default: 4.
    charging, optimizer, potentialprobe:
        Parts of your own, in place of the default :class:`JelliumCharging`,
        :class:`DirectOptimizer` and :class:`WorkFunctionProbe`.  The keys
        that configure a default part -- ``excess_electrons`` and
        ``jelliumregion``; ``target_potential``, ``tol``, ``slope``,
        ``mixer``, ``slope_regression_depth``, ``max_potential_stepsize``,
        ``max_iters`` and ``max_reruns`` -- cannot go together with one of
        your own: give those settings to your part instead.
    """

    name = 'sjm'

    def __init__(self,
                 solvation: Solvation | dict | None = None,
                 potentiostat: Potentiostat | dict | None = None):
        if solvation is None:
            from gpaw.solvation.sjm import SJMPower12Potential
            solvation = water = get_HW14_water_kwargs()
            u = water['cavity'].effective_potential
            water['cavity'].effective_potential = SJMPower12Potential(
                u.atomic_radii, u.u0)
        self.solvation = _from_dict(solvation, Solvation)
        if not self.solvation.dipolelayer:
            self.solvation.dipolelayer = GaussianDerivativeDipole(
                zero_vacuum=True)
        self.potentiostat = (potentiostat
                             if isinstance(potentiostat, Potentiostat) else
                             Potentiostat.from_dict(potentiostat))

    def build(self, builder: DFTComponentsBuilder) -> list[Extension]:
        solvation = self.solvation.build(builder)
        pot = self.potentiostat
        opt = pot.optimizer
        pot.log = log = builder.log
        pot.solvation = solvation
        warns = []
        wf_tol = (builder.params.convergence or {}).get('work function')
        if (opt.target_potential is not None and wf_tol is not None and
                wf_tol >= opt.tol):
            warns.append(
                'It appears that your work function convergence criterion '
                f'({wf_tol:g}) is higher than your desired potential '
                f'tolerance ({opt.tol:g}).  This may lead to issues with '
                'potential convergence.')
        if pot.always_adjust and opt.target_potential is None:
            warns.append(always_adjust_warning)
        for msg in warns:
            warnings.warn(msg)
            log(textwrap.fill('Warning: ' + msg) + '\n')

        z1, z2 = pot.charging.initialize(builder.fine_grid,
                                         builder.interpolation_desc,
                                         builder.relpos_ac,
                                         cavity=solvation.cavity)
        # A rebuilt calculation starts a fresh equilibration:
        pot.new_geometry()

        log('Solvated jellium method (SJM):')
        log(f'  jellium region: {z1:.3f} to {z2:.3f}  # Å')
        for key, default in default_parts.items():
            if type(getattr(pot, key)) is not default:
                log(f'  {key}: {type(getattr(pot, key)).__name__}')
        if opt.target_potential is None:
            log('  mode: constant charge')
            log(f'  excess electrons: {pot.excess_electrons:+.5f}')
        else:
            log('  mode: constant potential')
            log(f'  target potential: {opt.target_potential:.5f}  # V')
            log(f'  tolerance: {opt.tol:.5f}  # V')
            log(f'  max iterations: {opt.max_iters}')
            if opt.max_potential_stepsize is not None:
                log(f'  max potential stepsize: {opt.max_potential_stepsize}'
                    f' V (halved up to {opt.max_reruns} times if exceeded)')
            log('  initial guess of excess electrons: '
                f'{pot.excess_electrons:+.5f}')
            if opt.slope is not None:
                log(f'  initial guess of slope: {opt.slope:.4f}  '
                    '# V/electron')
            if isinstance(opt, DirectOptimizer):
                log(f'  slope damping (mixer): {opt.mixer:.2f}')
                log('  slope regressed over the last '
                    f'{opt.slope_regression_depth:d} attempts')
            if pot.always_adjust:
                log('  always adjust: the charge is stepped on to the next '
                    'geometry')
        return [solvation, pot]

    def change(self, potentiostat: dict) -> None:
        """Change potentiostat settings, before or after a calculation.

        Takes the keys of the ``potentiostat`` dict of :class:`SJM`, with
        the same meaning -- all but the parts, which are fixed when the SJM
        object is created.  After a calculation the settings reach it
        right away, and the density and the wave functions are kept, so the
        next ``get_potential_energy()`` continues the SCF loop from the
        current solution instead of starting over.  So keep the SJM object
        in a variable and change the settings through it::

            sjm = SJM(...)
            atoms.calc = GPAW(..., extensions=[sjm])
            atoms.get_potential_energy()
            sjm.change(potentiostat={'target_potential': 4.3})
            atoms.get_potential_energy()

        A new target drops a step staged by ``always_adjust``.  After a
        converged calculation, and with a negative slope estimate, it also
        steps the excess electrons toward the new target, unless
        ``excess_electrons`` is given.  A new ``slope`` or jellium region
        makes the optimizer forget its attempts.

        The live values are read off the potentiostat and its parts:
        ``sjm.potentiostat.excess_electrons``,
        ``sjm.potentiostat.optimizer.target_potential``, ...  Setting them
        there directly skips what this does on top: clearing the results and
        recomputing the occupations, density and potential, which makes the
        next ``get_potential_energy()`` run the SCF loop again.

        Raises ``ValueError`` for an unknown key, or for ``mixer`` and
        ``slope_regression_depth`` on an optimizer without them, and
        ``RuntimeError`` for a jellium region on a built calculation that
        has not run yet.
        """
        pot = self.potentiostat
        dft, log, opt = pot.dft, pot.log, pot.optimizer
        for key in potentiostat:
            if key not in changeable:
                raise ValueError(
                    f'Unexpected potentiostat key: {key}.  Allowed are: '
                    f'{", ".join(sorted(changeable))}.')
            if key in {'mixer', 'slope_regression_depth'} and not hasattr(
                    opt, key):
                raise ValueError(f'{type(opt).__name__} has no {key}.')
        rerun = False  # does the SCF loop have to run again?

        if 'grand_output' in potentiostat:
            pot.grand_output = potentiostat['grand_output']
            if dft is not None and dft.results:
                # Only which energy is reported changes:
                pot.write_results(dft.results, log)
        if 'tol' in potentiostat:
            opt.tol = potentiostat['tol']
            if (opt.target_potential is not None and
                    not opt.converged(self.get_electrode_potential())):
                rerun = True

        if 'always_adjust' in potentiostat:
            pot.always_adjust = potentiostat['always_adjust']
            if not pot.always_adjust:
                pot.pending_excess_electrons = None
            elif potentiostat.get('target_potential',
                                  opt.target_potential) is None:
                warnings.warn(always_adjust_warning)
        for key in ['mixer', 'slope_regression_depth', 'max_iters',
                    'max_reruns', 'max_potential_stepsize']:
            if key in potentiostat:
                setattr(opt, key, potentiostat[key])
        if 'slope' in potentiostat:
            # Whatever the user says replaces the estimate, so the attempts
            # behind the old one go too:
            opt.reset()
            opt.slope = potentiostat['slope']
            if opt.slope is None:
                log('Slope estimate reset; it will be guessed again.')
            else:
                log(f'Slope set to {opt.slope:.4f} V/electron.')

        if 'jelliumregion' in potentiostat:
            charging = pot.charging
            if dft is None and hasattr(charging, 'mask_r'):
                raise RuntimeError('Run the built calculation before '
                                   'changing its jellium region.')
            charging.region = build_jellium_region(
                potentiostat['jelliumregion'], cavity=self.solvation.cavity)
            if dft is not None:
                mask_r, z1, z2 = charging.region.mask(charging.mask_r.desc,
                                                      dft.relpos_ac)
                charging.set_region(mask_r)
                opt.forget_attempts()  # they belong to the old region
                log(f'Jellium region changed to {z1:.3f} to {z2:.3f}  # Å')
                rerun = True

        if 'target_potential' in potentiostat:
            target = potentiostat['target_potential']
            opt.set_target_potential(target)
            pot.pending_excess_electrons = None  # staged for the old target
            if target is None:
                log('Switched to constant-charge mode with '
                    f'{pot.excess_electrons:+.5f} excess electrons.')
            else:
                log(f'Target potential changed to {target:.5f} +/- '
                    f'{opt.tol:.5f} V.')
                slope = opt.slope
                if ('excess_electrons' not in potentiostat and
                        slope is not None and slope < 0 and
                        not np.isnan(self.get_electrode_potential())):
                    # Guess the charge needed for the new target potential:
                    pot.excess_electrons += (
                        target - self.get_electrode_potential()) / slope
                    log(f'Excess electrons changed to '
                        f'{pot.excess_electrons:+.5f} based on a slope of '
                        f'{slope:.4f} V/electron.')
            rerun = True

        if 'excess_electrons' in potentiostat:
            pot.excess_electrons = potentiostat['excess_electrons']
            pot.pending_excess_electrons = None  # an explicit value wins
            log(f'Excess electrons changed to {pot.excess_electrons:+.5f}.')
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
            pot.attempts, pot.reruns = [], 0

    def get_electrode_potential(self) -> float:
        """Electrode potential of the last converged SCF loop, in V.

        As read by the potential probe; for the default
        :class:`WorkFunctionProbe` it is the work function at the top of the
        cell.  The same as
        ``calc.results['electrode_potential']``; NaN until an SCF loop has
        converged.
        """
        return self.potentiostat.electrode_potential

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

        A ``'z'`` trace has one row per point of the full mesh, so fd and
        plane-wave traces of the same cell line up row by row (see
        ``_write_trace_in_z()``).

        This needs a calculation to have been run, and raises
        ``RuntimeError`` before.  It is collective: call it on all ranks.
        """
        pot = self.potentiostat
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
                # Collective: ase.io.write() broadcasts its outcome from rank
                # 0, so every rank must call it.  Under a rank-0 guard the
                # broadcast is left for the next collective to pick up.
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
            charge_r.data[:] = -pot.excess_electrons * mask_r.data
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
        return {key: {'name': type(part).__name__, **part.todict()}
                for key, part in [('solvation', self.solvation),
                                  ('potentiostat', self.potentiostat)]}


def _write_trace_in_z(a_r: UGArray, filename: str) -> None:
    """Write a quantity as a function of the z coordinate only.

    ``a_r`` must be a serial (gathered) array.

    A real-space grid with a zero boundary in z does not store the z = 0
    point, which a plane-wave (FFT) grid does.  That row is padded with a
    copy of the first stored value, so traces of the same cell have the same
    rows in every mode.
    """
    a_z = a_r.data.mean(axis=(0, 1))
    z_z = a_r.desc.xyz()[0, 0, :, 2] * Bohr
    if a_r.desc.zerobc_c[2]:
        a_z = np.concatenate([a_z[:1], a_z])
        z_z = np.concatenate([[0.0], z_z])
    with open(filename, 'w', encoding='utf-8') as out:
        for z, val in zip(z_z, a_z):
            out.write(f'{z:f} {val:1.8f}\n')
