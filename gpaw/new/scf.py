from __future__ import annotations

import itertools
import time
import warnings
from collections.abc import Callable
from math import inf
from types import SimpleNamespace
from typing import Any

import numpy as np
from gpaw import KohnShamConvergenceError
from gpaw.core.atom_arrays import AtomDistribution
from gpaw.convergence_criteria import (Criterion, check_convergence,
                                       dict2criterion)
from gpaw.mpi import serial_comm
from gpaw.new.energies import DFTEnergies
from gpaw.new.ibzwfs import IBZWaveFunctions
from gpaw.new.logger import indent
from gpaw.typing import Array2D


class TooFewBandsError(KohnShamConvergenceError):
    """Not enough bands for CBM+x convergence criterion."""


class SCFLoop:
    def __init__(self,
                 hamiltonian,
                 occ_calc,
                 eigensolver,
                 mixer,
                 comm,
                 convergence,
                 maxiter,
                 potential_parallelization=None):
        self.hamiltonian = hamiltonian
        self.eigensolver = eigensolver
        self.mixer = mixer
        self.occ_calc = occ_calc
        self.comm = comm
        self.convergence = create_convergence_criteria(convergence)
        self.maxiter = maxiter
        self.niter = 0
        self.update_density_and_potential = True
        self.fix_fermi_level = False
        self.potential_parallelization = potential_parallelization

    def __repr__(self):
        return 'SCFLoop(...)'

    def __str__(self):
        return (f'eigensolver:\n{indent(self.eigensolver)}\n'
                f'{self.mixer}\n'
                f'occupation numbers:\n{indent(self.occ_calc)}\n')

    def iterate(self,
                ibzwfs: IBZWaveFunctions,
                density,
                potential,
                energies: DFTEnergies,
                pot_calc,
                *,
                maxiter=None,
                calculate_forces=None,
                log=None):
        cc = self.convergence
        maxiter = maxiter or self.maxiter

        if log:
            log('convergence criteria:')
            for criterion in cc.values():
                if criterion.description is not None:
                    log('- ' + criterion.description)
            log(f'maximum number of iterations: {self.maxiter}\n')

        self.mixer.reset()

        self.occ_calc.initialize_reference_orbitals()

        if self.update_density_and_potential:
            dens_error = self.mixer.mix(density)
        else:
            dens_error = 0.0

        for self.niter in itertools.count(start=1):
            eig_error, wfs_error, energies = self.eigensolver.iterate(
                ibzwfs, density, potential,
                self.hamiltonian, pot_calc, energies)
            nelectrons = density.nvalence - density.charge + pot_calc.charge
            e_band, e_entropy, e_extrapolation = ibzwfs.calculate_occs(
                self.occ_calc,
                nelectrons,
                fix_fermi_level=self.fix_fermi_level)

            if hasattr(self.hamiltonian, 'hybrid_energy_contributions'):
                ecc, evc, evv, dekin = (
                    self.hamiltonian.hybrid_energy_contributions())
                energies.set(hybrid_xc_cc=ecc,
                             hybrid_xc_vc=evc,
                             hybrid_xc_vv=evv,
                             hybrid_kinetic_correction=dekin)

            energies.set(band=e_band,
                         entropy=e_entropy,
                         extrapolation=e_extrapolation)

            ctx = SCFContext(
                log, self.niter, energies,
                ibzwfs, density, potential,
                wfs_error, dens_error, eig_error,
                self.comm, calculate_forces,
                pot_calc, self.update_density_and_potential)

            yield ctx

            converged, converged_items, entries = check_convergence(cc, ctx)
            nconverged = self.comm.sum_scalar(int(converged))
            assert nconverged in [0, self.comm.size], converged_items

            if log:
                write_iteration(cc, converged_items, entries, ctx, log)

            if converged:
                converged = all(
                    ext.post_scf_convergence(
                        ibzwfs, nelectrons, self.occ_calc, self.mixer, log)
                    for ext in pot_calc.extensions)
                if converged:
                    break

            if self.niter == maxiter:
                if wfs_error < inf:
                    raise KohnShamConvergenceError
                raise TooFewBandsError

            if self.update_density_and_potential:
                density.update(ibzwfs, ked=pot_calc.xc.type == 'MGGA')
                dens_error = self.mixer.mix(density)
                potential, energies = calculate_potential(
                    density, potential, pot_calc, ibzwfs, self.comm,
                    self.potential_parallelization)
                energies.sanity_check()

        self.eigensolver.postprocess(
            ibzwfs, density, potential, self.hamiltonian,
            maxiter=maxiter, cc=cc, log=log)


def calculate_potential(density,
                        potential,
                        pot_calc,
                        ibzwfs,
                        world_comm,
                        potential_parallelization):
    kpt_band_comm = ibzwfs.kpt_band_comm
    use_full = potential_parallelization == 'full'
    if not use_full or kpt_band_comm.size == 1:
        potential, energies, _ = pot_calc.calculate(
            density, ibzwfs, potential.vHt_x)
        return potential, energies

    if getattr(density.nt_sR, 'data', None) is None:
        potential, energies, _ = pot_calc.calculate(
            density, ibzwfs, potential.vHt_x)
        return potential, energies

    disabled_reason = getattr(pot_calc, '_full_potential_disabled_reason',
                              None)
    if disabled_reason is not None:
        potential, energies, _ = pot_calc.calculate(
            density, ibzwfs, potential.vHt_x)
        return potential, energies

    unsupported_reason = full_potential_unsupported_reason(pot_calc, density)
    if unsupported_reason is not None:
        pot_calc._full_potential_disabled_reason = unsupported_reason
        warn_full_potential_fallback(unsupported_reason, world_comm)
        potential, energies, _ = pot_calc.calculate(
            density, ibzwfs, potential.vHt_x)
        return potential, energies

    full_density = distribute_density_to_all_ranks(
        density, kpt_band_comm, world_comm)
    construction_error = None
    try:
        full_pot_calc = create_full_potential_calculator(pot_calc,
                                                         full_density)
    except ValueError as err:
        if not is_full_potential_construction_error(err):
            raise
        construction_error = str(err)

    construction_failed = world_comm.sum_scalar(
        int(construction_error is not None))
    if construction_failed:
        reason = (construction_error or
                  'full potential calculator construction failed on '
                  'another rank')
        pot_calc._full_potential_disabled_reason = reason
        warn_full_potential_fallback(reason, world_comm)
        potential, energies, _ = pot_calc.calculate(
            density, ibzwfs, potential.vHt_x)
        return potential, energies
    if full_pot_calc is None:
        warn_full_potential_fallback(
            f'{pot_calc.__class__.__name__} is not supported by '
            'full potential parallelization',
            world_comm)
        potential, energies, _ = pot_calc.calculate(
            density, ibzwfs, potential.vHt_x)
        return potential, energies

    full_vHt_x = None
    if potential.vHt_x is not None:
        full_vHt_x = potential.vHt_x.redist(
            electrostatic_desc(full_pot_calc),
            kpt_band_comm,
            serial_comm)

    full_potential, energies, _ = full_pot_calc.calculate(
        full_density, None, full_vHt_x, serial_comm)
    potential = full_potential.redist(
        pot_calc.grid,
        electrostatic_desc(pot_calc),
        density.D_asii.layout.atomdist,
        serial_comm,
        kpt_band_comm)
    return potential, energies


def distribute_density_to_all_ranks(density,
                                    kpt_band_comm,
                                    world_comm):
    grid = density.nt_sR.desc
    full_grid = grid.new(comm=world_comm)
    full_atomdist = AtomDistribution(
        full_grid.ranks_from_fractional_positions(density.nct_aX.relpos_ac),
        full_grid.comm)
    xdesc = density_xdesc(density)
    full_xdesc = xdesc.new(comm=world_comm)

    return density.redist(full_grid,
                          full_xdesc,
                          full_atomdist,
                          kpt_band_comm,
                          serial_comm)


def distribute_density_to_all_ranks_and_back(density,
                                             kpt_band_comm,
                                             world_comm):
    if kpt_band_comm.size == 1:
        return density

    if getattr(density.nt_sR, 'data', None) is None:
        return density

    grid = density.nt_sR.desc
    xdesc = density_xdesc(density)
    full_density = distribute_density_to_all_ranks(
        density, kpt_band_comm, world_comm)

    roundtrip_density = full_density.redist(
        grid,
        xdesc,
        density.D_asii.layout.atomdist,
        serial_comm,
        kpt_band_comm)

    density.nt_sR = roundtrip_density.nt_sR
    density.taut_sR = roundtrip_density.taut_sR
    density.D_asii = roundtrip_density.D_asii
    density.nct_aX = roundtrip_density.nct_aX
    density.tauct_aX = roundtrip_density.tauct_aX
    density.grid = roundtrip_density.grid
    density._nct_R = None
    density._tauct_R = None
    density.symplan = None
    return density


def density_xdesc(density):
    if hasattr(density.nct_aX, 'grid'):
        return density.nct_aX.grid
    return density.nct_aX.pw


def warn_full_potential_fallback(reason, comm):
    if comm.rank == 0:
        warnings.warn(
            f"Falling back from parallel={{'potential': 'full'}}: {reason}",
            RuntimeWarning)


def full_potential_unsupported_reason(pot_calc, density):
    if density.ncomponents != 1:
        return ('spin-dependent densities are not supported by full '
                'potential parallelization')

    if pot_calc.xc.name == 'QNA':
        return 'QNA is not supported by full potential parallelization'

    if pot_calc.extensions:
        return 'extensions are not supported by full potential parallelization'

    if hasattr(pot_calc, 'pw') and not pot_calc.grid.pbc_c.all():
        return ('non-periodic plane-wave calculations are not supported by '
                'full potential parallelization')

    poisson_solver = getattr(pot_calc, 'poisson_solver', None)
    solver = getattr(poisson_solver, 'solver', None)
    if getattr(solver, 'use_charge_center', False):
        return ('Poisson solvers using charge-center compensation are not '
                'supported by full potential parallelization')

    return None


def is_full_potential_construction_error(err):
    return str(err).startswith('Cannot construct interpolator.')


def create_full_potential_calculator(pot_calc, full_density):
    signature = full_potential_signature(pot_calc, full_density)
    cached = getattr(pot_calc, '_full_potential_calculator', None)
    if cached is not None and getattr(cached, '_signature', None) == signature:
        return cached

    full_pot_calc = _create_full_potential_calculator(pot_calc, full_density)
    if full_pot_calc is not None:
        full_pot_calc.density_atomdist = full_density.D_asii.layout.atomdist
        full_pot_calc._signature = signature
        pot_calc._full_potential_calculator = full_pot_calc
    return full_pot_calc


def full_potential_signature(pot_calc, full_density):
    grid = full_density.nt_sR.desc
    atomdist = full_density.D_asii.layout.atomdist
    return (pot_calc.__class__,
            tuple(grid.size_c),
            grid.comm.size,
            grid.comm.rank,
            tuple(atomdist.rank_a.tolist()),
            atomdist.comm.size,
            atomdist.comm.rank,
            pot_calc.relpos_ac.tobytes())


def _create_full_potential_calculator(pot_calc, full_density):
    from gpaw.new.fd.pot_calc import FDPotentialCalculator
    from gpaw.new.pw.pot_calc import PlaneWavePotentialCalculator

    if isinstance(pot_calc, FDPotentialCalculator):
        return create_full_fd_potential_calculator(pot_calc, full_density)
    if isinstance(pot_calc, PlaneWavePotentialCalculator):
        return create_full_pw_potential_calculator(pot_calc, full_density)
    return None


def create_full_fd_potential_calculator(pot_calc, full_density):
    from gpaw.new.fd.pot_calc import FDPotentialCalculator

    full_grid = full_density.nt_sR.desc
    full_fine_grid = pot_calc.fine_grid.new(comm=full_grid.comm)
    try:
        poisson_solver = pot_calc.poisson_solver.new(
            grid=full_fine_grid, xp=pot_calc.xp)
    except NotImplementedError:
        return None

    xc = pot_calc.xc.new(grid=full_fine_grid, xp=pot_calc.xp)
    return FDPotentialCalculator(
        full_grid,
        full_fine_grid,
        pot_calc.setups,
        xc,
        poisson_solver,
        relpos_ac=pot_calc.relpos_ac,
        atomdist=full_density.D_asii.layout.atomdist,
        interpolation_stencil_range=pot_calc.interpolation_stencil_range,
        extensions=[],
        xp=pot_calc.xp)


def create_full_pw_potential_calculator(pot_calc, full_density):
    from gpaw.new.pw.pot_calc import PlaneWavePotentialCalculator

    if pot_calc.xc.exx_fraction != 0.0:
        return None

    full_grid = full_density.nt_sR.desc
    full_fine_grid = pot_calc.fine_grid.new(comm=full_grid.comm)
    full_pw = pot_calc.pw.new(comm=full_grid.comm)
    try:
        poisson_solver = pot_calc.poisson_solver.new(
            pwg=pot_calc.poisson_solver.pwg.new(comm=full_grid.comm),
            pwh=electrostatic_desc(pot_calc).new(comm=full_grid.comm),
            grid=full_grid,
            fine_grid=full_fine_grid,
            setups=pot_calc.setups,
            relpos_ac=pot_calc.relpos_ac,
            atomdist=full_density.D_asii.layout.atomdist,
            xp=pot_calc.xp)
    except NotImplementedError:
        return None

    xc = pot_calc.xc.new(grid=full_fine_grid, xp=pot_calc.xp)
    return PlaneWavePotentialCalculator(
        full_grid,
        full_fine_grid,
        full_pw,
        pot_calc.setups,
        xc,
        poisson_solver,
        relpos_ac=pot_calc.relpos_ac,
        atomdist=full_density.D_asii.layout.atomdist,
        soc=pot_calc.soc,
        xp=pot_calc.xp,
        extensions=[])


def electrostatic_desc(pot_calc):
    if hasattr(pot_calc, 'pw'):
        ps = pot_calc.poisson_solver
        if hasattr(ps, 'pwh'):
            return ps.pwh
        if hasattr(ps, 'pwg'):
            return ps.pwg
    return pot_calc.fine_grid


class SCFContext:
    def __init__(self,
                 log,
                 niter: int,
                 energies: DFTEnergies,
                 ibzwfs,
                 density,
                 potential,
                 wfs_error: float,
                 dens_error: float,
                 eig_error: float,
                 comm,
                 calculate_forces: Callable[[], Array2D],
                 pot_calc,
                 update_density_and_potential):
        self.log = log
        self.niter = niter
        self.energies = energies
        self.ibzwfs = ibzwfs
        self.density = density
        self.potential = potential
        energy = energies.total_extrapolated
        self.ham = SimpleNamespace(e_total_extrapolated=energy,
                                   get_workfunctions=self._get_workfunctions)
        self.wfs = SimpleNamespace(nvalence=density.nvalence +
                                   pot_calc.charge,
                                   world=comm,
                                   eigensolver=SimpleNamespace(
                                       error=wfs_error),
                                   nspins=density.ndensities,
                                   collinear=density.collinear)
        self.dens = SimpleNamespace(
            calculate_magnetic_moments=density.calculate_magnetic_moments,
            fixed=not update_density_and_potential,
            error=dens_error)
        self.eig_error = eig_error
        self.calculate_forces = calculate_forces
        self.poisson_solver = pot_calc.poisson_solver

    def _get_workfunctions(self, _):
        vacuum_level = self.potential.get_vacuum_level()
        (fermi_level,) = self.ibzwfs.fermi_levels
        wf = vacuum_level - fermi_level
        delta = self.poisson_solver.dipole_layer_correction()
        return np.array([wf + delta, wf - delta])


def create_convergence_criteria(criteria: dict[str, Any]
                                ) -> dict[str, Criterion]:
    criteria = criteria.copy()
    for k, v in [('energy', 0.0005),        # eV / electron
                 ('density', 1.0e-4),       # electrons / electron
                 ('eigenstates', 4.0e-8)]:  # eV^2 / electron
        if k not in criteria:
            criteria[k] = v
    # Gather convergence criteria for SCF loop.
    custom = criteria.pop('custom', [])
    for name, criterion in criteria.items():
        if hasattr(criterion, 'todict'):
            # 'Copy' so no two calculators share an instance.
            criteria[name] = dict2criterion(criterion.todict())
        else:
            criteria[name] = dict2criterion({name: criterion})

    if not isinstance(custom, (list, tuple)):
        custom = [custom]
    for criterion in custom:
        if isinstance(criterion, dict):  # from .gpw file
            msg = ('Custom convergence criterion "{:s}" encountered, '
                   'which GPAW does not know how to load. This '
                   'criterion is NOT enabled; you may want to manually'
                   ' set it.'.format(criterion['name']))
            warnings.warn(msg)
            continue

        criteria[criterion.name] = criterion
        msg = ('Custom convergence criterion {:s} encountered. '
               'Please be sure that each calculator is fed a '
               'unique instance of this criterion. '
               'Note that if you save the calculator instance to '
               'a .gpw file you may not be able to re-open it. '
               .format(criterion.name))
        warnings.warn(msg)

    for criterion in criteria.values():
        criterion.reset()

    return criteria


def write_iteration(criteria, converged_items, entries, ctx, log):
    custom = (set(criteria) -
              {'energy', 'eigenstates', 'density'})

    eigensolver_name = getattr(ctx.wfs.eigensolver, 'name', None)
    print_iloop = False
    if eigensolver_name == 'etdm-fdpw':
        if ctx.wfs.eigensolver.iloop is not None or \
                ctx.wfs.eigensolver.outer_iloop is not None:
            print_iloop = True

    if ctx.niter == 1:
        header = '|iter     |     time |      energy | eigst |  dens |'
        for name in custom:
            criterion = criteria[name]
            header += f'{criterion.tablename:>5s} |'
        if ctx.wfs.nspins == 2:
            header += '{:>8s} |'.format('magmom')
        if print_iloop:
            header += '{:>12s}|'.format('inner loop')
        log('SCF iterations')
        log(header)
        log(''.join(c if c == '|' else '-' for c in header))

    def format_conv(fmt: str, name: str) -> str:
        """Add "c" to number and color it green if converged."""
        txt = fmt.format(entries.get(name, ''))
        if converged_items.get(name):
            return log.green + txt + log.reset + 'c|'
        return txt + ' |'

    # Iterations and time.
    now = time.localtime()
    line = ('|iter:{:4d}| {:02d}:{:02d}:{:02d} |'
            .format(ctx.niter, *now[3:6]))

    # Energy.
    line += format_conv('{:>12s}', 'energy')

    # Eigenstates.
    line += format_conv('{:>6s}', 'eigenstates')

    # Density.
    line += format_conv('{:>6s}', 'density')

    # Custom criteria (optional).
    for name in custom:
        line += format_conv('{:>5s}', name)

    # Magnetic moment (optional).
    if ctx.wfs.nspins == 2 or not ctx.wfs.collinear:
        totmom_v, _ = ctx.dens.calculate_magnetic_moments()
        if ctx.wfs.collinear:
            line += f'{totmom_v[2]:+9.4f}|'
        else:
            line += ' {:+.1f},{:+.1f},{:+.1f}|'.format(*totmom_v)

    # Inner loop etdm
    if print_iloop:
        iloop_counter = (ctx.wfs.eigensolver.eg_count_iloop +
                         ctx.wfs.eigensolver.eg_count_outer_iloop)
        line += (f'{iloop_counter:12d}|')

    log(line.rstrip())
    log.fd.flush()
