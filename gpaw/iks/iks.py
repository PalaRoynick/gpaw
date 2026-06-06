from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np
from ase.units import Ha
from gpaw.calculator import GPAW
from gpaw.old.density import Density
from gpaw.iks import tools as iks_tools
from gpaw.iks import optimize as iks_optimize

from gpaw.iks.minres import iKSLinearSolver
from gpaw.sphere.lebedev import Y_nL, weight_n
from gpaw.utilities import pack_hermitian
from gpaw.xc.numericalxc import FermiAmaldi, NumericalXC, PAWPotMode
from gpaw.new.backwards_compatibility import FakeDensity

if TYPE_CHECKING:
    from collections.abc import MutableMapping
    from typing import Any, Optional

    from ase import Atoms
    from gpaw.old.arraydict import ArrayDict
    from gpaw.old.grid_descriptor import GridDescriptor
    from gpaw.typing import Array2D, ArrayND


class DegeneracyDescriptor:
    def __init__(self, kpt, bd=None, tolerance=1e-3, occ_tolerance=1e-10):
        self.kpt = kpt
        self.tolerance = tolerance
        self.occ_tolerance = occ_tolerance
        self.blocks = []
        self.blocks_occ = []
        self.occ = np.empty(1)
        self.n_d = []
        self.bd = bd
        self.vdiff_nn = None
        self.idx_map = {}

    def update(self):
        f_n = self.kpt.f_n
        if self.bd is not None:
            f_n = self.bd.collect(f_n, broadcast=True)
        self.occ = np.flatnonzero(f_n > self.occ_tolerance)
        eps_n = self.kpt.eps_n
        sort_idx = np.argsort(eps_n)
        eps_sorted = eps_n[sort_idx]
        gaps = np.diff(eps_sorted) > self.tolerance
        break_indices = np.where(gaps)[0] + 1
        groups = np.split(sort_idx, break_indices)
        self.blocks = [g.tolist() for g in groups]
        self.blocks_occ = [
            g[np.isin(g, self.occ)].tolist()
            for g in groups
        ]
        self.n_d = [len(b) for b in self.blocks]
        self.blocks_occ = [b for b in self.blocks_occ if b]
        self.idx_map = {band_idx: array_idx for array_idx,
                        band_idx in enumerate(self.occ)}


# throughout it's been assumed that finegd and gd (both for the hamiltonian and the density)
# all have the same distribution which (I think) is not always true.
# for example look at density.get_all_electron_density to see how to properly handle different distributions.

class iKS:
    # TODO: add explaination on lambdareg, lambdacore and n_regularization, degeneracy_tol, occ_tol, weights_sum_spins
    """
    OECO-iKS: Orbital Expansion Constrained Optimization Inverse Kohn–Sham procedure.
    This class implements an inverse Kohn–Sham (iKS) approach using orbital expansion
    and constrained optimization to recover the exchange–correlation (xc) potential
    that reproduces a target density.

    Parameters
    ----------
    target : GPAW or Density
        The target density or reference GPAW calculation whose density is used in the
        inversion procedure. If a separate density matrix (``P_snn``) is provided, the
        wave function object (``wfs``) from ``target`` is also used to construct the
        corresponding density. A typical use case is a Hartree–Fock calculation to
        obtain a local exact-exchange potential.

    calc : GPAW
        A GPAW calculator from which all DFT parameters for the iKS procedure are inherited.
        Note that differing grid spacings or plane-wave cutoffs between ``target`` and
        ``calc`` are not supported. If the exchange–correlation functional is not a
        :class:`~gpaw.xc.numericalxc.NumericalXC` object, it will be replaced by the
        Fermi–Amaldi potential (initialized with the same PAW setups as
        ``calc.hamiltonian.xc``) as a starting guess. The resulting inverted potentials
        can be saved by writing the calculator to a ``.gpw`` file (e.g.,
        ``calc.hamiltonian.xc.vxc_sg`` for the smooth grid and ``xc.vxc_xasz`` for the
        atomic part), or by calling ``xc.pack_pots()``. These potentials are restored upon
        reading the file. Observers attached to ``calc`` are triggered per optimization
        iteration, not per SCF iteration.

    P_snn : array or None, optional
        A spin-dependent density matrix from a higher-level calculation. If omitted,
        both the smooth density and atomic density matrix are taken directly from ``target``.

    weights : str or array, default="inverse_density"
        Specifies how density differences are weighted in real space. Supported predefined
        weights (relative to the target density) include:
        ``"uniform"``, ``"density_sqrt"``, ``"density"``, ``"inverse_density"``,
        ``"inverse_density_squared"``, ``"mask"``, and ``"fr"``.

    atomic_weights : str or mapping, default="radial_inverse_density"
        Weighting scheme for density differences inside the augmentation spheres. Can be
        specified as a predefined string
        (same options as ``weights`` with/without a "radial_" to ommit/include the angular components)
        or a custom mapping
        from atom indices to arrays. Custom weights must have shapes:
          - Angular: ``{a: (s, n_Lebedev, rgd.N)}`` [experimental]
          - Radial: ``{a: (s, 1, rgd.N)}``
        If assigning custom angular weights, set ``atomic_weights = 'custom'`` before
        calling :func:`~initialize()`.

    atoms : Atoms or None, optional
        An ASE ``Atoms`` object used to initialize the calculator.
        Defaults to ``calc.atoms or atoms or target.atoms``.

    stencil : int, default=3
        The stencil radius used for Jacobian interpolation to the fine grid and for evaluating
        the xc energy via the virial theorem (when ``virial_xc_energy=True``).

    nfixed : int, default=1
        Minimum number of SCF steps performed with fixed potentials after each update.
        By default, the SCF loop stops as soon as convergence is achieved. If ``nfixed=0``,
        only one SCF iteration is performed in each iKS step.

    linear_solver: dict | str | None, default = 'minres'
        The linear solver algorithm used to calcualte the lagrange multipliers p_n for the gradients.
        iKS uses `scipy.sparse.linalg` linear solvers except for the minres which a locally modified version is used.
        linear_solver=None results in a purely orbital expansion scheme.
        The additional keywords and arguments are passed to the underlying method directly.
        For example linear_solver={'name': 'minres', 'rtol':1e-6, 'maxiter':600, 'ortho': True}.

    use_coarse_grid : bool, default=False
        If True, the inversion is done on the coarse grid and the resulting potential is
        interpolated to the fine grid. If False, the Jacobian is interpolated and the
        xc potential remains on the fine grid.

    kin_en_using_band : bool, default=False
        If True, evaluates the kinetic energy using band energies.


    log_l1_ae : bool, default=True
        If True, reports the L1 error of the all-electron density at the final iteration.
        (disable this if the used setups does not support all-electron densities, e.g. ``ae`` or ``hgh``).

    nxc_kwargs : dict, optional
        Additional keyword arguments passed to the NumericalXC functional.
        This options are ignored if ``calc.hamiltonian.xc`` is already a :class:`~gpaw.xc.numericalxc.NumericalXC`.
        The most important options are:
            paw_potmode : str, default="lebedev"
                Determines how atomic xc-potentials are treated during optimization:
                  - ``"lebedev"``: Potentials and gradients are represented on Lebedev quadratures [recommended].
                  - ``"angular"``: Only the angular components of atomic xc-potentials are optimized.
                  - ``"direct"``: The xc corrections to the Hamiltonian :math:`\\delta \\Delta E_{xc} / \\delta D^a_{i_1 i_2}` are inverted instead of the atomic potentials.
                  - ``"none"``: PAW corrections are set to zero.
            virial_xc_energy : bool, default=False
                If True, evaluates the exchange–correlation energy using the virial theorem.
                Otherwise, the xc kernel is treated electrostatically. Ignored if the functional is
                already a .
        For more information look at the documentation of :class:`~gpaw.xc.numericalxc.NumericalXC`.

    Examples
    --------
    A basic example of running an iKS calculation for a water molecule. The target
    density is first computed using PBE. Then, the iKS calculator is initialized
    with a :class:`~gpaw.xc.numericalxc.NumericalXC` (LDA-based) and used to recover
    the xc potential that reproduces the target density.

    >>> from ase.build import molecule
    >>> from gpaw import GPAW
    >>> from gpaw.iks import iKS
    >>> from gpaw.xc.numericalxc import NumericalXC
    >>>
    >>> # 1. Set up the target calculation
    >>> atoms = molecule('H2O')
    >>> atoms.center(vacuum=3.5)
    >>> target_calc = GPAW(mode='pw', xc='PBE', txt='pbe.txt')
    >>> atoms.calc = target_calc
    >>> atoms.get_potential_energy()
    >>>
    >>> # 2. Set up iKS calculator with LDA-based NumericalXC
    >>> nxc = NumericalXC.from_other_xc('LDA', target_calc.density, setup_name='PBE')
    >>> iks_calc = GPAW(mode='pw', xc=nxc, txt='iks.txt')
    >>>
    >>> # 3. Initialize and run the iKS procedure
    >>> iks = iKS(target=target_calc.density, calc=iks_calc, atoms=atoms)
    >>> iks.run()
    """

    def __init__(
        self,
        target: GPAW | Density,
        calc: GPAW,
        P_snn: Optional[ArrayND] = None,
        weights: str | ArrayND = "uniform",
        atomic_weights: str | MutableMapping = "radial_uniform",
        atoms: Atoms | None = None,
        lambdareg: float = 0.0,
        lambdacore: float = 0.0,
        reg_scheduler_params: dict[str, int | float] | None = None,
        corereg_scheduler_params: dict[str, int | float] | None = None,
        correg_xc: str = None,
        stencil: int = 3,
        nfixed: int = 2,
        linear_solver: dict | str = "minres",
        use_coarse_grid: bool = False,
        apply_occupation_correction: bool = True,
        kin_en_using_band: bool = True,
        log_l1_ae: bool = True,
        mix_paw_error: bool = True,
        degeneracy_tol: float = 1e-3,
        occ_tol: float = 1e-10,
        nxc_kwargs: dict[str, Any] | None = None,
        weights_sum_spins: bool = True,
    ):
        self.target = target
        atoms = calc.atoms or atoms
        if atoms is None:
            assert isinstance(target, GPAW)
            atoms = target.atoms
        self.P_snn = P_snn
        self.lambdareg = lambdareg
        self.lambdacore = lambdacore
        reg_scheduler_params = reg_scheduler_params or {}
        corereg_scheduler_params = corereg_scheduler_params or {}
        self.correg_xc = correg_xc
        self.regsched = iks_optimize.StaircaseScheduler(
            lambdareg, **reg_scheduler_params)
        self.coresched = iks_optimize.StaircaseScheduler(
            lambdacore, **corereg_scheduler_params)
        self.niter: int = 0
        self.weights_sum_spins = weights_sum_spins

        self.initialize_from_calc(target, calc, atoms, weights, atomic_weights)
        self.stencil = stencil
        self.nfixed = nfixed
        self.apply_occupation_correction = apply_occupation_correction
        self.kin_en_using_band = kin_en_using_band
        if isinstance(linear_solver, dict):
            self.linear_solver = linear_solver.pop("name")
            self.ls_kwargs = linear_solver
            self.ls_kwargs.setdefault("maxiter", 600)
            self.ls_kwargs.setdefault("rtol", 1e-6)
        else:
            self.linear_solver = linear_solver
            self.ls_kwargs = {"maxiter": 600, "rtol": 1e-6}
        self.use_coarse_grid: bool = use_coarse_grid
        self.mix_paw_error = mix_paw_error
        self.L_iter: list[float] = []
        self.LD_iter: list[float] = []
        self.opt: Optional[Any] = None
        self.initialized = False
        self.log_l1_ae = log_l1_ae
        self.nxc_kwargs = nxc_kwargs or {}
        self.degeneracy_tol = degeneracy_tol
        self.occ_tol = occ_tol

    def initialize_from_calc(self, target, calc, atoms, weights, atomic_weights):
        self.calc = calc
        if not calc.initialized:
            calc.initialize(atoms)
        if not (calc.wfs.positions_set and calc.hamiltonian.positions_set):
            calc.set_positions(atoms)
        self.fmt = iks_tools.LogFormatter(calc.log)
        self.wfs = calc.wfs
        self.density = calc.density
        self.hamiltonian = calc.hamiltonian
        self.atompart = calc.hamiltonian.atom_partition
        self.setups = calc.hamiltonian.setups
        self.init_target_density(self.P_snn)
        self.nspins: int = self.tdens.nspins
        self.weights_sG, self.weights = iks_tools.get_weights(
            weights, self.tdens.nt_sG, self.density.gd, self.weights_sum_spins)
        self.weights_asng, self.atomic_weights = iks_tools.get_atomic_weights(
            atomic_weights, calc.setups, self.tdens.D_asp, self.weights_sum_spins
        )

    def init_target_density(self, P_snn: Optional[ArrayND] = None) -> None:
        target = self.target
        wfs = None
        if isinstance(target, Density):
            tdens = target
        elif hasattr(target, 'density'):
            tdens = target.density
            wfs = target.wfs
        else:
            tdens = target  # FakeDensity
        if wfs is None:
            assert P_snn is None, "Provide the target calculator (for wfs) to use P_snn"
        if not isinstance(target, Density):
            if isinstance(tdens, FakeDensity):
                from gpaw.iks.tools import fakdensity2old
                tdens = fakdensity2old(
                    tdens, self.calc.density, self.calc.hamiltonian)

            tdens.set_positions(self.hamiltonian.spos_ac,
                                self.density.atom_partition)  # for nct_G
            iks_tools.update_density_from_density_matrix(tdens, wfs, P_snn)

        density = self.density
        tdens.gd = density.gd
        tdens.finegd = density.finegd
        tdens.redistributor = density.redistributor
        tdens.set_positions(self.hamiltonian.spos_ac, density.atom_partition)
        tdens.calculate_normalized_charges_and_mix()
        # tdens.fixed = True
        self.tdens = tdens

    def initialize(self) -> None:
        self.eigensolver_name: str = getattr(self.wfs.eigensolver, "name", "")
        self.gd: GridDescriptor = self.hamiltonian.gd
        self.finegd: GridDescriptor = self.hamiltonian.finegd
        self.coarse2fine = self.density.interpolate

        def fine2coarse(a_g):
            a_G = self.hamiltonian.restrict_and_collect(a_g)
            return self.gd.distribute(a_G)

        self.fine2coarse = fine2coarse
        self.init_xc()
        self.ls_u: list[iKSLinearSolver] = []
        self.dn_sG = self.gd.empty(self.density.nspins)
        self.dD_asp = self.atompart.arraydict(
            self.density.D_asp.shapes_a, self.density.D_asp.dtype
        )
        self.initialized = True

    def init_xc(self) -> None:
        xc = self.hamiltonian.xc
        if not isinstance(xc, NumericalXC):
            self.nxc_kwargs.setdefault("paw_potmode", "lebedev")
            self.nxc_kwargs.setdefault("setup_name", xc.get_setup_name())
            self.nxc_kwargs.setdefault("stencil", self.stencil)
            xc = FermiAmaldi(
                self.tdens, **self.nxc_kwargs)
            self.calc.parameters["xc"] = xc
        self.J_xasz: list[ArrayDict] = [
            self.atompart.arraydict(v_asz.shapes_a, dtype=float)
            for v_asz in xc.vxc_xasz
        ]
        xc.initialize(self.density, self.hamiltonian, self.wfs)
        xc.set_positions(self.hamiltonian.spos_ac)
        xc.set_grid_descriptor(self.finegd)
        self.hamiltonian.xc = xc
        self.xc: NumericalXC = xc
        self.nxc_kwargs = xc.todict()
        # regularization stuff
        self.wreg_ag = None
        self.vc_axg = None
        if self.lambdacore != 0.0:
            self.wreg_ag = iks_tools.calculate_regularization_weights(
                self.setups)
            self.vc_axg = iks_tools.calc_core_atomic_potentials(
                self.setups, self.correg_xc or self.xc.setup_name)

    def irun_scf(self) -> None:
        """perform scf with the target density and update the inverted density from wfs"""
        scf = self.calc.scf
        scf.reset()
        scf.niter = 1
        self.density.mixer.reset()
        self.density.error = None
        self.density.calculate_normalized_charges_and_mix()
        self.hamiltonian.update(self.tdens, self.wfs,
                                kin_en_using_band=self.kin_en_using_band)
        while scf.niter <= scf.maxiter:
            if self.eigensolver_name == "etdm-fdpw":
                raise NotImplementedError  # TODO
            else:
                self.wfs.eigensolver.iterate(self.hamiltonian, self.wfs)
                e_entropy = self.wfs.calculate_occupation_numbers(
                    self.calc.scf.fix_fermi_level
                )
            self.hamiltonian.get_energy(
                e_entropy, self.wfs, kin_en_using_band=self.kin_en_using_band, e_sic=0.0
            )
            ctx = scf.check_convergence(
                self.density, self.hamiltonian, self.wfs, self.calc.log, lambda i: None
            )
            yield ctx

            if scf.converged and (scf.niter >= self.nfixed):
                break

            self.density.update(self.wfs)
            # update hamiltonian using the target density
            self.hamiltonian.update(
                self.tdens, self.wfs, kin_en_using_band=self.kin_en_using_band)
            scf.niter += 1
            self.fmt.flush()
            if self.nfixed == 0:
                break

    def run_scf(self):
        for _ in self.irun_scf():
            pass

    def iterate(self) -> None:
        if self.wfs.kptband_comm.size > 1:
            # we need to find the bug that in the k-point parallelization causes
            # the values of potentials (and consequently the densities or vice versa) diverge on each core
            # this is a hack for now which along the syncing in the optimizers and lagrangian fixes the issue.
            self.wfs.kptband_comm.broadcast(self.xc.vxc_sg, 0)
            for v_asz in self.xc.vxc_xasz:
                partition = v_asz.partition
                v_asz.redistribute(partition.as_serial())
                v_asz.redistribute(partition)
        self.xc.set_grid_descriptor(self.hamiltonian.finegd)
        self.finegd = self.hamiltonian.finegd
        self.gd = self.hamiltonian.gd
        self.run_scf()
        wfs = self.wfs
        wfs.orthonormalize()
        self.density.update(wfs)
        self.update_diff()
        self.dd_u = []
        for kpt in wfs.kpt_u:
            dd = DegeneracyDescriptor(
                kpt, tolerance=self.degeneracy_tol, occ_tolerance=self.occ_tol)
            dd.update()
            # print('dd.blocks_occ', dd.kpt.q, dd.blocks_occ)
            self.dd_u.append(dd)
            vdiff_nn = self.calculate_vdiff_matrix_elements(kpt)
            dd.vdiff_nn = vdiff_nn

        # TODO: implement for multiple fermi levels (fixedmagmom=True)
        if self.apply_occupation_correction:
            eta = self.calculate_eta()
        else:
            eta = 0.0
            self.dfde_un = np.zeros(
                (len(wfs.kpt_u), self.wfs.bd.nbands), dtype=float)
        self.ls_u = []
        for kpt, dd, dfde_n in zip(wfs.kpt_u, self.dd_u, self.dfde_un):
            p0_nG = self.calculate_p_expansion(dd, dd.vdiff_nn, eta, dfde_n)
            ls = iKSLinearSolver(
                self, dd=dd, method=self.linear_solver, **self.ls_kwargs)
            ls.run(p0_nG)
            self.ls_u.append(ls)
            if wfs.occupations.name != 'zero-width' or self.apply_occupation_correction:
                ls.apply_occupation_correction(dfde_n, eta)
        self.logminres()

    def logminres(self):
        from gpaw.mpi import broadcast_string
        kd = self.wfs.kd
        ls_u = self.ls_u
        self.fmt.log(' k-index    spin    minres-iters    degenaracy')
        self.fmt.log('................................................')
        for k in range(kd.nibzkpts):
            for s in range(kd.nspins):
                rank, u = kd.get_rank_and_index(k, s)
                if kd.comm.rank == rank:
                    ls = ls_u[u]
                    msg = (f'  {k}          {s}          '
                           f'{ls.iters}      {ls.blocks_msg}')
                else:
                    msg = None
                msg = broadcast_string(msg, root=rank, comm=kd.comm)
                self.fmt.log(msg)

    def calculate_p_expansion(self, dd, vdiff_nn, eta, dfde_n) -> Array2D:
        nbands = self.wfs.bd.nbands
        nocc = len(dd.occ)
        if nocc == 0:
            return np.zeros(1, dtype=float)
        kpt = dd.kpt
        p_mG = self.wfs.empty(nocc, q=kpt.q)
        for bm in dd.blocks_occ:
            for m in bm:
                p_mG[dd.idx_map[m]][:] = 0.0
            for n in range(nbands):
                if n in bm:
                    for m in bm:
                        coeff = vdiff_nn[n, m]
                        if m == n:
                            coeff += eta
                        p_mG[dd.idx_map[m]] += 0.5 * \
                            coeff * dfde_n[m] * kpt.psit_nG[n]
                else:
                    for m in bm:
                        f_m = kpt.f_n[m]
                        eps_m = kpt.eps_n[m]
                        eps_n = kpt.eps_n[n]
                        p_mG[dd.idx_map[m]] += f_m / (eps_m - eps_n) * \
                            vdiff_nn[n, m] * kpt.psit_nG[n]
        return p_mG

    def calculate_vdiff_matrix_elements(self, kpt):
        psit_nR = self.get_realspace_psit(kpt)
        vdiffpsit_nG = 2 * \
            (self.weights_sG[kpt.s] * self.dn_sG[kpt.s])[np.newaxis] * psit_nR
        if self.wfs.mode == "pw":
            b_nR = vdiffpsit_nG.copy()
            vdiffpsit_nG = self.wfs.empty(len(psit_nR), q=kpt.q)
            for b_G, b_R in zip(vdiffpsit_nG, b_nR):
                b_G[:] = self.wfs.pd.fft(b_R, q=kpt.q)
        vdiff_nn = self.wfs.integrate(
            kpt.psit_nG, vdiffpsit_nG, global_integral=True)

        vdiffpaw_nn = np.zeros(vdiff_nn.shape, dtype=vdiff_nn.dtype)
        # apply PAW corrections
        for a, P_ni in kpt.P_ani.items():
            w_ng = self.weights_asng[a][kpt.s]
            Phi_ii = iks_tools.calculate_diff_overlap_matrix(
                self.setups[a],
                w_ng,
                self.atomic_weights,
                self.dD_asp[a][kpt.s],
                self.mix_paw_error,
            )
            vdiffpaw_nn += 2 * \
                np.einsum("ni,ij,mj->nm", P_ni.conj(), Phi_ii, P_ni).real
        self.gd.comm.sum(vdiffpaw_nn)
        vdiff_nn += vdiffpaw_nn
        return vdiff_nn

    def calculate_eta(self):
        self.dfde_un = []
        enum = 0.0
        denom = 0.0
        if self.wfs.collinear and self.nspins == 1:
            degeneracy = 2
        else:
            degeneracy = 1
        for kpt, dd in zip(self.wfs.kpt_u, self.dd_u):
            f_n, dfde_n, _ = self.wfs.occupations.distribution(
                kpt.eps_n * Ha, self.wfs.fermi_levels * Ha)
            # 1/eV -> 1/Ha
            dfde_n = dfde_n * (Ha * kpt.weightk * degeneracy)
            self.dfde_un.append(dfde_n)
            for n in dd.occ:
                enum += dd.vdiff_nn[n, n].real * dfde_n[n]
                denom += dfde_n[n]
        enum = self.wfs.kptband_comm.sum_scalar(enum)
        denom = self.wfs.kptband_comm.sum_scalar(denom)
        if abs(denom) < 1e-20:
            denom = np.inf
        eta = enum / denom
        self.fmt.log(f'occuupation correction eta: {eta:.4e}')
        return eta

    def get_realspace_psit(self, kpt):
        if self.wfs.mode == "lcao":
            self.wfs.initialize_wave_functions_from_lcao()
        # no band parallelization in iKS
        psit_nG = self.wfs.bd.collect(kpt.psit_nG, broadcast=True)
        if self.wfs.mode == "pw":
            psit_nG = np.asarray(
                [self.wfs.pd.ifft(psit_G, q=kpt.q) for psit_G in psit_nG]
            )
        return psit_nG

    def lagrangian(self) -> tuple[float, float, float, float]:
        L1 = self.gd.integrate(
            self.weights_sG * np.abs(self.dn_sG), global_integral=True
        ).sum()
        L2 = self.gd.integrate(
            self.weights_sG * (self.dn_sG**2), global_integral=True
        ).sum()
        lambdareg = self.lambdareg
        if lambdareg != 0.0:
            grad_svg = self.xc.kernel.calculate_gradient(self.xc.vxc_sg)
            sigma_sg = np.einsum('sv...,sv...->s...', grad_svg, grad_svg)
            II = lambdareg * \
                self.finegd.integrate(sigma_sg, global_integral=True).sum()
            # print('regularization lag', I, self.calc.world.rank)
            self.fmt.log(f'regularization correction: {II:.4e}')
            L1 += II
            L2 += II

        LD1 = np.array([0.0])
        LD2 = np.array([0.0])
        Lreg = 0.0
        vxc_asz = None
        lambdacore = self.lambdacore
        if lambdacore != 0.0:
            assert self.xc.paw_potmode in [
                PAWPotMode.lebedev, PAWPotMode.angular]
            vxc_asz = self.hamiltonian.atomdist.from_work(self.xc.vxc_xasz[0])
            vxct_asz = self.hamiltonian.atomdist.from_work(self.xc.vxc_xasz[1])
        for a, dD_sp in self.dD_asp.items():
            xcc = self.setups[a].xc_correction
            if xcc is None:
                continue
            dD_sLq = np.inner(dD_sp, xcc.B_pqL.T)
            if self.mix_paw_error:
                dn_sLg = np.dot(dD_sLq, xcc.n_qg - xcc.nt_qg)
            else:
                dn_sLg = np.dot(dD_sLq, xcc.n_qg)
                dnt_sLg = np.dot(dD_sLq, xcc.nt_qg)
            Lmax = dn_sLg.shape[1]
            for s in range(self.nspins):
                w_ng = self.weights_asng[a][s]
                if "radial_" in self.atomic_weights:
                    w_ng = w_ng.repeat(len(Y_nL), axis=0)
                for n, Y_L in enumerate(Y_nL[:, :Lmax]):
                    dn_g = np.dot(Y_L, dn_sLg[s])
                    LD1 += weight_n[n] * \
                        xcc.rgd.integrate(w_ng[n] * np.abs(dn_g))
                    LD2 += weight_n[n] * \
                        xcc.rgd.integrate(w_ng[n] * (dn_g**2))
                    if not self.mix_paw_error:
                        dnt_g = np.dot(Y_L, dnt_sLg[s])
                        LD1 += weight_n[n] * \
                            xcc.rgd.integrate(w_ng[n] * np.abs(dnt_g))
                        LD2 += weight_n[n] * \
                            xcc.rgd.integrate(w_ng[n] * (dnt_g**2))
                    if lambdacore != 0.0:
                        if self.xc.paw_potmode is PAWPotMode.lebedev:
                            dv_g = vxc_asz[a][s][n] - self.vc_axg[a][0]
                            Lreg += lambdacore * \
                                weight_n[n] * \
                                xcc.rgd.integrate(self.wreg_ag[a] * dv_g**2)
                            dv_g = vxct_asz[a][s][n] - self.vc_axg[a][1]
                            Lreg += lambdacore * \
                                weight_n[n] * \
                                xcc.rgd.integrate(self.wreg_ag[a] * dv_g**2)

                if lambdacore != 0.0 and self.xc.paw_potmode is PAWPotMode.angular:
                    dv_g = vxc_asz[a][s][0] - self.vc_axg[a][0]
                    Lreg += lambdacore * \
                        xcc.rgd.integrate(self.wreg_ag[a] * dv_g**2)
                    dv_g = vxct_asz[a][s][0] - self.vc_axg[a][1]
                    Lreg += lambdacore * \
                        xcc.rgd.integrate(self.wreg_ag[a] * dv_g**2)

        LD1 += Lreg
        LD2 += Lreg
        comm = self.dD_asp.partition.comm
        comm.sum(LD1)
        comm.sum(LD2)
        if lambdacore != 0.0:
            Lreg = comm.sum_scalar(Lreg)
            self.fmt.log(f'Atomic regularization: {Lreg:.4e}')

        # return L1, L2, LD1.item(), LD2.item()
        # sync kpoints :()
        L1_val, L2_val = L1, L2
        LD1_val, LD2_val = LD1.item(), LD2.item()
        if self.wfs.kptband_comm.size > 1:
            lags = np.array([L1_val, L2_val, LD1_val, LD2_val], dtype=float)
            self.wfs.kptband_comm.broadcast(lags, 0)
            L1_val, L2_val, LD1_val, LD2_val = lags[0], lags[1], lags[2], lags[3]

        return L1_val, L2_val, LD1_val, LD2_val

    def jacobian(self) -> ArrayND:
        J_sG = self.gd.zeros(self.density.nspins)
        for u, kpt in enumerate(self.wfs.kpt_u):
            J_sG[kpt.s] += self.ls_u[u].calculate_jacobian_one_k()
        self.wfs.kptband_comm.sum(J_sG)
        for J_G in J_sG:
            self.wfs.kd.symmetry.symmetrize(J_G, self.gd)
        if self.use_coarse_grid:
            J_sg = J_sG
        else:
            J_sg = self.finegd.zeros(self.density.nspins)
            self.coarse2fine(J_sG, J_sg)
        lambdareg = self.lambdareg
        if lambdareg != 0.0:
            lapl_sg = self.xc.kernel.calculate_lapl(self.xc.vxc_sg)
            if self.use_coarse_grid:
                J_sg -= 2 * lambdareg * self.fine2coarse(lapl_sg)
            else:
                J_sg -= 2 * lambdareg * lapl_sg
        self.J_sg = J_sg
        return J_sg

    def atomic_jacobian(self) -> list[ArrayDict]:
        if self.xc.paw_potmode is PAWPotMode.none:
            return []

        shapes_a = {
            a: (self.nspins, s.ni, s.ni)
            for a, s in enumerate(self.setups)
        }
        dp_asii = self.atompart.arraydict(shapes_a, self.wfs.dtype)
        for a, dp_sii in dp_asii.items():
            dp_sii[:] = 0.0
            for kpt, ls in zip(self.wfs.kpt_u, self.ls_u):
                dp_ii = np.einsum(
                    "ni,nj->ij", ls.Pp_ani[a].conj(), kpt.P_ani[a][ls.dd.occ])
                dp_ii += dp_ii.conj().T
                dp_sii[kpt.s] += dp_ii
            self.wfs.kptband_comm.sum(dp_sii)

        if len(self.wfs.kd.symmetry.op_scc) != 0:
            dp_asii.redistribute(self.atompart.as_serial())
            self.symmetrize_atomic_adjoint_matrices(
                dp_asii, self.wfs.kd.symmetry.a_sa)
            dp_asii.redistribute(self.atompart)

        dp_asp = self.atompart.arraydict(
            self.density.D_asp.shapes_a, self.wfs.dtype)
        for a, dp_sii in dp_asii.items():
            for s in range(self.nspins):
                dp_asp[a][s] = pack_hermitian(dp_sii[s])
        self.update_atomic_jacobian_from_adjoint_matrices(dp_asp)
        for J_asz in self.J_xasz:
            J_asz.check_consistency()
        return self.J_xasz

    def symmetrize_atomic_adjoint_matrices(self, dp_asii, a_sa):
        if not dp_asii:
            return
        time_reversal = self.wfs.kd.symmetry.time_reversal

        def symmetrize(rot, a, dp_aii, map_sa):
            ni = rot.R_sii.shape[1]
            D_ii = np.zeros((ni, ni), dtype=dp_aii[0].dtype)
            for s, R_ii in enumerate(rot.R_sii):
                D_mapped = dp_aii[map_sa[s][a]]

                if time_reversal:
                    D_mapped = D_mapped.conj()

                D_ii += R_ii @ D_mapped @ R_ii.T
            return D_ii / len(map_sa)

        atomrot = self.setups.atomrotations
        nspins = next(iter(dp_asii.values())).shape[0]
        for s in range(nspins):
            dp_aii = [dp_asii[a][s] for a in range(len(dp_asii))]
            for a, dp_ii in enumerate(dp_aii):
                dp_asii[a][s] = symmetrize(
                    atomrot.get_by_a(a), a, dp_aii, a_sa)

    def update_atomic_jacobian_from_adjoint_matrices(self, dp_asp):
        vxc_asz = None
        vxct_asz = None
        lambdacore = self.lambdacore
        if lambdacore != 0.0:
            vxc_asz = self.hamiltonian.atomdist.from_work(self.xc.vxc_xasz[0])
            vxct_asz = self.hamiltonian.atomdist.from_work(self.xc.vxc_xasz[1])
        for a, dp_sp in dp_asp.items():
            if self.xc.paw_potmode is PAWPotMode.direct:
                self.J_xasz[0][a][:] = dp_asp[a]
            else:
                setup = self.setups[a]
                if setup.xc_correction is None:
                    continue
                xcc = setup.xc_correction
                dp_sLq = np.inner(dp_sp, xcc.B_pqL.T)
                dv_sLg = np.dot(dp_sLq, xcc.n_qg)
                dvt_sLg = np.dot(dp_sLq, xcc.nt_qg)
                if self.xc.paw_potmode is PAWPotMode.lebedev:
                    Lmax = dvt_sLg.shape[1]
                    for s in range(self.nspins):
                        self.J_xasz[0][a][s] = np.dot(
                            Y_nL[:, :Lmax], dv_sLg[s]).real
                        self.J_xasz[1][a][s] = np.dot(
                            Y_nL[:, :Lmax], dvt_sLg[s]).real
                        if lambdacore != 0.0:
                            dv = vxc_asz[a][s, :] - \
                                self.vc_axg[a][0][np.newaxis]
                            self.J_xasz[0][a][s, :] += 2 * \
                                lambdacore * \
                                self.wreg_ag[a][np.newaxis] * dv
                            dv = vxct_asz[a][s, :] - \
                                self.vc_axg[a][1][np.newaxis]
                            self.J_xasz[1][a][s, :] += 2 * \
                                lambdacore * \
                                self.wreg_ag[a][np.newaxis] * dv
                else:
                    assert self.xc.paw_potmode is PAWPotMode.angular
                    self.J_xasz[0][a][:] = dv_sLg.real
                    self.J_xasz[1][a][:] = dvt_sLg.real
                    if lambdacore != 0.0:
                        dv = vxc_asz[a][s, :] - \
                            self.vc_axg[a][0][np.newaxis]
                        self.J_xasz[0][a][s, :] += 2 * \
                            lambdacore * \
                            self.wreg_ag[a][np.newaxis] * dv
                        dv = vxct_asz[a][s, :] - \
                            self.vc_axg[a][1][np.newaxis]
                        self.J_xasz[1][a][s, :] += 2 * \
                            lambdacore * \
                            self.wreg_ag[a][np.newaxis] * dv

    def converged(self, L, LD, J_sg, Ltol, LDtol, Jtol) -> bool:
        Jmax = self.finegd.comm.max_scalar(abs(J_sg).max())
        return (L < Ltol) and (Jmax < Jtol) and (LD < LDtol)

    def update_diff(self) -> None:
        self.dn_sG[...] = self.tdens.nt_sG - self.density.nt_sG
        D_asp = self.density.D_asp
        for a, D_sp in D_asp.items():
            self.dD_asp[a][...] = self.tdens.D_asp[a] - D_sp

    def warmup(self) -> None:
        """
        Initialize the system by running an SCF cycle on the initial guess
        and updating the hamiltonian with the target density.
        """
        if not self.initialized:
            self.initialize()
        self.fmt.section_title(
            "OECO-iKS: Running single scf calculation with "
            "the intial guess with target density updates."
        )
        self.update_diff()
        L1, L2, LD1, LD2 = self.lagrangian()
        self.run_scf()
        calc = self.calc
        scf = calc.scf
        if scf.converged:
            self.fmt.log("Single-point calculation converged after "
                         f"{scf.niter} iterations")
        else:
            self.fmt.log(
                "warning: Single-point calculation did not converge after "
                f"{scf.maxiter} iterations"
            )
        calc.summary()
        calc.call_observers(self.niter, final=True)
        # save results
        e_free = self.hamiltonian.e_total_free
        e_extrapolated = self.hamiltonian.e_total_extrapolated
        calc.results["energy"] = e_extrapolated * Ha
        calc.results["free_energy"] = e_free * Ha
        self.update_diff()
        L1, L2, LD1, LD2 = self.lagrangian()
        L1_uw, L2_uw, LD1_uw, LD2_uw, LAE1_uw, LAE2_uw = self.uwnorm(ae=True)
        self.fmt.log(
            "Warmup (initial) L1 errors -> "
            f"ps={L1:<.4e}, atomic={LD1:<.4e}\n"
            f"\t\t raw (unweighted): ae={LAE1_uw}, "
            f"ps={L1_uw:<.4e}, atomic={LD1_uw:<.4e}"
        )
        self.fmt.log(
            "Warmup (initial) L2 errors -> total: "
            f"ps={L2:<.4e}, atomic={LD2:<.4e}\n"
            f"\t\t raw (unweighted): ae={LAE2_uw}, "
            f"ps={L2_uw:<.4e}, atomic={LD2_uw:<.4e}"
        )

    def uwnorm(self, ae=False, per_electron=False) -> tuple[float, ...]:
        gd = self.gd
        Nt = 1
        if per_electron:
            Nt = gd.integrate(self.tdens.nt_sG.sum(0))
        l1_uw = gd.integrate(
            abs(self.tdens.nt_sG - self.density.nt_sG).sum(0)) / Nt
        l2_uw = gd.integrate(
            ((self.tdens.nt_sG - self.density.nt_sG) ** 2).sum(0)) / Nt
        l1_atomic = 0.0
        l2_atomic = 0.0
        for a, dD_sp in self.dD_asp.items():
            xcc = self.setups[a].xc_correction
            if xcc is None:
                continue
            dD_sLq = np.inner(dD_sp, xcc.B_pqL.T)
            if self.mix_paw_error:
                dn_sLg = np.dot(dD_sLq, xcc.n_qg - xcc.nt_qg)
            else:
                dn_sLg = np.dot(dD_sLq, xcc.n_qg)
                dnt_sLg = np.dot(dD_sLq, xcc.nt_qg)
            Lmax = dn_sLg.shape[1]
            Nc = 1
            if per_electron:
                Nc = self.setups[a].Nc or 1
            for s in range(self.nspins):
                for n, Y_L in enumerate(Y_nL[:, :Lmax]):
                    dn_g = np.dot(Y_L, dn_sLg[s])
                    l1_atomic += weight_n[n] * \
                        xcc.rgd.integrate(np.abs(dn_g)) / Nc
                    l2_atomic += weight_n[n] * xcc.rgd.integrate(dn_g**2) / Nc
                    if not self.mix_paw_error:
                        dnt_g = np.dot(Y_L, dnt_sLg[s])
                        l1_atomic += weight_n[n] * \
                            xcc.rgd.integrate(np.abs(dnt_g)) / Nc
                        l2_atomic += weight_n[n] * \
                            xcc.rgd.integrate(dnt_g**2) / Nc

        comm = self.dD_asp.partition.comm
        l1_atomic = comm.sum_scalar(l1_atomic)
        l2_atomic = comm.sum_scalar(l2_atomic)
        l1_ae = np.nan
        l2_ae = np.nan
        if ae and self.log_l1_ae:
            spos_ac = self.hamiltonian.spos_ac
            naet_sg, _ = self.tdens.get_all_electron_density(
                spos_ac=spos_ac, gridrefinement=2
            )
            nae_sg, _ = self.density.get_all_electron_density(
                spos_ac=spos_ac, gridrefinement=2
            )
            N = 1
            if per_electron:
                N = gd.integrate(naet_sg.sum(0))
            l1_ae = gd.integrate(abs(naet_sg - nae_sg).sum(0)) / N
            l2_ae = gd.integrate(((naet_sg - nae_sg) ** 2).sum(0)) / N
        if ae:
            return l1_uw, l2_uw, l1_atomic, l2_atomic, l1_ae, l2_ae
        else:
            return l1_uw, l2_uw, l1_atomic, l2_atomic

    def get_atomic_jac_maximum(self):
        Jmax_x = []
        for J_asz in self.J_xasz:
            Jmax = -np.inf
            for J_sz in J_asz.values():
                if J_sz.size == 0:
                    Jmax = max(Jmax, 0.0)
                else:
                    Jmax = max(Jmax, abs(J_sz).max())
            Jmax = self.density.gd.comm.max_scalar(Jmax)
            Jmax_x.append(Jmax)
        return Jmax_x

    def todict(self):
        linear_solver = {"name": self.linear_solver, **self.ls_kwargs}
        return dict(
            weights=self.weights,
            atomic_weights=self.atomic_weights,
            apply_occupation_correction=self.apply_occupation_correction,
            reg_params=self.regsched.todict(),
            core_reg_params=self.coresched.todict(),
            stencil=self.stencil,
            nfixed=self.nfixed,
            linear_solver=linear_solver,
            use_coarse_grid=self.use_coarse_grid,
            kin_en_using_band=self.kin_en_using_band,
            nxc_kwargs=self.nxc_kwargs,
            mix_paw_error=self.mix_paw_error,
        )

    def get_xcpotential_space_size(self):
        if self.use_coarse_grid:
            size = np.prod(self.gd.get_size_of_global_array()) * self.nspins
        else:
            size = np.prod(
                self.finegd.get_size_of_global_array()) * self.nspins
        for v_asz in self.xc.vxc_xasz:
            size += sum([np.prod(shape) for shape in v_asz.shapes_a])
        return size

    def preopt(self, title, opt_params):
        self.niter = 0
        self.fmt.boxed_title(title)
        self.fmt.key_value("number of (xc) parameters",
                           f"{self.get_xcpotential_space_size():,}")
        self.fmt.print_dict(opt_params)
        self.fmt.print_dict(self.todict())
        self.fmt.section_title("Optimization cycle")
        columns = [(4, "\t", "Iter"), (11, "L1 error", "ps"), (11, "\t", "atomic"),
                   (11, "L2 error", "ps"), (11, "\t", "atomic"), (11, "max-grad", "ps")]
        if self.xc.paw_potmode is not PAWPotMode.none:
            columns.append((11, "", "atomic-true"))
            if self.xc.paw_potmode is not PAWPotMode.direct:
                columns.append((11, "", "atomic-ps"))

        line1 = "".join([f"{txt:^{w}}" for w, txt, _ in columns])
        line2 = "".join(
            [f"{txt:^{w}} " for w, _, txt in columns]
        )  # right-align with a space
        self.fmt.log(line1 + "\n" + line2)
        self.fmt.separator("-")
        self.fmt.flush()

    def afteropt(self, iterate=True) -> None:
        if iterate:
            self.iterate()
        self.calc.summary()
        self.calc.call_observers(self.niter, final=True)
        L1_uw, L2_uw, LD1_uw, LD2_uw, LAE1_uw, LAE2_uw = self.uwnorm(ae=True)
        self.fmt.log("Final L1 raw (unweighted) errors -> "
                     f"all-electron: {LAE1_uw:<.4e}, ps={L1_uw:.4e}, atomic={LD1_uw:.4e}")
        self.fmt.log("Final L2 raw (unweighted) errors -> "
                     f"all-electron: {LAE2_uw:<.4e}, ps={L2_uw:.4e}, atomic={LD2_uw:.4e}")
        # save results
        e_free = self.hamiltonian.e_total_free
        e_extrapolated = self.hamiltonian.e_total_extrapolated
        self.calc.results["energy"] = e_extrapolated * Ha
        self.calc.results["free_energy"] = e_free * Ha

    def logiter(
        self,
        t: int,
        L1: float,
        L2: float,
        LD1: float,
        LD2: float,
        J_sg: ArrayND,
        sep="-",
    ) -> None:
        Jmax = self.finegd.comm.max_scalar(abs(J_sg).max())
        Jmax_x = self.get_atomic_jac_maximum()
        L1_uw, L2_uw, LD1_uw, LD2_uw = self.uwnorm()
        columns1 = [(12, L1), (12, LD1), (12, L2), (12, LD2), (12, Jmax)]
        for Jmaxa in Jmax_x:
            columns1.append((12, Jmaxa))
        columns2 = [(12, L1_uw), (12, LD1_uw), (12, L2_uw), (12, LD2_uw)]
        line1 = f"{t:<5d}" + \
            "".join([f"{number:>{w}.5e}" for w, number in columns1])
        line2 = "raw  " + \
            "".join([f"{number:>{w}.5e}" for w, number in columns2])
        self.fmt.separator(sep)
        self.fmt.log(line1)
        self.fmt.log(line2)
        self.fmt.separator(sep)
        self.fmt.flush()

    def run_adam(self, maxiter: int = 300, eta_pseudo: float = 0.001, eta_paw: float = 0.001,
                 Ltol: float = 1e-6, LDtol: float = 1e-4, Jtol: float = 1e-9,
                 beta1: float = 0.9, beta2: float = 0.999,) -> list[float]:
        if not self.initialized:
            self.initialize()
        L_iter = self.L_iter = []
        opt_params = {
            "maxiter": maxiter,
            "eta-pseudo": eta_pseudo,
            "eta-paw": eta_paw,
            "beta1": beta1,
            "beta2": beta2,
            "smooth Lagrangian tolerance": Ltol,
            "atomic Lagrangian tolerance": LDtol,
            "gradients-tolerance": Jtol,
        }
        self.preopt("Adam optimizer for OECO-iKS procedure", opt_params)
        # optimization loop
        L_iter = iks_optimize.iks_adam_opt(
            self, maxiter, eta_pseudo, eta_paw, Ltol, LDtol, Jtol, beta1, beta2
        )
        self.afteropt()
        return L_iter

    def run_fastest_descent(
        self,
        maxiter: int = 300,
        alpha_pseudo: float | callable[[int], float] = 1.0,
        alpha_paw: float | callable[[int], float] = 1.0,
        Ltol: float = 1e-6,
        LDtol: float = 1e-4,
        Jtol: float = 1e-9,
    ) -> list[float]:
        """
        Perform gradient descent (steepest descent) to minimize the Lagrangian.

        Parameters
        ----------
        maxiter : int, default=300
            Maximum number of descent iterations.
        alpha_pseudo : float, default=1.0
            Step size for updating the smooth exchange–correlation potential.
        alpha_paw : float, default=1.0
            Step size for updating atomic (PAW) exchange–correlation potentials.
        Ltol : float, default=1e-6
            Convergence threshold for the Lagrangian value. Iteration stops when |L| < Ltol.
        LDtol : float, default=1e-4
            Convergence threshold for atomic density matrices.
        Jtol : float, default=1e-9
            Convergence threshold for the Jacobian of the smooth potential.
            Converged if max|J| < Jtol.

        Returns
        -------
        list[float]
            History of Lagrangian values at each iteration.
        """
        if not self.initialized:
            self.initialize()
        # optimization loop
        self.L_iter = []
        self.LD_iter = []
        opt_kwargs = {
            "maxiter": maxiter,
            "alpha-pseudo": alpha_pseudo,
            "alpha-paw": alpha_paw,
            "smooth Lagrangian tolerance": Ltol,
            "atomic Lagrangian tolerance": LDtol,
            "gradients-tolerance": Jtol,
        }
        self.preopt(
            "fastest-descent optimizer for OECO-iKS procedure", opt_kwargs)

        vxc_sg = self.xc.vxc_sg
        if self.use_coarse_grid:
            vxc_sg = self.gd.zeros(self.nspins)
            self.fine2coarse(self.xc.vxc_sg, vxc_sg)

        alphaps = alpha_pseudo
        alphapaw = alpha_paw

        for t in range(1, maxiter):
            self.niter = t
            # step
            self.iterate()
            vxc_xasz = self.xc.vxc_xasz
            L1, L2, LD1, LD2 = self.lagrangian()
            L = L2
            LD = LD2

            self.L_iter.append(L)
            self.LD_iter.append(LD)
            J_sg = self.jacobian()
            self.atomic_jacobian()
            # update pseudo
            if callable(alpha_pseudo):
                alphaps = alpha_pseudo(t)
            vxc_sg -= alphaps * J_sg
            if self.use_coarse_grid:
                self.coarse2fine(vxc_sg, self.xc.vxc_sg)
            # update paw
            if callable(alpha_paw):
                alphapaw = alpha_paw(t)
            for vxc_asz, J_asz in zip(vxc_xasz, self.J_xasz):
                J_asz = self.hamiltonian.atomdist.to_work(J_asz)
                for a, vxc_sz in vxc_asz.items():
                    vxc_sz[:] -= alphapaw * J_asz[a]
                vxc_asz.check_consistency()

            self.logiter(t, L1, L2, LD1, LD2, J_sg)
            # check convergence
            self.calc.call_observers(self.niter)
            if self.converged(L, LD, J_sg, Ltol, LDtol, Jtol):
                self.fmt.log("OECO-iKS converged!")
                break
        self.afteropt()
        return self.L_iter

    def run_scipy(
        self,
        method: str = "CG",
        options: dict[str, str | float | int] = {},
        scale: float = 1.0,
        **opt_kwargs,
    ) -> list[float]:
        """
        Optimize the Lagrangian using :func:`~scipy.optimize.minimize`.

        Parameters
        ----------
        method : str, default="CG"
            Name of the SciPy optimization algorithm to use.
        options : dict[str, Any], default={"gtol": 1e-9, "maxiter": 313}
            Dictionary of solver-specific options passed directly to ``minimize``.
        opt_kwargs : Any
            Additional keyword arguments passed to the ``minimize`` method.
        scale: float, default=1.0
            Scale for the Lagrangian and gradients passed to the optimizer.

        Returns
        -------
        list[float]
            History of pseudo-Lagrangian values at each objective evaluation.

        Notes
        -----
        To retrieve the individual potentials and results, use:
          - ``iks.xc.vxc_sg``: Smooth exchange–correlation potential, v_xc[n(r)]
          - ``iks.xc.vxc_xasz``: PAW-specific potentials according to the ``paw_potmode``
          - ``iks.opt.result``: The SciPy minimization result (available only on the master core)
        """
        _options: dict[str, Any] = {
            "gtol": 1e-24, "ftol": 1e-24, "maxiter": 313}
        # methods that don't accept gtol
        if method in ["Netwon-CG"]:
            _options.pop("gtol")
        elif method in ["trust-constr"]:
            _options.pop("ftol")
        _options.update(options)
        if not self.initialized:
            self.initialize()
        self.L_iter = []
        self.LD_iter = []
        opts = {"method": method, "options": options, "opt_kwargs": opt_kwargs}
        self.preopt(
            f"SciPy {method} optimizer for OECO-iKS procedure",
            opts,
        )
        opt = iks_optimize.iKSObj(self, scale)
        opt.run(method, _options, **opt_kwargs)
        self.opt = opt
        self.afteropt()
        return self.L_iter

    def run(
        self,
        method: str = "LBFGS",
        maxiter: int = 1000,
        Ltol: float = 1e-6,
        Jtol: float = 1e-9,
        **kwargs,
    ) -> list[float]:
        """
        Run the iKS optimization using the specified algorithm.

        This method provides a unified interface to the available optimization solvers.

        Parameters
        ----------
        method : str, default="LBFGS"
            Optimization algorithm to use. Available options include:
              - "LBFGS": Uses a custom implementation of LBFGS algorithm (recommended)
              - "fastest-descent": Steepest descent optimization.
              - "adam": Adam optimizer.
              - "LBFGS_ASE"
              - Any other string is interpreted as a SciPy minimization method.
        maxiter : int, default=1000
            Maximum number of iterations or function evaluations.
        Ltol : float, default=1e-6
            Tolerance on the Lagrangian value. Ignored for SciPy-based solvers
            (use ``options['ftol']`` instead).
        Jtol : float, default=1e-24
            Tolerance on the gradient (Jacobian) norm. Typically used as the
            primary convergence criterion. Maps to ``gtol`` in SciPy.
        **kwargs : Any
            Additional keyword arguments passed to the selected solver.

        Returns
        -------
        list[float]
            History of Lagrangian values (smooth component) at each iteration.

        Notes
        -----
        This function acts as a unified API wrapper. For detailed control over
        solver-specific parameters, see the documentation for:
        :func:`~run_fastest_descent`, :func:`~run_adam` and :func:`~run_scipy`
        """
        if method == "LBFGS":
            self.opt = iks_optimize.iKSLBFGS(
                self, **kwargs, maxiter=maxiter, Ltol=Ltol, Jtol=Jtol)
            self.opt.run()
        elif method == "fastest-descent":
            return self.run_fastest_descent(
                maxiter=maxiter, Ltol=Ltol, Jtol=Jtol, **kwargs
            )
        elif method == "adam":
            return self.run_adam(maxiter=maxiter, Ltol=Ltol, Jtol=Jtol, **kwargs)
        elif method == "LBFGS_ASE":
            from ase.optimize import LBFGS
            ikso = self.ikso = iks_optimize.iKSOptimizable(self)
            kwargs.setdefault('memory', 40)
            kwargs.setdefault('maxstep', 1.0)
            kwargs.setdefault('use_line_search', True)
            ikso.run(LBFGS, maxiter=maxiter, Jmax=Jtol, **kwargs)
        else:
            options = {"gtol": Jtol, "maxiter": maxiter}
            if "options" in kwargs:
                options.update(kwargs.pop("options"))
            return self.run_scipy(method=method, options=options, **kwargs)
