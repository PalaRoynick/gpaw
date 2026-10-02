"""Gather and scatter homogeneous PAW XC tensor batches."""

from __future__ import annotations
from abc import ABC, abstractmethod

from gpaw.gpu import as_np
from gpaw.sphere.lebedev import weight_n
from gpaw.utilities import pack_hermitian, unpack_hermitian

from typing import Any

class PAWXCBatch(ABC):
    """One shape-compatible collection of PAW XC tensor inputs."""
    def __init__(self, c_a, D_asii, V_aL, setups, functional):
        self.functional = functional

        xp = D_asii.layout.xp
        self.xp = xp
        nspins = D_asii.mydims[0]
        self.spinpol = nspins == 2
        self.atom_indices = c_a
        self.V_aL = xp.asarray([V_aL[c] for c in c_a])
        D_asii_range = {a[0]: (a[1], a[2]) for a in D_asii.layout.myindices}

        # Construct indices
        D_saii = D_asii.data
        setup = setups[c_a[0]]
        self.setup = setup
        D_indices_aii = xp.empty((len(c_a), setup.ni, setup.ni), dtype=int)
        for D_index_ii, c in zip(D_indices_aii, c_a):
            D_index_ii[:] = xp.arange(D_asii_range[c][0],
                                      D_asii_range[c][1]).reshape(
                                          (setup.ni, setup.ni))
        self.D_indices_ap = pack_hermitian(D_indices_aii, xp=self.xp)
        diag_p = self.xp.eye(setup.ni, dtype=bool)[
            self.xp.triu_indices(setup.ni)]
        self.nondiag_p = self.xp.where(~diag_p)[0]

        # Construct arrays
        self.density_sap = D_saii.real[:, self.D_indices_ap]
        self.density_sap[..., self.nondiag_p] *= 2
        self.gradient_sap = xp.zeros_like(self.density_sap)

        xcc = setup.xc_correction

        if xcc is not None:
            rgd = xcc.rgd
            ng = xcc.ng
            self.basis_pqL = xp.asarray(xcc.B_pqL)
            self.ae_radial_qg = xp.asarray(xcc.n_qg)
            self.pseudo_radial_qg = xp.asarray(xcc.nt_qg)
            self.nc_g = xp.asarray(xcc.nc_g)
            self.nct_g = xp.asarray(xcc.nct_g)
            self.nc0_sg = xp.empty_like(self.nc_g, shape=(nspins, ng))
            self.nct0_sg = xp.empty_like(self.nct_g, shape=(nspins, ng))

            self.nc0_sg[:] = xp.sqrt(4 * xp.pi) / nspins * self.nc_g
            self.nct0_sg[:] = xp.sqrt(4 * xp.pi) / nspins * self.nct_g

            if xcc.nc_corehole_g is not None and self.spinpol:
                self.nc0_sg += 0.5 * xp.sqrt(4 * xp.pi) \
                    * xp.asarray(xcc.nc_corehole_g[None, :]) \
                    * xp.array([-1, 1])[:, None]

            self.angular_nL = xp.asarray(xcc.Y_nL)
            self.angular_weight_n = xp.asarray(weight_n)

            self.radial_weight_g = xp.asarray(rgd.dv_g)
            self.reference_energy = xp.asarray(xcc.e_xc0)

        self.K_p = xp.asarray(setup.K_p)
        self.M_p = xp.asarray(setup.M_p)
        self.MB_p = xp.asarray(setup.MB_p)
        self.M_pp = xp.asarray(setup.M_pp)
        self.Delta_pL = xp.asarray(setup.Delta_pL)

    def set_density(self, D_asii, V_aL):
        D_saii = D_asii.data.real
        self.density_sap = D_saii[:, self.D_indices_ap]
        self.density_sap[..., self.nondiag_p] *= 2
        self.gradient_sap[:] = 0
        self.V_aL = self.xp.asarray([V_aL[c] for c in self.atom_indices])

    def evaluate(self):
        if self.setup.xc_correction is None:
            return self.xp.zeros(len(self.atom_indices)), self.gradient_sap

    def unpack(self):
        setup = self.setup
        D_ap = self.density_sap.sum(axis=0)
        dH_ap = (self.K_p + self.M_p +
                 self.MB_p + 2.0 * D_ap @ self.M_pp.T +
                 self.V_aL @ self.Delta_pL.T)
        ekin_a = D_ap @ self.K_p + setup.Kc
        e0_a = setup.MB + D_ap @ self.MB_p
        ecou_a = setup.M + D_ap[:, None] @ (
            self.M_p + D_ap @ self.M_pp.T)[:, :, None]
        ecou_a = ecou_a.reshape(-1)
        return as_np(ekin_a), as_np(e0_a), as_np(ecou_a), dH_ap

    @abstractmethod
    def evaluate(self):
        pass


def pack_xc_batches(D_asii, V_aL, setups, *, functional) -> list[PAWXCBatch]:
    """Group local atoms by tensor shape and gather their input arrays."""
    myid_a = {ind: setups.id_a[ind] for ind in D_asii.layout.atomdist.indices}
    atoms_batch: dict[int, list[Any]] = {}
    for a, id in myid_a.items():
        atoms_batch[id] = atoms_batch.get(id, []) + [a]

    assert functional.name in {'LDA', 'PBE'}

    cls: type[PAWXCBatch]
    if functional.name == 'LDA':
        from gpaw.new.xc.lda_xc_batch import LDAXCCorrection as cls
    else:
        from gpaw.new.xc.gga_xc_batch import PBEXCCorrection as cls

    return [cls(a_a, D_asii, V_aL, setups, functional=functional)
            for a_a in atoms_batch.values()]


def unpack_xc_batches(batches, results) -> dict[int, dict[str, Any]]:
    """Upnack xc batch energies and gradients into atom-indexed mappings."""
    corr_a = {}

    for batch, (energies, gradients) in zip(batches, results, strict=True):
        ekin_a, e0_a, ecou_a, dH_ap = batch.unpack()

        exc_a = as_np(energies)
        gradients += dH_ap
        gradients_saii = unpack_hermitian(gradients, xp=batch.xp)
        for position, atom in enumerate(batch.atom_indices):
            corr_a[atom] = {'e_xc': float(exc_a[position]),
                            'e_kin': float(ekin_a[position]),
                            'e_0': float(e0_a[position]),
                            'e_cou': float(ecou_a[position]),
                            'e_soc': float(0),
                            'gradient_sp': gradients[:, position],
                            'gradient_sii': gradients_saii[:, position]}
    return corr_a
