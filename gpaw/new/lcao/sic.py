"""Get PZ SIC energy and gradient"""


class PZSICLCAO:
    """"""

    def __init__(self, ibzwfs, density, pot_calc, sic_use_coarse_grid=True):
        self.setups = ibzwfs._wfs_u[:].setups
        self.basis_functions = ibzwfs._wfs_u[:].basis
        self.scaled_atom_pos = ibzwfs._wfs_u[:].relpos_ac

        # self.grid is UGDesc (presumably)
        self.grid = density.nt_sR.desc
        self.coarsegd = self.grid._gd
        self.finegd = pot_calc.fine_grid._gd

        self.sic_use_coarse_grid = sic_use_coarse_grid

        # setups = wfs.setups
        pass


# def get_orbital_potential_matrix
