"""Module for calculating the multipole moment(upto the hexadeacapole) 
   and polarizability(upto the quadrupole-quadrupole) tensors .

See:

   https://arxiv.org/abs/2409.10184
"""

import numpy as np
from gpaw.external import apply_field_gradient, apply_field
import _gpaw
from ase.units import Bohr, Ha


def calculate_pert_moments( atoms, 
                            pert_magnitude,
                            pert_direction,
                            pert_type):
    """Helper function to switch between field and field gradient perturbations.

    atoms: ASE instance
        ASE Atoms object.
    pert_magnitude: 
        Strength of field/field-gradient perturbation
    pert_direction: Direction of the field/field-gradient
        For fields values are : x, y, and z
        Field-gradient values are: xx, xy, xz, yy, yz, zz
    pert_type: Type of perturbation
        Choose between field (apply_field) 
        and field-gradient (apply_field_gradient)
    """

    shifted_point = np.array([0.0,0.0,0.0])
    E   = pert_type(atoms, 
                    shifted_point,
                    pert_magnitude ,
                    pert_direction)
    
    energy = atoms.get_potential_energy()
    CM = atoms.cell.diagonal() / (2.0*Bohr)

    D_v = np.zeros(3)
    Q_v = np.zeros((3,3))
    O_v = np.zeros((3,3,3))
    H_v = np.zeros((3,3,3,3))

    _gpaw.calculate_moments(atoms.calc.density.finegd.beg_c,
                atoms.calc.density.finegd.h_cv.diagonal().copy(),
                CM,
                atoms.calc.density.rhot_g,
                D_v, Q_v, O_v, H_v)

    atoms.calc.density.finegd.comm.sum(D_v)     
    atoms.calc.density.finegd.comm.sum(Q_v)
    atoms.calc.density.finegd.comm.sum(O_v)
    atoms.calc.density.finegd.comm.sum(H_v)


    return energy, D_v, Q_v, O_v, H_v

class MomTensor:
    def __init__(self,
                 atoms):
        """Multipole moment tensor class.

        atoms: ASE instance
            ASE Atoms object.
        """
        self.moment_ref = calculate_pert_moments(atoms, 0.0, 'xx', apply_field_gradient) 
        self.e_0, self.d_0, self.q_0, self.o_0, self.h_0 = self.moment_ref


    def get_dipole(self):
        """Get the dipole tensor"""

        return self.d_0

    def get_qpole(self):
        """Get the qpole tensor"""

        return self.q_0

    def get_opole(self):
        """Get the opole tensor"""

        return self.o_0

    def get_hpole(self):
        """Get the hpole tensor"""

        return self.h_0

    def get_multipole_moments(self):
        """Get the allthe multipole moment tensors in 
        the order dipole, qpole, opole, hpole"""

        return self.moment_ref



class PolarTensor:

    def __init__(self,
                 atoms, 
                 field_strength = 0.486/Ha*Bohr,
                 field_grad_strength = 0.0486/Ha*Bohr*Bohr,
                 irrep = False):
        """Multipole polarizability tensor class.

        atoms: ASE instance
            ASE Atoms object.
        field_strength: magnitude of perturbation by the field
            The default value of '0.486/Ha*Bohr' was used based on 
            the discripancy between the moment-based (https://arxiv.org/abs/2409.10184) 
            and energy-based scheme(https://doi.org/10.1002/jcc.24431) 
            and using convergence of the tensor values.
        field_grad_strength: magnitude of perturbation by the field-gradient
            The default value was selected similarly to the case of the field perturbation.
        irrep : flag to provide the irreducible representation
            It defaults to the full represent. Hence irrep = False.
        """
        self.atoms        =   atoms

        self.field_direction_x          =   'x'
        self.field_direction_y          =   'y'
        self.field_direction_z          =   'z'
        self.field_strength             =   field_strength 

        self.field_grad_direction_xx   =   'xx'
        self.field_grad_direction_xy   =   'xy'
        self.field_grad_direction_xz   =   'xz'
        self.field_grad_direction_yy   =   'yy'
        self.field_grad_direction_yz   =   'yz'
        self.field_grad_direction_zz   =   'zz'
        self.field_grad_strength    =   field_grad_strength

        self.irrep = irrep

        """Calculation of dpole, qpole, opole and hpole tensor components 
        for different perturbations"""

        self.moment_field_pert_x      =   calculate_pert_moments(   self.atoms, 
                                                                    self.field_strength,
                                                                    self.field_direction_x,
                                                                    apply_field)
        self.moment_field_pert_y      =   calculate_pert_moments(   self.atoms, 
                                                                    self.field_strength,
                                                                    self.field_direction_y,
                                                                    apply_field)
        self.moment_field_pert_z      =   calculate_pert_moments(   self.atoms, 
                                                                    self.field_strength,
                                                                    self.field_direction_z,
                                                                    apply_field)
 

        self.moment_field_grad_pert_xx =   calculate_pert_moments(  self.atoms, 
                                                                    self.field_grad_strength,
                                                                    self.field_grad_direction_xx,
                                                                    apply_field_gradient) 
        
        self.moment_field_grad_pert_xy =   calculate_pert_moments(  self.atoms, 
                                                                    self.field_grad_strength,
                                                                    self.field_grad_direction_xy,
                                                                    apply_field_gradient) 
        
        self.moment_field_grad_pert_xz =   calculate_pert_moments(  self.atoms, 
                                                                    self.field_grad_strength,
                                                                    self.field_grad_direction_xz,
                                                                    apply_field_gradient) 
        
        self.moment_field_grad_pert_yy =   calculate_pert_moments(  self.atoms, 
                                                                    self.field_grad_strength,
                                                                    self.field_grad_direction_yy,
                                                                    apply_field_gradient) 
        
        self.moment_field_grad_pert_yz =   calculate_pert_moments(  self.atoms, 
                                                                    self.field_grad_strength,
                                                                    self.field_grad_direction_yz,
                                                                    apply_field_gradient) 
        
        self.moment_field_grad_pert_zz =   calculate_pert_moments(  self.atoms, 
                                                                    self.field_grad_strength,
                                                                    self.field_grad_direction_zz,
                                                                    apply_field_gradient) 
        
        """The unpertubred dpole, qpole, opole and hpole reference values"""
        self.moment_ref = calculate_pert_moments(self.atoms, 0.0, 'xx', apply_field_gradient) 





    def calc_dd_pol(self):
        """calculate the dipole-dipole polarizability tensor"""
        _, d_0, _, _, _ = self.moment_ref
        _, d_x, _, _, _ = self.moment_field_pert_x 
        _, d_y, _, _, _ = self.moment_field_pert_y 
        _, d_z, _, _, _ = self.moment_field_pert_z 

        alp_i_x = (d_x - d_0)/self.field_strength
        alp_i_y = (d_y - d_0)/self.field_strength
        alp_i_z = (d_z - d_0)/self.field_strength

        """Irreducible representation tensor"""

        self.alp_ij_IRREP     = np.zeros((6))
        self.alp_ij_IRREP[0]  = alp_i_x[0]
        self.alp_ij_IRREP[1]  = alp_i_x[1]
        self.alp_ij_IRREP[2]  = alp_i_x[2]
        self.alp_ij_IRREP[3]  = alp_i_y[1]
        self.alp_ij_IRREP[4]  = alp_i_y[2]
        self.alp_ij_IRREP[5]  = alp_i_z[2] 

        """Full representation tensor"""
        self.alp_ij = np.zeros((3,3))
        self.alp_ij[:,0] = alp_i_x
        self.alp_ij[:,1] = alp_i_y
        self.alp_ij[:,2] = alp_i_z



        if self.irrep == True:
            return np.round(-self.alp_ij_IRREP,4)
        else:
            return np.round(-self.alp_ij,4)

    def calc_dq_pol(self):
        """calculate the dipole-quadrupole polarizability tensor"""

        _, _, q_0, _, _ = self.moment_ref
        _, _, q_x, _, _ = self.moment_field_pert_x 
        _, _, q_y, _, _ = self.moment_field_pert_y 
        _, _, q_z, _, _ = self.moment_field_pert_z 
        _, d_xx, _, _, _ = self.moment_field_grad_pert_xx 
        _, d_yy, _, _, _ = self.moment_field_grad_pert_yy 
        _, d_zz, _, _, _ = self.moment_field_grad_pert_zz

        A_i_xx = (d_xx - d_yy)/self.field_grad_strength
        A_i_yy = (d_yy - d_zz)/self.field_grad_strength
        A_i_zz = (d_zz - d_xx)/self.field_grad_strength
        A_ij_x = (q_x - q_0)/self.field_strength
        A_ij_y = (q_y - q_0)/self.field_strength
        A_ij_z = (q_z - q_0)/self.field_strength


        """Irreducible representation tensor"""
        self.A_ijk_IRREP                = np.zeros((18))
        self.A_ijk_IRREP[[0,6,12]]      = A_i_xx
        self.A_ijk_IRREP[[3,9,15]]      = A_i_yy
        self.A_ijk_IRREP[[5,11,17]]     = A_i_zz
        self.A_ijk_IRREP[[1,2,4]]       = A_ij_x[[0,0,1],[1,2,2]]
        self.A_ijk_IRREP[[7,8,10]]      = A_ij_y[[0,0,1],[1,2,2]]
        self.A_ijk_IRREP[[13,14,16]]    = A_ij_z[[0,0,1],[1,2,2]]



        """Full representation tensor"""
        self.A_ijk = np.zeros((3,3,3))
        self.A_ijk[:,0,0] = A_i_xx
        self.A_ijk[:,1,1] = A_i_yy
        self.A_ijk[:,2,2] = A_i_zz
        self.A_ijk[0,[0,0,1],[1,2,2]] = A_ij_x[[0,0,1],[1,2,2]]
        self.A_ijk[1,[0,0,1],[1,2,2]] = A_ij_y[[0,0,1],[1,2,2]]
        self.A_ijk[2,[0,0,1],[1,2,2]] = A_ij_z[[0,0,1],[1,2,2]]
        self.A_ijk[0,[1,2,2],[0,0,1]] = A_ij_x[[0,0,1],[1,2,2]]
        self.A_ijk[1,[1,2,2],[0,0,1]] = A_ij_y[[0,0,1],[1,2,2]]
        self.A_ijk[2,[1,2,2],[0,0,1]] = A_ij_z[[0,0,1],[1,2,2]]

        if self.irrep == True:
            return np.round(-self.A_ijk_IRREP,4)
        else:
            return np.round(-self.A_ijk,4)



    def calc_qq_pol(self):
        """calculate the quadrupole-quadrupole polarizability tensor"""

        _, _,  q_0, _, _ = self.moment_ref
        _, _, q_xx, _, _ = self.moment_field_grad_pert_xx 
        _, _, q_xy, _, _ = self.moment_field_grad_pert_xy 
        _, _, q_xz, _, _ = self.moment_field_grad_pert_xz 
        _, _, q_yy, _, _ = self.moment_field_grad_pert_yy 
        _, _, q_yz, _, _ = self.moment_field_grad_pert_yz 
        _, _, q_zz, _, _ = self.moment_field_grad_pert_zz


        C_ijxy = (q_xy - q_0)/self.field_grad_strength
        C_ijyz = (q_yz - q_0)/self.field_grad_strength
        C_ijxz = (q_xz - q_0)/self.field_grad_strength
        C_xyxy = C_ijxy[0,1]/2.0
        C_xyyz = C_ijyz[0,1]/2.0
        C_xzyz = C_ijyz[0,2]/2.0
        C_yzyz = C_ijyz[1,2]/2.0
        C_xyxz = C_ijxz[0,1]/2.0
        C_xzxz = C_ijxz[0,2]/2.0
        C_ijxxyy = (q_xx - q_yy)/self.field_grad_strength
        C_ijyyzz = (q_yy - q_zz)/self.field_grad_strength
        C_ijzzxx = (q_zz - q_xx)/self.field_grad_strength
        C_xxxx =  -C_ijxxyy[0,0]/3.0
        C_xxyy =  -C_ijxxyy[1,1]/3.0
        C_xxzz =  -C_ijxxyy[2,2]/3.0
        C_yyyy =  -C_ijyyzz[1,1]/3.0
        C_yyzz =  -C_ijyyzz[2,2]/3.0
        C_zzzz =  -C_ijzzxx[2,2]/3.0
        C_ijxxyy = (q_xx - q_yy)/self.field_grad_strength
        cxyxx =  -C_ijxxyy[0,1]/3.0
        cxzxx =  -C_ijxxyy[0,2]/3.0
        cyzxx =  -C_ijxxyy[1,2]/3.0
        C_ijxxyy = (q_yy - q_zz)/self.field_grad_strength
        cxyyy =  -C_ijxxyy[0,1]/3.0
        cxzyy =  -C_ijxxyy[0,2]/3.0
        cyzyy =  -C_ijxxyy[1,2]/3.0
        C_ijxxyy = (q_zz - q_xx)/self.field_grad_strength
        cxyzz =  -C_ijxxyy[0,1]/3.0
        cxzzz =  -C_ijxxyy[0,2]/3.0
        cyzzz =  -C_ijxxyy[1,2]/3.0
        cxyxx = (q_xy[0,0]- q_0[0,0])/(2*self.field_grad_strength)
        cxyyy = (q_xy[1,1]- q_0[1,1])/(2*self.field_grad_strength)
        cxyzz = (q_xy[2,2]- q_0[2,2])/(2*self.field_grad_strength)
        cxzxx = (q_xz[0,0]- q_0[0,0])/(2*self.field_grad_strength)
        cxzyy = (q_xz[1,1]- q_0[1,1])/(2*self.field_grad_strength)
        cxzzz = (q_xz[2,2]- q_0[2,2])/(2*self.field_grad_strength)
        cyzxx = (q_yz[0,0]- q_0[0,0])/(2*self.field_grad_strength)
        cyzyy = (q_yz[1,1]- q_0[1,1])/(2*self.field_grad_strength)
        cyzzz = (q_yz[2,2]- q_0[2,2])/(2*self.field_grad_strength)

        """Irreducible representation tensor"""
        self.C_ijkl_IRREP = np.zeros(21)
        self.C_ijkl_IRREP[0]   = round(float(C_xxxx),5)
        self.C_ijkl_IRREP[1]   = round(float(cxyxx),5)
        self.C_ijkl_IRREP[2]   = round(float(cxzxx),5)
        self.C_ijkl_IRREP[3]   = round(float(C_xxyy),5)
        self.C_ijkl_IRREP[4]   = round(float(cyzxx),5)
        self.C_ijkl_IRREP[5]   = round(float(C_xxzz),5)
        self.C_ijkl_IRREP[6]   = round(float(C_xyxy),5)
        self.C_ijkl_IRREP[7]   = round(float(C_xyxz),5)
        self.C_ijkl_IRREP[8]   = round(float(cxyyy),5)
        self.C_ijkl_IRREP[9]   = round(float(C_xyyz),5)
        self.C_ijkl_IRREP[10]  = round(float(cxyzz),5)
        self.C_ijkl_IRREP[11]  = round(float(C_xzxz),5)
        self.C_ijkl_IRREP[12]  = round(float(cxzyy),5)
        self.C_ijkl_IRREP[13]  = round(float(C_xzyz),5)
        self.C_ijkl_IRREP[14]  = round(float(cxzzz),5)
        self.C_ijkl_IRREP[15]  = round(float(C_yyyy),5)
        self.C_ijkl_IRREP[16]  = round(float(cyzyy),5)
        self.C_ijkl_IRREP[17]  = round(float(C_yyzz),5)
        self.C_ijkl_IRREP[18]  = round(float(C_yzyz),5)
        self.C_ijkl_IRREP[19]  = round(float(cyzzz),5)
        self.C_ijkl_IRREP[20]  = round(float(C_zzzz),5)

        """Full representation tensor"""
 
        self.C_ijkl = np.zeros((3,3,3,3))

        self.C_ijkl[0,0,0,0] = self.C_ijkl_IRREP[0]
        self.C_ijkl[[0,1,0,0],
                    [1,0,0,0],
                    [0,0,0,1],
                    [0,0,1,0]] = self.C_ijkl_IRREP[1]
        self.C_ijkl[[0,2,0,0],
                    [2,0,0,0],
                    [0,0,0,2],
                    [0,0,2,0]] = self.C_ijkl_IRREP[2]
        self.C_ijkl[[1,0],
                    [1,0],
                    [0,1],
                    [0,1]] = self.C_ijkl_IRREP[3]
        self.C_ijkl[[1,2,0,0],
                    [2,1,0,0],
                    [0,0,1,2],
                    [0,0,2,1]] = self.C_ijkl_IRREP[4]
        self.C_ijkl[[2,0],
                    [2,0],
                    [0,2],
                    [0,2]] = self.C_ijkl_IRREP[5]
        self.C_ijkl[[0,0,1,1],
                    [1,1,0,0],
                    [0,1,0,1],
                    [1,0,1,0]] = self.C_ijkl_IRREP[6]
        self.C_ijkl[[0,0,2,2,0,1,0,1],
                    [2,2,0,0,1,0,1,0],
                    [0,1,0,1,0,0,2,2],
                    [1,0,1,0,2,2,0,0]] = self.C_ijkl_IRREP[7]
        self.C_ijkl[[1,1,0,1],
                    [1,1,1,0],
                    [0,1,1,1],
                    [1,0,1,1]] = self.C_ijkl_IRREP[8]
        self.C_ijkl[[1,1,2,2,0,1,0,1],
                    [2,2,1,1,1,0,1,0],
                    [0,1,0,1,1,1,2,2],
                    [1,0,1,0,2,2,1,1]] = self.C_ijkl_IRREP[9]
        self.C_ijkl[[2,2,0,1],
                    [2,2,1,0],
                    [0,1,2,2],
                    [1,0,2,2]] = self.C_ijkl_IRREP[10]
        self.C_ijkl[[0,0,2,2],
                    [2,2,0,0],
                    [0,2,0,2],
                    [2,0,2,0]] = self.C_ijkl_IRREP[11]
        self.C_ijkl[[1,1,0,2],
                    [1,1,2,0],
                    [0,2,1,1],
                    [2,0,1,1]] = self.C_ijkl_IRREP[12]
        self.C_ijkl[[1,1,2,2,0,2,0,2],
                    [2,2,1,1,2,0,2,0],
                    [0,2,0,2,1,1,2,2],
                    [2,0,2,0,2,2,1,1]] = self.C_ijkl_IRREP[13]
        self.C_ijkl[[2,2,0,2],
                    [2,2,2,0],
                    [0,2,2,2],
                    [2,0,2,2]] = self.C_ijkl_IRREP[14]
        self.C_ijkl[1,1,1,1] = self.C_ijkl_IRREP[15]
        self.C_ijkl[[1,2,1,1],
                    [2,1,1,1],
                    [1,1,1,2],
                    [1,1,2,1]] = self.C_ijkl_IRREP[16]
        self.C_ijkl[[2,1],
                    [2,1],
                    [1,2],
                    [1,2]] = self.C_ijkl_IRREP[17]
        self.C_ijkl[[1,1,2,2],
                    [2,2,1,1],
                    [1,2,1,2],
                    [2,1,2,1]] = self.C_ijkl_IRREP[18]
        self.C_ijkl[[2,2,1,2],
                    [2,2,2,1],
                    [1,2,2,2],
                    [2,1,2,2]] = self.C_ijkl_IRREP[19]
        self.C_ijkl[2,2,2,2] = self.C_ijkl_IRREP[20]

        if self.irrep == True:
            return np.round(self.C_ijkl_IRREP,4)
        else:
            return np.round(self.C_ijkl,4)

