import numpy as np
import os
from gpaw import restart, KohnShamConvergenceError
from gpaw.hubbard import aoom
from gpaw.utilities import unpack_density
from ase.units import Hartree
from ase.parallel import parprint, world, broadcast
from ase.geometry import find_mic
import time

class LinearResponseU():

    def __init__(self, atoms, calc, LR_atoms, alphas,
                 supercell=(1,1,1), lr_spin=None,
                 background=False, debug=False, redo=False,
                 fix_pot=None):
        """
        Initialize linear response U object.
        atoms: structure to consider; if doing supercell calculations, please
               provide only the unit cell and use the supercell parameter
        calc: calculator to use
        LR_atoms: atoms to be perturbed in LR calculations
        alphas: perturbation values to use
        supercell: how many times to repeat atoms in each direction
        lr_spin: None for all spin channels, 0 for up, 1 for down (I think)
        background: use background elements per Phys. Rev. B 71, 035105 (2005) ?
        debug: print additional debug output?
        redo: ignore output files that are already present, redo LR from scratch
        """

        self.atoms = atoms
        self.calc = calc
        self.LR_atoms = LR_atoms
        self.alphas = alphas
        self.lr_spin = lr_spin
        self.background = background
        self.debug = debug
        self.redo = redo
        self.world = world

        if fix_pot is not None:
            parprint('WARNING: fix_pot does not do anything, the Hubbard potential is always fixed.')

        #self.setups = setups

#        if self.calc.symmetry["point_group"] is not False:
#            parprint('POINT GROUP SYMMETRY IS ON! Make sure you actually want this.')
#            parprint('Symmetry will cause supercell calculations to be wrong,')
#            parprint('because responses of symmetry-equivalent atoms will become the same.')

        # print full matrices
        np.set_printoptions(threshold=np.inf)

        self.unit_cell = self.atoms.copy()
        self.atoms = self.atoms*supercell

        self.indices = self.generate_indices(self.atoms, self.LR_atoms)

        self.atoms.calc = calc




    def get_LR_Umatrix(self, U0 = None, keep_initial_gpw = False, initial_from_file = None):
        """
        Calculate the U matrix using linear response.
        U0: initial U value for each atom (as an array; maybe dict would be nicer?)
        keep_initial_gpw: if True, do not delete the initial density file
        initial_from_file: use supplied .gpw for initial density
        """

        if initial_from_file:
            filename = initial_from_file
            self.atoms, self.calc = restart(filename)
        else:
            # initialize calculator so we can read hubbard parameters from
            # hamiltonian; could get them manually without having to go
            # through the whole initialization, but this is ok for now
            self.calc.initialize(atoms=self.atoms)
            filename = 'LR_initial.gpw'
            # allow for different initial U on every site:
            #if U0 is not None:
            #    assert(len(U0) == len(self.atoms))
            #    setup_strings = []
            #    for i in range(0,len(U0)):
            #        ls = 'spdf'
            #        l = ls[self.calc.hamiltonian.setups[i].Hubl]
            #        scale = self.calc.hamiltonian.setups[i].Hubs
            #        setup_strings.append(':{0},{1},{2}'.format(l, U0[i], scale))
            #        setups = dict(list(enumerate(setup_strings)))
            #    if self.debug:
            #        parprint(setups)
            #self.atoms.calc = self.atoms.calc.new(setups=self.setups)
            #self.atoms.calc = self.atoms.calc
            # produce the initial density for LR calculations
            self.calc.calculate(atoms=self.atoms)
            self.calc.write(filename, mode='all')
            self.atoms, self.calc = restart(filename)

        n_resp_atoms = len(self.resp_atoms)
        bare_resp = np.zeros((n_resp_atoms, n_resp_atoms))
        conv_resp = np.zeros((n_resp_atoms, n_resp_atoms))

        for a in self.LR_atoms:
            for alpha in self.alphas:
                self.world.barrier() # make sure files are in sync
                if self.lr_spin == 0:
                    output_file = 'LR_' + str(a) + '_up_' + str(alpha) + '.txt'
                elif self.lr_spin == 1:
                    output_file = 'LR_' + str(a) + '_down_' + str(alpha) + '.txt'
                else:
                    output_file = 'LR_' + str(a) + '_' + str(alpha) + '.txt'

                # let's not redo calculations; if restarting, the aborted
                # calculation's output must be deleted by hand (for now,
                # though this could be automated)
                if os.path.exists(output_file):
                    if not self.redo:
                        parprint(output_file + ' exists already!')
                        continue

                self.world.barrier() # make sure files are in sync

                # reading the initial state from disk results in additional
                # I/O that could probably be avoided, but this is pretty handy
                # and works out okay for now
                self.atoms, self.calc = restart(filename, txt=output_file, maxiter=1)
                
                # put some stuff in place so we have a hamiltonian to fiddle with
                self.calc.initialize(atoms=self.atoms)
                self.calc.set_positions(atoms=self.atoms)
                #self.calc.calculate(system_changes=[])

                #self.calc.hamiltonian.update_hubbard_parameters(a, alpha/Hartree,
                #                                                lr_u=True, fix_pot=self.fix_pot)
                self.calc.hamiltonian.update(self.calc.density)  # update the whole hamiltonian
                # to take the changed hubbard parameters into account; if this is not done,
                # the alphas only start to affect the second SCF iteration
                self.calc.hamiltonian.setups[a].hubbard_u.alpha = alpha/Hartree
                self.calc.hamiltonian.setups[a].hubbard_u.lr_spin = self.lr_spin
                self.calc.hamiltonian.update(self.calc.density)  # update the whole hamiltonian

                self.calc.density.mixer.reset() # reset mixer; seems to fix things with bare occup!
                # apparently the mixer history is never zeroed when we start calculations
                # like this; also apparently the history from the previous calculation _is_ still
                # used when doing the first iteration for the perturbed calculation;
                # obviously it's not optimal to have 75 % old densities mixed in with the first
                # iteration density

                self.calc.scf.reset()
 
                # used to make SCF object store first-iteration density;
                # seems much smarter to not edit scf.py and just do a separate
                # one-iteration calculation here?

                # get first-iteration density
                #self.calc.new(maxiter=1)
                try:
                    self.calc.calculate(system_changes=[]) # system_changes=[] -> no initialization
                except KohnShamConvergenceError:
                    parprint('1st iteration done')
                    pass
                D_asp_bare = self.calc.density.D_asp.copy()
                
                # get converged density
                self.calc.scf.maxiter=333
#                self.calc.scf.reset()
#                conv_calc = self.calc.new(maxiter=333)
#                self.calc = conv_calc
#                self.atoms.calc = conv_calc
                
                self.world.barrier()
                self.calc.calculate(system_changes=[])
                D_asp_conv = self.calc.density.D_asp.copy()

                if self.debug:
                    parprint('calculation of alpha', alpha, 'for atom', a, 'done')
                # we don't want the calculator to reinitialize in the middle of LR_U, because
                # this causes old density/hamiltonian to disappear, U/alpha to be re-read from
                # the original setup strings, etc.
                
                self.world.barrier()
                # log occupations
                output = '\nLinear response occupations:\n'
                output += str(a) + ' '
                for r in self.resp_atoms:
                    output += str(r) + ' '
                output += '\n' + str(alpha) + ' '
                for r in self.resp_atoms:
                    l = self.calc.hamiltonian.setups[r].hubbard_u.l
                    bare_occup = self.N_occup(D_asp_bare, r, l, spin=self.lr_spin)
                    output += str(bare_occup) + ' '
                output += '\n' + str(alpha) + ' '
                for r in self.resp_atoms:
                    l = self.calc.hamiltonian.setups[r].hubbard_u.l
                    conv_occup = self.N_occup(D_asp_conv, r, l, spin=self.lr_spin)
                    output += str(conv_occup) + ' '
                output += '\n'

                self.calc.log(output)
                # make sure data gets written before somebody wants to read it
                self.calc.log.flush()

        self.world.barrier() # make sure everything has been written

        bare_occs, conv_occs = self.fetch_occupations()
        parprint('bare_occs', bare_occs)
        parprint('conv_occs', conv_occs)
        U_matrix = self.calculate_U(bare_occs, conv_occs)

        # never delete user-supplied .gpw
        # make sure we only delete after everybody is done
        self.world.barrier()
        if not keep_initial_gpw and not initial_from_file:
            try:
                os.remove(filename)
            except OSError:
                pass

        return U_matrix



    def calculate_U(self, bare_occs_list, conv_occs_list):
        """
        Function to calculate U matrices from occupation data.

        bare_occs: list of matrices containing the bare occupations
        conv_occs: list of matrices containing the converged occupations

        matrix format:
        pert_i  resp_indices
        alpha_1 occ_1 occ_2 ... occ_n
        ...
        alpha_n occ_1 occ_2 ... occ_n

        One such matrix for each perturbed atom.
        """

#        indices = self.generate_indices()

        bare_resp = self.construct_response(bare_occs_list)
        conv_resp = self.construct_response(conv_occs_list)

        if self.background:
            bare_resp = self.add_background(bare_resp)
            conv_resp = self.add_background(conv_resp)

        parprint('bare_resp', bare_resp)
        parprint('conv_resp', conv_resp)

        if self.background:
            # the matrices with background elements are singular by construction,
            # so perturb them a little to make them invertible;
            # if perturbations are equal, singularities cancel out in the
            # subtraction and we get a nice result (idea from resp_mat.f90);
            # result is actually independent of the magnitude of the perturbation
            # (though extreme values are of course numerically problematic)
            U_matrix = np.linalg.inv(bare_resp + 0.01) - np.linalg.inv(conv_resp + 0.01)
        else:
            U_matrix = np.linalg.inv(bare_resp) - np.linalg.inv(conv_resp)

        return U_matrix

    def fetch_occupations(self):
        bare_occs_list = []
        conv_occs_list = []

        for a in self.LR_atoms:
            bare_occs = []
            conv_occs = []
            for alpha in self.alphas:
                if self.lr_spin == 0:
                    filename = 'LR_' + str(a) + '_up_' + str(alpha) + '.txt'
                elif self.lr_spin == 1:
                    filename = 'LR_' + str(a) + '_down_' + str(alpha) + '.txt'
                else:
                    filename = 'LR_' + str(a) + '_' + str(alpha) + '.txt'
                at_occ = 0
                with open(filename, 'r') as f:
                    for line in f:
                        if at_occ == 1:
                            resp_atom_str = line.strip()
                            at_occ += 1
                        elif at_occ == 2:
                            bare_occ_str = line.strip()
                            at_occ += 1
                        elif at_occ == 3:
                            conv_occ_str = line.strip()
                            at_occ = 0
                        elif line == 'Linear response occupations:\n':
                            at_occ += 1

                bare_occ = [float(i) for i in bare_occ_str.split(' ')]
                conv_occ = [float(i) for i in conv_occ_str.split(' ')]
                bare_occs.append(bare_occ)
                conv_occs.append(conv_occ)
            
            resp_atoms = [int(i) for i in resp_atom_str.split(' ')]
            bare_occs.insert(0, resp_atoms)
            conv_occs.insert(0, resp_atoms)

            bare_occs_list.append(np.asarray(bare_occs))
            conv_occs_list.append(np.asarray(conv_occs))

        return bare_occs_list, conv_occs_list



    def generate_indices(self, atoms, LR_atoms):
        atom_indices = []
        for atom in self.atoms:
            atom_indices.append(atom.index)

        # make a list of dictionaries to keep track of various indices
        # (could just be a matrix, but maybe this is more readable)
        dicts = [None for x in range(len(atom_indices))]

        pos = self.atoms.get_positions()
        # get the wrapped positions to help find equivalent atoms
        cell = self.atoms.get_cell()
        # get_positions(wrap=True) may not work as expected for atoms on edges of cell
        # contract cell a tiny bit to fix
        #self.atoms.set_cell(self.unit_cell.get_cell() - 1e-10*np.eye(3))
        self.atoms.set_cell((1-1e-8)*self.unit_cell.get_cell())
        pos_wrap = self.atoms.get_positions(wrap=True)
        self.atoms.set_cell(cell)

        for atom in self.atoms:
            dict = {}
            dict["atom"] = atom.index
            dict["equiv"] = []
            if atom.index in self.LR_atoms:
               dict["LR"] = self.LR_atoms.index(atom.index)
               dict["resp"] = dict["LR"]
            dicts[atom.index] = dict
        
        for atom1 in self.atoms:
            for atom2 in self.atoms:
                if np.linalg.norm(pos_wrap[atom1.index] - pos_wrap[atom2.index]) < 1e-6:
                    dicts[atom1.index]["equiv"].append(atom2.index)

        self.resp_atoms = []
        for atom in self.atoms:
            dict = dicts[atom.index]
            eq = dict["equiv"]
            for x in self.LR_atoms:
                if x in eq:
                    self.resp_atoms.append(atom.index)
                    if "copies" not in dicts[eq[0]]:
                        dicts[eq[0]]["copies"] = 0
                    else:
                        dicts[eq[0]]["copies"] += 1
                    copies = dicts[eq[0]]["copies"]
                    n_LR_atoms = len(self.LR_atoms)
                    dict["resp"] = dicts[eq[0]]["resp"] + copies*n_LR_atoms

        if self.debug:
            parprint('pos', pos)
            parprint('pos_wrap', pos_wrap)
            parprint('dicts',dicts)
            parprint('resp_atoms', self.resp_atoms)

        return dicts

            
    def construct_response(self, occs_list):
        n_resp_atoms = np.size(occs_list[0], 1) - 1
        resp = np.zeros((n_resp_atoms, n_resp_atoms))
        for occs in occs_list:
            if self.debug:
                parprint('occs', occs)
                parprint('n_resp_atoms', n_resp_atoms)
            perturbed_atom = int(occs[0, 0])
            pert_i = self.indices[perturbed_atom]["resp"]
            alphas = occs[1:, 0]

            for col in range(1, n_resp_atoms+1):
                response_atom = int(occs[0, col])
                slope = self.get_slope(alphas, occs[1:,col])
                resp_i = self.indices[response_atom]["resp"]
                resp[resp_i, pert_i] = slope
            # these outputs are excessive for production runs even in debug mode
#                if self.debug:
#                    parprint('pert_i', pert_i)
#                    parprint('resp_i', resp_i)
#                    parprint('slope', slope)
#                    parprint('resp', resp)

        parprint('resp_before_completion', resp)

        # complete response matrices by geometry
        for pert_atom in self.resp_atoms:
            # find the index of the corresponding, actually perturbed atom
            for eq in self.indices[pert_atom]["equiv"]:
                if "LR" in self.indices[eq]:
                    actual_pert_atom = eq
            # find vector that points from actually perturbed atom to current atom
            # vector = self.atoms[pert_atom].position - self.atoms[actual_pert_atom].position
            vector = self.atoms.get_distance(actual_pert_atom, pert_atom, mic=True, vector=True)
            # get response matrix indices for both
            actual_pert_i = self.indices[actual_pert_atom]["resp"]
            pert_i = self.indices[pert_atom]["resp"]
            for resp_atom in self.resp_atoms:
                actual_resp_atom_pos = self.atoms[resp_atom].position - vector
                for eq in self.indices[resp_atom]["equiv"]:
                    resp_atom_pos = self.atoms[eq].position
                    #parprint(actual_resp_atom_pos)
                    #parprint(resp_atom_pos)
                    #parprint(actual_resp_atom_pos - resp_atom_pos)
                    dist_vec = find_mic([actual_resp_atom_pos - resp_atom_pos], self.atoms.cell)
                    #parprint(dist_vec[0])
                    if np.linalg.norm(dist_vec[0]) < 1e-6:
                        actual_resp_atom = eq
             # excessive debug output
#                if self.debug:
#                    parprint('actual_pert_i', actual_pert_i)
#                    parprint('pert_i', pert_i)
#                    parprint('pert_atom', pert_atom)
#                    parprint('resp_atom', resp_atom)
#                    parprint('actual_resp_atom', actual_resp_atom)
                actual_resp_i = self.indices[actual_resp_atom]["resp"]
                resp_i = self.indices[resp_atom]["resp"]
                resp[resp_i, pert_i] = resp[actual_resp_i, actual_pert_i]

        if self.debug:
            parprint('resp', resp)

        return resp


    def add_background(self, mat):
        """
        Helper function to add background elements to response matrices.
        Defined according to Phys. Rev. B 71, 035105 (2005).

        mat: input matrix
        """

        mat = np.vstack([mat, -np.sum(mat, axis=0)])
        # vstack to make the sum array vertical
        mat = np.hstack([mat, -np.vstack(np.sum(mat, axis=1))])
        return mat

    #TODO Self-consistent U does not work with new output strategy!
    # Shouldn't be too hard to fix, probably just add initial U to
    # output filenames (or make directories for each SC iteration)
    # and tinker a bit
    def get_self_consistent_Umatrix(self, U0=None, maxiter=20, tol=0.1, steffensen=False):
        """
        Calculate the (diagonal elements of the) U matrix self-consistently.

        U0: initial U matrix / vector
        maxiter: maximum number of iterations
        tol: when max(abs(U1-U0)) < tol, calculation is converged
        steffensen: use Steffensen's method to accelerate convergence
        """

        self.calc.initialize(atoms=self.atoms) # to initialize setups so we can read HubU
        if U0 is None:
            U0 = np.zeros(len(self.atoms))
            for a in range(0,len(self.atoms)):
                U0[a] = self.calc.hamiltonian.setups[a].HubU*Hartree
        else:
            U0 = np.array(U0)
 
        if U0.shape == (2,2):
            U0 = U0.diagonal()

        assert(len(U0) == len(self.atoms))

        U_hist = U0

        if not steffensen:
            for i in range(0,maxiter):
                U1 = np.zeros(len(self.atoms))
                U1[self.LR_atoms] = self.get_LR_Umatrix(U0=U0).diagonal()
                U_hist = np.vstack((U_hist, U1))
                if self.debug:
                    parprint('U_hist', U_hist)
                if max(abs(U1-U0)) < tol:
                    return U_hist
                U0 = U1

        else:
            i = 0
            while i < maxiter:
                U1 = np.zeros(len(self.atoms))
                U1[self.LR_atoms] = self.get_LR_Umatrix(U0=U0).diagonal()
                U_hist = np.vstack((U_hist, U1))
                if self.debug:
                    parprint('U_hist', U_hist)
                if max(abs(U1-U0)) < tol:
                    return U_hist
                i += 1
                U2 = np.zeros(len(self.atoms))
                U2[self.LR_atoms] = self.get_LR_Umatrix(U0=U1).diagonal()
                U_hist = np.vstack((U_hist, U2))
                if self.debug:
                    parprint('U_hist', U_hist)
                if max(abs(U2-U1)) < tol:
                    return U_hist
                i += 1
                U_steff = U0 - (U1 - U0)**2 / (U2 - 2*U1 + U0)
                U0[self.LR_atoms] = U_steff[self.LR_atoms]
                U_hist = np.vstack((U_hist, U0))
                if self.debug:
                    parprint('U_hist', U_hist)

        parprint('Self-consistent U calculation did not converge! History:')
        parprint(U_hist)

    def get_slope(self, x, y):
        """
        Get linearly regressed slope of (x,y) data.

        x: independent variable
        y: dependent variable
        """

        linreg_y = y
        linreg_X = np.vstack([x, np.ones(len(x))]).T
        regression = np.linalg.lstsq(linreg_X, linreg_y)
        slope, intercept = regression[0]
        residuals = regression[1]
        if self.debug:
            parprint('linreg_y', linreg_y)
            parprint('slope, intercept', slope, intercept)
            parprint('residuals', residuals)
        return slope


    def N_occup(self, D_asp, a, l, scale = 1, spin=None):
        """
        Get occupancy of site.

        a: atom index
        l: azimuthal quantum number
        scale: scale the projections/do not do it
        selfconsistent: if False, get occupation from first SCF iteration
                        if True, get occupation from last SCF iteration
        """

        setup = self.calc.hamiltonian.setups[a]
        nspin = self.calc.hamiltonian.nspins
        N = np.array([0.])
        if self.calc.density.atom_partition.rank_a[a] == self.calc.world.rank:
            N = 0.
            if spin == None:
                for s in range(nspin):
                    N_mm, V = aoom(unpack_density(D_asp[a][s]),l,setup.l_j,setup.n_j,setup.N0_q,scale=scale)
                    N += np.trace(N_mm)
            else:
                    N_mm, V = aoom(unpack_density(D_asp[a][spin]),l,setup.l_j,setup.n_j,setup.N0_q,scale=scale)
                    N += np.trace(N_mm)

            N = np.array([N])
        self.calc.hamiltonian.world.broadcast(N, self.calc.density.atom_partition.rank_a[a])
        return N[0]
