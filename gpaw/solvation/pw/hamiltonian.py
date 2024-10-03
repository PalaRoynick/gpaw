import numpy as np
from ase.units import Ha

from gpaw.pw.hamiltonian import ReciprocalSpaceHamiltonian
from gpaw.fd_operators import Gradient
from gpaw.io.logger import indent
from gpaw.templogger import logg
from gpaw.solvation.pw.poisson import WrappedWeightedFDPoissonSolver

# XXX/CB: file currently under development

# XXX/CB: Seems like the key decisions we need to make right now are
# all about what we want to keep in real space and what we want to carry in
# planewave. If we take a hint from vanilla reciprocal space, we can get a few
# answers, but we need to decide how to tackle the cavity/dielectric
# management. My proposal is to first implement this so that it is essentially
# just a real space calculation, except we fft everything before doing any
# super calls and see if we can replicate standard solvation results. Bit of a
# Frankenstein, proof of concept experiment.


class SolvationReciprocalSpaceHamiltonian(ReciprocalSpaceHamiltonian):

    def __init__(
        self,
        # solvation-related arguments:
        cavity, dielectric, interactions,
        # ReciprocalSpaceHamiltonian arguments:
        gd, finegd, pd2, pd3, nspins, collinear,
        setups, timer, xc, world, xc_redistributor,
        vext=None, psolver=None, redistributor=None, realpbc_c=None,
            charge=0.0):
        """Constructor of SolvationReciprocalSpaceHamiltonian class.

        Additional arguments not present in ReciprocalSpaceHamiltonian:
        cavity       -- A Cavity instance.
        dielectric   -- A Dielectric instance.
        interactions -- A list of Interaction instances.
        """
        self.solvation = {'cavity': cavity,
                          'dielectric': dielectric,
                          'interactions': interactions}

        if self.solvation['cavity'].depends_on_el_density or \
           self.solvation['interactions'].depends_on_el_density:
            raise NotImplementedError(
                'Current implementation would require an inverse' +
                'Fourier transform in the step that creates dependence')

        # XXX/CB: First 4 lines will remain the same, update will be in our
        # new cavity and dielectric objs if we go that route
        self.solvation['cavity'].set_grid_descriptor(finegd)
        self.solvation['dielectric'].set_grid_descriptor(finegd)
        for ia in self.solvation['interactions']:
            ia.set_grid_descriptor(finegd)
        if psolver is None:
            psolver = WrappedWeightedFDPoissonSolver()
        psolver.set_dielectric(self.dielectric)
        self.gradient = None

        ReciprocalSpaceHamiltonian.__init__(
            self=self,
            gd=gd, finegd=finegd, pd2=pd2, pd3=pd3, nspins=nspins,
            collinear=collinear, setups=setups, timer=timer, xc=xc,
            world=world, xc_redistributor=xc_redistributor,
            vext=vext, psolver=psolver, redistributor=redistributor,
            realpbc_c=realpbc_c, charge=charge)

        for ia in interactions:
            setattr(self, 'e_' + ia.subscript, None)
        self.new_atoms = None

        # XXX/CB: Only question is whether these will be in q or g...my guess
        # is that for first run, we try to have most things in grid, and slowly
        # move over to pw. Can explain what I mean by this.
        self.vt_ia_g = None

        self.e_el_free = None
        self.e_el_extrapolated = None

        raise NotImplementedError('TODO')

    # XXX/REPLACE:
    # def real_space_conversion(arr):
    #     return self.pd3.ifft(arr)

    def __str__(self):
        s = ReciprocalSpaceHamiltonian.__str__(self) + '\n'
        s += '  Solvation:\n'
        components = [self.solvation['cavity'],
                      self.solvation['dielectric']] + \
            self.solvation['interactions']
        s += ''.join([indent(str(c), 2) for c in components])
        return s

    def estimate_memory(self, mem):
        ReciprocalSpaceHamiltonian.estimate_memory(self, mem)
        # XXX/CB: Should be identical. Super call is really pulling from
        # Hamiltonian, and other estimators we will get to when replicating a
        # pw cavity, dielectric, etc. if we do them that way

        ReciprocalSpaceHamiltonian.estimate_memory(self, mem)
        solvation = mem.subnode('Solvation')
        objs = [
            ('Cavity', self.solvation['cavity']),
            ('Dielectric', self.solvation['dielectric']),
        ] + [
            ('Interaction: ' + ia.subscript, ia)
            for ia in self.solvation['interactions']
        ]

        for name, obj in objs:
            obj.estimate_memory(solvation.subnode(name))

    def update_atoms(self, atoms, log):
        logg('SolvationReciprocalSpaceHamiltonian.update_atoms called.',
             traceback=True)
        self.new_atoms = atoms.copy()
        log('Solvation position-dependent initialization:')
        self.solvation['cavity'].update_atoms(atoms, log)
        for ia in self.solvation['interactions']:
            ia.update_atoms(atoms, log)

    def initialize(self):
        self.gradient = [
            Gradient(self.finegd, i, 1.0, self.poisson.nn) for i in (0, 1, 2)
        ]
        self.vt_ia_g = self.finegd.zeros()

        # XXX/CB: these shouldn't change...they depend on our auxiliary files.
        self.solvation['cavity'].allocate()
        self.solvation['dielectric'].allocate()
        for ia in self.solvation['interactions']:
            ia.allocate()
        ReciprocalSpaceHamiltonian.initialize(self)

        raise NotImplementedError('TODO')

    def update(self, density, wfs=None, kin_en_using_band=True):
        # XXX/CB: Need to find a way to intercovert real and reciprocal space
        # density objects...or, can just hold in reciprocal and convert back
        # for rare cases where we have depends_on_el_density = True

        # XXX/CB: should be highly similar, only depends on the space in
        # which we hold our running potentials.

        self.timer.start('Hamiltonian')
        if self.vt_sg is None:
            with self.timer('Initialize Hamiltonian'):
                self.initialize()

        # XXX/CB: should also be highly similar, updated in aux. Theme
        # of similarity in this func. corroborated by the fact that
        # pw/hamiltonian does NOT override update.

        cavity_changed = self.solvation['cavity'].update(
            self.new_atoms, density)
        # XXX/REPLACE:
        # cavity_changed = self.solvation['cavity'].update(
        #     self.new_atoms, self.real_space_conversion(density.rhot_q))
        if cavity_changed:
            self.solvation['cavity'].update_vol_surf()
            self.solvation['dielectric'].update(self.solvation['cavity'])

        finegd_energies = self.update_pseudo_potential(density)
        self.finegd.comm.sum(finegd_energies)

        # XXX/CB: Would need to use some additional code from real space
        # solvation hamiltonian to make this work for cavity/ia depending on
        # el density.

        if len(self.solvation['interactions']) > 0:
            for vt_g in self.vt_sg:
                # XXX/CB: linear operation is fine here, even if we go
                # to q
                vt_g += self.vt_ia_g
        Eias = np.array([ia.E for ia in self.solvation['interactions']])

        # XXX/CB: This is all well and good...but assumes that we hold density
        # in pw, which we should

        # XXX/CB: does gd parallelization still apply here?
        Ekin1 = self.gd.comm.sum_scalar(self.calculate_kinetic_energy(density))
        W_aL = self.calculate_atomic_hamiltonians(density)
        atomic_energies = self.update_corrections(density, W_aL)
        self.world.sum(atomic_energies)

        # XXX/CB: Should be fine here too, provided our superclass follows the
        # same implicit "interface" as real space Hamiltonian
        energies = atomic_energies
        energies[1:] += finegd_energies
        energies[0] += Ekin1

        if not kin_en_using_band:
            assert wfs is not None
            with self.timer('New Kinetic Energy'):
                energies[0] = \
                    self.calculate_kinetic_energy_directly(density, wfs)

        # XXX/CB: rest looks like an ok translation to me, no illegal Fourier
        # ops.
        (self.e_kinetic0, self.e_coulomb, self.e_zero,
         self.e_external, self.e_xc) = energies

        self.finegd.comm.sum(Eias)

        self.solvation['cavity'].communicate_vol_surf(self.world)
        for E, ia in zip(Eias, self.solvation['interactions']):
            setattr(self, 'e_' + ia.subscript, E)

        self.new_atoms = None
        self.timer.stop('Hamiltonian')

    def update_pseudo_potential(self, density):
        if self.solvation['cavity'].depends_on_el_density:
            raise NotImplementedError(
                'Current implementation would require an inverse' +
                'Fourier transform in the step that creates dependence')
        return ReciprocalSpaceHamiltonian.update_pseudo_potential(
            self, density)

    def calculate_forces(self, dens, F_av):
        # TODO/AAP: Note we may be able to punt on this temporarily; if we
        # only call calc.get_potential_energy() and not calc.get_forces() in
        # our initial scripts, until we get that working. Also note ASE has a
        # numerical forces (done by finite difference, since force is
        # derivative of - potential energy), that we may be able to use to
        # check our results.

        self.el_force_correction(dens, F_av)
        # XXX/REPLACE:
        # real_dens = self.real_space_conversion(dens.rhot_q)
        # self.el_force_correction(real_dens, F_av)
        for ia in self.solvation['interactions']:
            if self.solvation['cavity'].depends_on_atomic_positions:
                for a, F_v in enumerate(F_av):
                    # XXX/CB: Potentially fine here, but varying implementation
                    # in our new cavity/interaction objs.
                    del_g_del_r_vg = self.solvation['cavity'].get_del_r_vg(
                        a, dens)
                    # XXX/REPLACE:
                    # del_g_del_r_vg = self.solvation['cavity'].get_del_r_vg(
                    #     a, real_dens)
                    for v in (0, 1, 2):
                        # XXX/CB: Integration here probably has to remain real
                        # space...i.e. not obvious to me that pd3.integrate
                        # would be a valid solution in the new basis.
                        F_v[v] -= self.finegd.integrate(
                            ia.delta_E_delta_g_g * del_g_del_r_vg[v],
                            global_integral=False)
            if ia.depends_on_atomic_positions:
                for a, F_v in enumerate(F_av):
                    del_E_del_r_vg = ia.get_del_r_vg(a, dens)
                    # XXX/REPLACE:
                    # del_E_del_r_vg = ia.get_del_r_vg(a, real_dens)
                    for v in (0, 1, 2):
                        F_v[v] -= self.finegd.integrate(
                            del_E_del_r_vg[v],
                            global_integral=False)

        return ReciprocalSpaceHamiltonian.calculate_forces(self, dens, F_av)
        # XXX/REPLACE:
        # return ReciprocalSpaceHamiltonian.calculate_forces(self, dens,
        # self.pd3.fft(F_av))

    def el_force_correction(self, dens, F_av):
        # XXX/CB: We're definitely going to want to keep this in real space.
        # Only question will be what we are holding in reciprocal space that we
        # need to manage here, e.g. potentially vHt_q?

        # XXX/CB: density input in real space upstream
        if not self.solvation['cavity'].depends_on_atomic_positions:
            return
        del_eps_del_g_g = self.solvation['dielectric'].del_eps_del_g_g
        fixed = 1 / (8 * np.pi) * del_eps_del_g_g * \
            self.grad_squared(self.vHt_g)  # XXX grad_vHt_g inexact in bmgs
        # XXX/REPLACE:
        # self.grad_squared(self.real_space_conversion(self.vHt_q))
        for a, F_v in enumerate(F_av):
            del_g_del_r_vg = self.solvation['cavity'].get_del_r_vg(a, dens)
            for v in (0, 1, 2):
                F_v[v] += self.finegd.integrate(
                    fixed * del_g_del_r_vg[v],
                    global_integral=False)

    def get_energy(self, e_entropy, wfs, kin_en_using_band=True, e_sic=None):
        # FIXME/AAP: Code below is lightly modified from FD version.
        # It looks like it will remain the same except the first line?
        # I'm raising the error for now until we can verify.

        # XXX/CB: This is good as is. Pulling straight from Hamiltonian, just
        # as in real space. Probably a compromise made in initial pw
        # implementation.

        ReciprocalSpaceHamiltonian.get_energy(self, e_entropy, wfs,
                                              kin_en_using_band, e_sic)

        # The total energy calculated by the parent class includes the
        # solvent electrostatic contributions but not the interaction
        # energies. We add those here and store the electrostatic energies.
        self.e_el_free = self.e_total_free
        self.e_el_extrapolated = self.e_total_extrapolated
        for ia in self.solvation['interactions']:
            self.e_total_free += getattr(self, 'e_' + ia.subscript)
        # XXX/CB: Looks fine, ASSUMING wfs.etc is a scalar
        self.e_total_extrapolated = (self.e_total_free +
                                     wfs.occupations.extrapolate_factor *
                                     e_entropy)
        return self.e_total_free

    def summary(self, wfs, log):
        # FIXME/AAP: This looks like it will be the same as in the FD
        # version; I've put it in unchanged.

        # This is almost duplicate code to gpaw/hamiltonian's
        # Hamiltonian.summary, but with the cavity and interactions added.

        log('Energy contributions relative to reference atoms:',
            f'(reference = {self.setups.Eref * Ha:.6f})\n')

        energies = [('Kinetic:      ', self.e_kinetic),
                    ('Potential:    ', self.e_coulomb),
                    ('External:     ', self.e_external),
                    ('XC:           ', self.e_xc),
                    ('Entropy (-ST):', self.e_entropy),
                    ('Local:        ', self.e_zero)]

        if len(self.interactions) > 0:
            energies += [('Interactions', None)]
            for ia in self.interactions:
                energies += [(f' {ia.subscript:s}:',
                              getattr(self, 'e_' + ia.subscript))]

        for name, e in energies:
            if e is not None:
                log('%-14s %+11.6f' % (name, Ha * e))
            else:
                log('%-14s' % (name))

        log('--------------------------')
        log('Free energy:   %+11.6f' % (Ha * self.e_total_free))
        log('Extrapolated:  %+11.6f' % (Ha * self.e_total_extrapolated))
        log()
        self.xc.summary(log)

        try:
            workfunctions = self.get_workfunctions(wfs)
        except ValueError:
            pass
        else:
            log('Dipole-layer corrected work functions: {:.6f}, {:.6f} eV'
                .format(*np.array(workfunctions) * Ha))
            log()

        self.cavity.summary(log)
