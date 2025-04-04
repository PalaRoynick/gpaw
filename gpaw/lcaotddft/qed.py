from __future__ import annotations

import warnings
from abc import ABC, abstractmethod

import numpy as np
from scipy import interpolate, special
from gpaw.tddft.spectrum import read_td_file_kicks
from ase.units import alpha, Hartree, Bohr
from gpaw.external import ConstantElectricField
from gpaw.lcaotddft.hamiltonian import KickHamiltonian
import matplotlib.pyplot as plt

krondelta = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])


def create_environment(environment='waveguide', **kwargs):
    if isinstance(environment, Environment):
        assert not kwargs, 'kwargs must not be given here'
        return environment

    if isinstance(environment, dict):
        assert not kwargs, 'please do not give kwargs together with dict'
        kwargs = dict(environment)
        environment = kwargs.pop('environment', 'waveguide')

    if environment == 'waveguide':
        return WaveguideEnvironment(**kwargs)
    if environment == 'vacuum':
        return VacuumEnvironment(**kwargs)
    if environment == 'cavity':
        return CavityEnvironment(**kwargs)
    if environment == 'dressedcavity':
        return DressedCavityEnvironment(**kwargs)

    raise ValueError(f'Unknown environment {environment}')


def is_invertible(a):
    return (a.shape[0] == a.shape[1] and
            np.linalg.matrix_rank(a) == a.shape[0])


def l_fit(x, y, cutfreq):
    from scipy.optimize import curve_fit

    cutupper = int(np.floor(len(x) / 2))
    xcut = x[:cutupper]
    cutlower = np.argmin((x[:cutupper] - cutfreq * 1.5)**2)
    xcut = xcut[cutlower:]
    ycut = y[cutlower:cutupper]

    def linear(xcut, a):
        return a * xcut

    popt, pcov = curve_fit(linear, xcut, ycut)
    y_subtracted = y - popt * x
    return [popt, y_subtracted]


def forward_finite_difference(coefficients: list[int],
                              data_tx: np.ndarray):
    """ Calculate derivatives by forward difference.

    The data corresponding to the last time is repeated several times,
    in order to obtain a result of the same shape as the original array.

    Parameters
    ----------
    coefficients
        List of coefficients.
        Check e.g. https://en.wikipedia.org/wiki/Finite_difference_coefficient
    data_tx
        Array of data, the first axis being the time axis.

    """
    N = len(data_tx)

    padded_shape = (N + len(coefficients) - 1, ) + data_tx.shape[1:]
    padded_data_tx = np.zeros(padded_shape, dtype=data_tx.dtype)
    padded_data_tx[:len(coefficients)] = data_tx[:1]
    padded_data_tx[len(coefficients):] = data_tx[1:]

    result_tx = np.zeros_like(data_tx)

    for start, coefficient in enumerate(coefficients):
        result_tx[:] += coefficient * padded_data_tx[start:start + N]

    return result_tx


def calculate_first_derivative(timestep, data_tx):
    """ Calculate the first derivative by first order forward difference.

    The data corresponding to the last time is repeated several times,
    in order to obtain a result of the same shape as the original array.

    Parameters
    ----------
    timestep
        Time step
    data_tx
        Array of data, the first axis being the time axis.

    Returns
    -------
    First derivative, array same shape as :attr:`data_tx`.
    """
    if len(data_tx) == 1 and timestep is None:
        return np.zeros_like(data_tx)

    # Coefficients from oldest times to newest times
    coefficients = [-1, 1]
    return forward_finite_difference(coefficients, data_tx) / timestep


def calculate_third_derivative(timestep, data_tx):
    """ Calculate the third derivative by first order forward difference.

    The data corresponding to the last time is repeated several times,
    in order to obtain a result of the same shape as the original array.

    Parameters
    ----------
    timestep
        Time step
    data_tx
        Array of data, the first axis being the time axis.

    Returns
    -------
    Third derivative, array same shape as :attr:`data_tx`.
    """
    if len(data_tx) == 1 and timestep is None:
        return np.zeros_like(data_tx)

    # Coefficients from oldest times to newest times
    coefficients = [-1, 3, -3, 1]
    # For a second order accuracy, the coefficients would be
    # [-5/2, 9, -12, 7, -3/2]
    return forward_finite_difference(coefficients, data_tx) / timestep ** 3


class RRemission:
    r"""
    Radiation-reaction potential according to Schaefer et al.
    [https://doi.org/10.1103/PhysRevLett.128.156402] and
    Schaefer [https://doi.org/10.48550/arXiv.2204.01602].
    The potential accounts for the friction
    forces acting on the radiating system of oscillating charges
    emitting into a single dimension. A more elegant
    formulation would use the current instead of the dipole.
    Please contact christian.schaefer.physics@gmail.com if any problems
    should appear or you would like to consider more complex systems.
    Big thanks to Tuomas Rossi and Jakub Fojt for their help.

    Parameters
    ----------
    environment
        Environment, or dictionary with environment parameters.

    Notes
    -----
    A legacy form for constructing the RRemission object is supported
    for backwards compatibility. In this form, two parameters are given.

        quantization_plane: float
            value of :math:`quantization_plane` in units of Å^2
        cavity_polarization: array
            value of :math:`cavity_polarization`; dimensionless (directional)
    """

    def __init__(self, *args):
        if len(args) == 1:
            self.environment = create_environment(args[0])
        elif len(args) == 2:
            # Legacy syntax
            self.environment = create_environment(
                'waveguide',
                quantization_plane=args[0],
                cavity_polarization=args[1])
            warnings.warn(
                "Use RRemission({'environment': 'waveguide', "
                "'quantization_plane': quantization_plane, "
                "'cavity_polarization': cavity_polarization})) instead of "
                'RRemission(quantization_plane, cavity_polarization).',
                FutureWarning)
        else:
            raise TypeError('Please provide only one argument '
                            '(two for legacy syntax)')

        # Recorded dipole moment over time
        # The entries all correspond to unique times
        # If there is a kick, the dipole after the kick is recorded
        # but not the dipole just before the kick
        self.dipole_tv = np.zeros((0, 3))
        # Time of last record
        self.time = 0

        # Time step in TDDFT
        self.timestep = None

    def read(self, reader):
        self.dipole_tv = reader.recorded_dipole / Bohr
        self.timestep = reader.timestep

    def write(self, writer):
        writer.write(recorded_dipole=self.dipole_tv * Bohr)
        writer.write(timestep=self.timestep)
        self.environment.write(writer)

    @classmethod
    def from_reader(cls, reader):
        parameters = reader.parameters.asdict()

        rremission = cls(parameters)
        rremission.read(reader)

        return rremission

    def initialize(self, paw):
        self.density = paw.density
        self.wfs = paw.wfs
        self.hamiltonian = paw.hamiltonian
        self.time = paw.time
        dipole_v = paw.density.calculate_dipole_moment()
        if len(self.dipole_tv) == 0:
            # If we are not reading from a restart file, this will be empty
            self.record_dipole(dipole_v)

        # Set up three external fields summing to the polarization cavity
        strength_v = self.environment.cavity_polarization * Hartree / Bohr
        assert len(strength_v) == 3
        ext_x = ConstantElectricField(strength_v[0], [1, 0, 0])
        ext_y = ConstantElectricField(strength_v[1], [0, 1, 0])
        ext_z = ConstantElectricField(strength_v[2], [0, 0, 1])
        ext_i = [ext_x, ext_y, ext_z]

        # Set up the dipole matrix
        get_matrix = self.wfs.eigensolver.calculate_hamiltonian_matrix
        self.V_iuMM = []
        for ext in ext_i:
            V_uMM = []
            hamiltonian = KickHamiltonian(self.hamiltonian, self.density, ext)
            for kpt in self.wfs.kpt_u:
                V_MM = get_matrix(hamiltonian, self.wfs, kpt,
                                  add_kinetic=False, root=-1)
                V_uMM.append(V_MM)
            self.V_iuMM.append(V_uMM)

    def record_dipole(self, dipole_v):
        """ Record the dipole for the next time step.

        Parameters
        ----------
        dipole_v
            The new dipole moment.
        """
        self.dipole_tv = np.vstack((self.dipole_tv, dipole_v))

    def update_dipole(self, time, dipole_v):
        """ Record the new dipole

        If the time has changed, record the new dipole in a new entry.

        If not, then it either means that we are iteratively
        performing propagator steps, or that a kick has been done.
        Either way, we overwrite the last entry in the record.

        Parameters
        ----------
        time
            The new time
        dipole_v
            The new dipole moment.
        """
        if time > self.time or len(self.dipole_tv) == 0:
            self.record_dipole(dipole_v)
            timestep = time - self.time
            if (self.timestep is not None and
                not np.isclose(self.timestep, timestep)):
                raise NotImplementedError(
                    'Variable time step in TDDFT not supported'
                    f'{self.timestep} != {timestep}')
            self.timestep = timestep
            self.time = time
        else:
            self.dipole_tv[-1] = dipole_v

    def get_MM(self, field_v, kpt):
        """ Get potential matrix.

        Parameters
        ----------
        field_v
            The radiation reaction field.
        kpt
            kpoint

        Returns
        -------
        The radiation reaction potential matrix.
        """
        kpt_rank, q = self.wfs.kd.get_rank_and_index(kpt.k)
        u = q * self.wfs.nspins + kpt.s

        Ni = len(self.V_iuMM)
        Vrr_MM = -field_v[0] * self.V_iuMM[0][u]
        for i in range(1, Ni):
            Vrr_MM -= field_v[i] * self.V_iuMM[i][u]

        return Vrr_MM

    def vradiationreaction(self, kpt, time):
        self.update_dipole(time, self.density.calculate_dipole_moment())

        # Set up the dyadic arrays
        self.environment.compute_dyadic_if_necessary(self.timestep,
                                                     self.maxtimesteps,
                                                     len(self.dipole_tv) - 1)

        field_v = self.environment.radiation_reaction_field(
            self.timestep, self.dipole_tv)

        Vrr_MM = self.get_MM(field_v, kpt)
        return Vrr_MM


class Environment(ABC):

    def write(self, writer):
        writer.child('parameters').write(**self.todict())

    @abstractmethod
    def radiation_reaction_field(self, timestep, dipole_tv):
        raise NotImplementedError

    @abstractmethod
    def todict(self):
        raise NotImplementedError

    def compute_dyadic_if_necessary(self, timestep, maxtimesteps, itert):
        pass


class WaveguideEnvironment(Environment):

    r""" The radiation reaction potential in a 1D waveguide is

    .. math::

        -\frac{4 \pi \alpha}{A}
        \boldsymbol{\varepsilon_c} \cdot \partial_t \boldsymbol{R}(t)
        \boldsymbol{\varepsilon_c} \cdot \hat{\boldsymbol{r}}

    Parameters
    ----------
    quantization_plane
        Quantization plane :math:`A` in units of Å^2.
    cavity_polarization
        The polarization of the cavity :math:`\boldsymbol{\varepsilon_c}`.
        Direction vector.
    """

    def __init__(self, quantization_plane, cavity_polarization):
        self.quantization_plane = quantization_plane / Bohr**2
        self.cavity_polarization = np.array(cavity_polarization)

    def radiation_reaction_field(self, timestep, dipole_tv):
        d1_dipole_v = calculate_first_derivative(timestep, dipole_tv)[-1]

        field_v = (4.0 * np.pi * alpha / self.quantization_plane
                   * (d1_dipole_v @ self.cavity_polarization)
                   * self.cavity_polarization)

        return field_v

    def todict(self):
        return {'environment': 'waveguide',
                'quantization_plane': self.quantization_plane * Bohr**2,
                'cavity_polarization': list(self.cavity_polarization)}


class VacuumEnvironment(Environment):

    r""" The radiation reaction potential in vacuum is

    .. math::

        -\frac{4 \alpha^3}{3} |\partial^3_t \boldsymbol{R}(t)
        \cdot \hat{\boldsymbol{r}}|
    """

    def radiation_reaction_field(self, timestep, dipole_tv):
        d1_dipole_v = calculate_first_derivative(timestep, dipole_tv)[-1]
        # 3D emission (factor 2 for correct WW-emission included)
        # currently the quantization_plane is overloaded with
        # the harmonic frequency [input in eV]
        # rr_amplify is an artificial amplification
        rr_amplify = 1e0
        if self.quantization_plane > 0:
            rr_argument_in = (((self.quantization_plane * Bohr**2)**2
                               / Hartree**2)
                              * np.square(d1_dipole_v)**0.5)
            rr_argument_in = np.array(3 * [rr_argument_in])
        else:
            d3_dipole_v = -calculate_third_derivative(timestep, dipole_tv)[-1]
            rr_argument_in = d3_dipole_v

        field_v = 4.0 * alpha**3 / 3.0 * rr_argument_in * rr_amplify

        return field_v

    def todict(self):
        return {'environment': 'vacuum'}


class CavityEnvironment(Environment):

    r"""

    Parameters
    ----------
    quantization_plane
        Quantization plane :math:`A` in units of Å^2.
    cavity_polarization
        The polarization of the cavity :math:`\boldsymbol{\varepsilon_c}`.
        Direction vector.
    resonances
        List of cavity resonances in units of eV.
    loss
        Loss of cavity in units of eV.
    ampl
        increase frequency resolution
     """

    def __init__(self,
                 quantization_plane,
                 cavity_polarization,
                 *,
                 resonance,
                 loss,
                 cutoff=None,
                 ampl=1):
        self.quantization_plane = quantization_plane / Bohr**2
        self.cavity_polarization = np.array(cavity_polarization)
        if cutoff is None:
            cutoff = resonance
        self.cavity_resonance = np.arange(resonance, cutoff + 1e-8, resonance
                                          ) / Hartree
        self.cavity_volume = (self.quantization_plane
                              * (np.pi / (alpha * self.cavity_resonance[0])))
        self.cavity_loss = loss / Hartree
        self.cutoff_lower = None
        self.cutofffrequency = None
        self.frequ_resolution_ampl = ampl

        self.dyadic = None

    def todict(self):
        return {'environment': 'cavity',
                'quantization_plane': self.quantization_plane * Bohr**2,
                'cavity_polarization': list(self.cavity_polarization),
                'resonance': self.cavity_resonance[0] * Hartree,
                'cutoff': self.cavity_resonance[-1] * Hartree,
                'loss': self.cavity_loss * Hartree,
                'ampl': self.frequ_resolution_ampl}

    def compute_dyadic(self, timestep, maxtimesteps, itert):
        self.dyadic = self.dyadicGt(timestep, maxtimesteps, itert=itert)

    def selffield(self,
                  timestep: float,
                  dipole_tv: np.ndarray):
        # While this is not the most efficient way to write the convolution,
        # it is not the bottleneck of the calculation. If the number of
        # time-steps is largely increased it would be reasonable to use
        # advanced integration strategies in order to shorten the time
        # for convolutions

        d1_dipole_tv = calculate_first_derivative(timestep, dipole_tv)
        d1_dipole_t = d1_dipole_tv @ self.cavity_polarization

        dyadic_t = self.dyadic[:len(dipole_tv) - 1]
        electric_rr_field = self.cavity_polarization * (
            np.dot(dyadic_t[::-1], d1_dipole_t[:-1])
            - 0.5 * dyadic_t[-1] * d1_dipole_t[0]
            - 0.5 * dyadic_t[0] * d1_dipole_t[-1])

        electric_rr_field *= 4. * np.pi * alpha**2 * timestep
        return -np.real(electric_rr_field)

    def radiation_reaction_field(self, timestep, dipole_tv):
        if len(dipole_tv) <= 1:
            return np.zeros(3)

        field_v = self.selffield(timestep, dipole_tv)

        return field_v

    def compute_dyadic_if_necessary(self, timestep, maxtimesteps, itert):
        if self.dyadic is None or len(self.dyadic) < maxtimesteps:
            # If dyadic does not exist, or is too short, then recalculate
            self.compute_dyadic(timestep, maxtimesteps, itert)

    def dyadicGt(self, timestep, maxtimesteps, *, itert):
        if itert > 0:
            self.frequ_resolution_ampl = (float(self.frequ_resolution_ampl) *
                                          float(itert) /
                                          float(maxtimesteps - 1))
        time_t = np.arange(0, timestep *
                           (maxtimesteps * self.frequ_resolution_ampl + 1),
                           timestep)
        # omega_w = 2 * np.pi * np.fft.fftfreq(len(time_t), timestep)
        G_w = 0
        envelope_t = np.exp(-self.cavity_loss * time_t)
        for omegares in self.cavity_resonance:
            g_t = envelope_t * np.sin(omegares * time_t) / omegares
            G_w += np.fft.fft(g_t)
        G_w *= 2. / self.cavity_volume / alpha**2
        print('USING BARE Gw0')

        Dt = - np.gradient(np.fft.ifft(G_w), timestep, axis=0)
        # For some reason the explicit derivative works best, version
        # below also possible but seems to have issues sometimes.
        # Notice that I moved a minus from down here to the sine G0
        # because the poles had been defined up side down
        # NOTE - I moved the sign back because there seem to be minor
        # numerical deviations otherwise -- check that again later
        # Dt[:,ii] = np.fft.ifft(1j * omegafft * Gw[:,ii].flatten())

        return Dt


class PrecomputedCavityEnvironment(Environment):

    r"""

    Parameters
    ----------
    quantization_plane
        Quantization plane :math:`A` in units of Å^2.
    cavity_polarization
        The polarization of the cavity :math:`\boldsymbol{\varepsilon_c}`.
        Direction vector.
    resonances
        List of cavity resonances in units of eV.
    loss
        Loss of cavity in units of eV.
    precomputedG
        (must be string or will be ignored)
        provide a file with the precomputed G(omega), omega > 0,
        the first column should be frequency and the 2rd, 3th and 4th
        should be xx, yy, zz component. This will be generalized
        at a later point.

        comment: If you provide a file, e.g. using the separate G_dyadic
                 tool, all remaining parameters will be ignored. Notice,
                 that quantization_plane will be used then as
                 artifical amplification, i.e., the internally used G
                 will be G = quantization_plane * G_in. This is done
                 to allow simple parameter scalings but naturally
                 deviates from the ab initio idea.
    ampl
        increase frequency resolution
     """

    def __init__(self,
                 quantization_plane,
                 cavity_polarization,
                 *,
                 resonance,
                 loss,
                 cutoff=None,
                 precomputedG=None,
                 ampl=1):
        self.quantization_plane = quantization_plane / Bohr**2
        self.cavity_polarization = np.array(cavity_polarization)
        if cutoff is None:
            cutoff = resonance
        self.cavity_resonance = np.arange(resonance, cutoff + 1e-8, resonance
                                          ) / Hartree
        self.cavity_volume = (self.quantization_plane
                              * (np.pi / (alpha * self.cavity_resonance[0])))
        self.cavity_loss = loss / Hartree
        self.cutoff_lower = None
        self.cutofffrequency = None
        self.precomputedG = precomputedG
        self.frequ_resolution_ampl = ampl

        self.dyadic = None
        self.dyadic_st = None

        self.Ggbamp = 1.
        self.precomputedG = None

    def todict(self):
        return {'environment': 'cavity',
                'quantization_plane': self.quantization_plane * Bohr**2,
                'cavity_polarization': list(self.cavity_polarization),
                'resonance': self.cavity_resonance[0] * Hartree,
                'cutoff': self.cavity_resonance[-1] * Hartree,
                'loss': self.cavity_loss * Hartree,
                'precomputedG': self.precomputedG,
                'ampl': self.frequ_resolution_ampl}

    def compute_dyadic(self, timestep, maxtimesteps, itert):
        [self.dyadic, self.dyadic_st] = self.dyadicGt(
            timestep, maxtimesteps, itert=itert)
        static = True
        if not static:
            self.dyadic_st = self.dyadic_st * 0
            print("You remove the static component from the dyadic.")

    def selffield(self,
                  timestep: float,
                  dipole_tv: np.ndarray):
        # While this is not the most efficient way to write the convolution,
        # it is not the bottleneck of the calculation. If the number of
        # time-steps is largely increased it would be reasonable to use
        # advanced integration strategies in order to shorten the time
        # for convolutions

        d1_dipole_tv = calculate_first_derivative(timestep, dipole_tv)

        electric_rr_field = np.zeros((3, ), dtype=complex)
        for ii in range(3):
            for jj in range(3):
                dyadic_t = self.dyadic[:len(dipole_tv) - 1, 3 * ii + jj]
                electric_rr_field[ii] += (
                    (4. * np.pi * alpha**2 * timestep *
                     (np.dot(dyadic_t[::-1], d1_dipole_tv[:-1, jj])
                      - 0.5 * dyadic_t[-1] * d1_dipole_tv[0, jj]
                      - 0.5 * dyadic_t[0] * d1_dipole_tv[-1, jj])))
                if np.sum(abs(self.dyadic_st)) != 0:
                    electric_rr_field[ii] += (
                        4. * np.pi * alpha**2
                        * self.dyadic_st[3 * ii + jj] *
                        dipole_tv[-1, jj])
        return -np.real(electric_rr_field)

    def radiation_reaction_field(self, timestep, dipole_tv):
        """
        rr_bg is the background radiation and uses the provided
        characteristic frequency self.cavity_resonance
        """
        rr_bg = (4.0 * (self.cavity_resonance * Bohr**2)**2
                 / Hartree**2 * alpha**3 / 3.0 * self.Ggbamp
                 * np.linalg.norm(self.d1_dipole_v))

        if len(dipole_tv) > 1:
            field_v = self.selffield(timestep, dipole_tv) + rr_bg
        else:
            field_v = np.zeros(3)

        return field_v

    def compute_dyadic_if_necessary(self, timestep, maxtimesteps, itert):
        if self.dyadic is None or len(self.dyadic) < maxtimesteps:
            # If dyadic does not exist, or is too short, then recalculate
            self.compute_dyadic(timestep, maxtimesteps, itert)

    def greens_tensor_frequency(self, timestep, maxtimesteps, *, itert):
        if itert > 0:
            self.frequ_resolution_ampl = (float(self.frequ_resolution_ampl) *
                                          float(itert) /
                                          float(maxtimesteps - 1))
        timeg = np.arange(0, timestep *
                          (maxtimesteps * self.frequ_resolution_ampl + 1),
                          timestep)
        omegafft = 2 * np.pi * np.fft.fftfreq(len(timeg), timestep)
        Gw0 = np.zeros((len(omegafft), 9), dtype=complex)
        Gst = np.zeros((9, ), dtype=complex)
        window_Gw0 = 1.
        window_Gw0_L = 1.
        G_clean = np.loadtxt(self.precomputedG + '_cleaned.dat',
                             dtype=complex)
        Gstin = np.loadtxt(self.precomputedG + '_static.dat',
                           dtype=complex)
        flip_xz_cav = True
        if flip_xz_cav is True:
            print("Careful, you are flipping x and z for G!")
            G_clean[:, [1, 9]] = G_clean[:, [9, 1]]
            Gstin[[1, 9]] = Gstin[[9, 1]]
        Gst = Gstin[1:]
        print("Currently, I ignore the static G when dressing with",
              "molecular polarizabilities. This is not quite",
              "consistent and should get corrected.")
        for ii in range(len(G_clean[0, 1:])):
            """
            The G that was computed for only some positive frequencies
            G_clean[:,0], is inverted for negative frequencies, attached
            in the np.fft format (0 to max frequency, min frequency to 0)
            and interpolate (GG_pj) to the used frequency resolution as
            specified by the propagation time and
            self.frequ_resolution_ampl. Note, that one has to rescale by
            the frequency-ratio between the oriG_cleanal and
            the interpolated, otherwise the norms are not preserved.
            Recall also that G is hermitian, i.e., G(-w*) = G*(w).
            """
            romin = G_clean[::-1, 0]
            revG = G_clean[::-1, ii + 1]
            omegafullin = np.real(np.append(G_clean[:, 0], -romin[:-1]))
            GG_pj = interpolate.interp1d(omegafullin,
                                         (np.append(G_clean[:, ii + 1],
                                          np.conjugate(revG[:-1]))),
                                         kind="cubic",
                                         fill_value=(0, 0),
                                         bounds_error=False)

            if len(G_clean[0, 1:]) == 3:
                shiftfac = 4
            elif len(G_clean[0, 1:]) == 9:
                shiftfac = 1
            else:
                print('ERROR: G_dyadic.dat is not properly formated')

            """
            The Sellmeier equation can provide a decent model for the
            refractive index over a given frequency domain.
            The window function defined below is adhoc but 'inspired'
            by the idea that high frequencies will not affect the
            valence dynamic of molecules. The function should be inspired
            by the provided G tensor.
            The factor len(omegafft) is supposed to compensate the
            normalization that will happen in the iFFT to get from
            the frequency domain back to time and D(t).
            """
            # self.cutoff_lower = 0.01
            self.cutofffrequency = 0.8
            if self.cutofffrequency is not None:
                window_Gw0 = 1 - (0.5 + 0.5 *
                                  special.erf((abs(omegafft) -
                                               self.cutofffrequency) * 25))
            if self.cutoff_lower is not None:
                window_Gw0_L = (0.5 + 0.5 *
                                special.erf((abs(omegafft) -
                                             self.cutoff_lower) * 250))
                print("CAREFUL, you are using window function(s) for G0w",
                      "and Xw (only upper) with a cutoff-energy [eV] ",
                      str(self.cutofffrequency * Hartree),
                      str(self.cutoff_lower * Hartree))

                print("CAREFUL, it could be that the cutoffs break",
                      "Kramers-Kroenig and lead to false imaginary",
                      "components in D.")

            Gw0[:, ii * shiftfac] = (GG_pj(omegafft) * window_Gw0 *
                                     window_Gw0_L *
                                     len(omegafft))

        return omegafft, Gw0, Gst

    def dyadicGt(self, timestep, maxtimesteps, *, itert):
        omegafft, Gw, Gst = self.greens_tensor_frequency(
            timestep, maxtimesteps, itert=itert)
        Dt = np.zeros_like(Gw)
        print('USING BARE Gw0')

        # We can move the sign to Dt later but the previous version
        # was missing a sign in g and for test-suit reasons I added this
        Gw = Gw * (-1)
        # if self.quantization_plane * Bohr**2 != 1.:
        #     Gw = Gw * self.quantization_plane * Bohr**2
        #     print("CAREFUL, you amplify the strength of G by a factor",
        #           "self.quantization_plane=",
        #           self.quantization_plane * Bohr**2)
        for ii in range(9):
            Dt[:, ii] = - \
                np.gradient(np.fft.ifft(Gw[:, ii].flatten()), timestep)
            # For some reason the explicit derivative works best, version
            # below also possible but seems to have issues sometimes.
            # Notice that I moved a minus from down here to the sine G0
            # because the poles had been defined up side down
            # NOTE - I moved the sign back because there seem to be minor
            # numerical deviations otherwise -- check that again later
            # Dt[:,ii] = np.fft.ifft(1j * omegafft * Gw[:,ii].flatten())

        return [Dt, Gst]


class DressedCavityEnvironment(CavityEnvironment):

    """
    Parameters
    ----------
    claussius
        using Claussius-Mossoti if 1, otherwise dilute limit.
    environmentens_in: array
        [0] number of ensemble oscillators,
        [1] Ve/V ratio of occupied ensemble to cavity volume,
        [2] resonance frequency of Drude-Lorentz for ensemble,
        [3] lossyness of ensemble oscillators,
        [4] plasma-frequency of ensemble oscillators
        comment: If [2], [3] or [4] equal 0, the code will attempt to load
                 3 dipole files to compute the polarizability matrix.
                 In that case [1] is ignored.
    """

    def __init__(self,
                 quantization_plane,
                 cavity_polarization,
                 resonance,
                 loss,
                 cutoff=None,
                 precomputedG=None,
                 ampl=1,
                 claussius=False,
                 environmentens_in=None):
        super().__init__(quantization_plane,
                         cavity_polarization,
                         resonance=resonance,
                         cutoff=cutoff,
                         loss=loss,
                         ampl=ampl)

        self.claussius = claussius
        assert environmentens_in is not None
        self.ensemble_number = environmentens_in[0]
        self.ensemble_occ_ratio = environmentens_in[1]
        self.ensemble_resonance = (environmentens_in[2]
                                   / Hartree)
        self.dmpval = environmentens_in[3]
        self.ensemble_loss = (environmentens_in[3]
                              * self.ensemble_resonance)
        self.ensemble_omegap = (environmentens_in[4]
                                / Hartree)

        self.dyadic = None
        self.dyadic_st = None

        self.Ggbamp = 1.
        self.precomputedG = None

    def todict(self):
        _dict = super().todict()
        if self.ensemble_resonance > 0:
            loss_out = self.ensemble_loss / self.ensemble_resonance
        else:
            loss_out = 0

        environmentens_in = [self.ensemble_number,
                             self.ensemble_occ_ratio,
                             self.ensemble_resonance,
                             loss_out,
                             self.ensemble_omegap * Hartree]
        _dict['environmentens_in'] = environmentens_in
        _dict['environment'] = 'dressedcavity'
        _dict['precomputedG'] = self.precomputedG
        _dict['claussius'] = self.claussius
        return _dict

    def compute_dyadic(self, timestep, maxtimesteps, itert):
        self.dyadic, self.dyadic_st = self.dyadicGt(
                timestep, maxtimesteps, itert=itert)

    def selffield(self,
                  timestep: float,
                  dipole_tv: np.ndarray):
        # While this is not the most efficient way to write the convolution,
        # it is not the bottleneck of the calculation. If the number of
        # time-steps is largely increased it would be reasonable to use
        # advanced integration strategies in order to shorten the time
        # for convolutions

        d1_dipole_tv = calculate_first_derivative(timestep, dipole_tv)

        electric_rr_field = np.zeros((3, ), dtype=complex)
        for ii in range(3):
            for jj in range(3):
                dyadic_t = self.dyadic[:len(dipole_tv) - 1, 3 * ii + jj]
                electric_rr_field[ii] += (
                    (4. * np.pi * alpha**2 * timestep *
                     (np.dot(dyadic_t[::-1], d1_dipole_tv[:-1, jj])
                      - 0.5 * dyadic_t[-1] * d1_dipole_tv[0, jj]
                      - 0.5 * dyadic_t[0] * d1_dipole_tv[-1, jj])))
                if np.sum(abs(self.dyadic_st)) != 0:
                    electric_rr_field[ii] += (
                        4. * np.pi * alpha**2
                        * self.dyadic_st[3 * ii + jj] *
                        dipole_tv[-1, jj])
        return -np.real(electric_rr_field)

    def greens_tensor_frequency(self, timestep, maxtimesteps, *, itert):
        if itert > 0:
            self.frequ_resolution_ampl = (float(self.frequ_resolution_ampl) *
                                          float(itert) /
                                          float(maxtimesteps - 1))
        timeg = np.arange(0, timestep *
                          (maxtimesteps * self.frequ_resolution_ampl + 1),
                          timestep)
        omegafft = 2 * np.pi * np.fft.fftfreq(len(timeg), timestep)
        Gw0 = np.zeros((len(omegafft), 9), dtype=complex)
        Gst = np.zeros((9, ), dtype=complex)
        window_Gw0 = 1.
        window_Gw0_L = 1.
        if self.precomputedG is None:
            g_omega = 0
            for omegares in self.cavity_resonance:
                g_omega += (2. / self.cavity_volume / alpha**2 *
                            np.fft.fft(np.exp(-self.cavity_loss * timeg)
                                       * np.sin(omegares * timeg) / omegares))
            for ii in range(3):
                for jj in range(3):
                    Gw0[:, 3 * ii + jj] = (g_omega *
                                           self.cavity_polarization[ii] *
                                           self.cavity_polarization[jj])
        else:
            G_clean = np.loadtxt(self.precomputedG + '_cleaned.dat',
                                 dtype=complex)
            Gstin = np.loadtxt(self.precomputedG + '_static.dat',
                               dtype=complex)
            flip_xz_cav = True
            if flip_xz_cav is True:
                print("Careful, you are flipping x and z for G!")
                G_clean[:, [1, 9]] = G_clean[:, [9, 1]]
                Gstin[[1, 9]] = Gstin[[9, 1]]
            Gst = Gstin[1:]
            print("Currently, I ignore the static G when dressing with",
                  "molecular polarizabilities. This is not quite",
                  "consistent and should get corrected.")
            for ii in range(len(G_clean[0, 1:])):
                """
                The G that was computed for only some positive frequencies
                G_clean[:,0], is inverted for negative frequencies, attached
                in the np.fft format (0 to max frequency, min frequency to 0)
                and interpolate (GG_pj) to the used frequency resolution as
                specified by the propagation time and
                self.frequ_resolution_ampl. Note, that one has to rescale by
                the frequency-ratio between the oriG_cleanal and
                the interpolated, otherwise the norms are not preserved.
                Recall also that G is hermitian, i.e., G(-w*) = G*(w).
                """
                romin = G_clean[::-1, 0]
                revG = G_clean[::-1, ii + 1]
                omegafullin = np.real(np.append(G_clean[:, 0], -romin[:-1]))
                GG_pj = interpolate.interp1d(omegafullin,
                                             (np.append(G_clean[:, ii + 1],
                                              np.conjugate(revG[:-1]))),
                                             kind="cubic",
                                             fill_value=(0, 0),
                                             bounds_error=False)

                if len(G_clean[0, 1:]) == 3:
                    shiftfac = 4
                elif len(G_clean[0, 1:]) == 9:
                    shiftfac = 1
                else:
                    print('ERROR: G_dyadic.dat is not properly formated')

                """
                The Sellmeier equation can provide a decent model for the
                refractive index over a given frequency domain.
                The window function defined below is adhoc but 'inspired'
                by the idea that high frequencies will not affect the
                valence dynamic of molecules. The function should be inspired
                by the provided G tensor.
                The factor len(omegafft) is supposed to compensate the
                normalization that will happen in the iFFT to get from
                the frequency domain back to time and D(t).
                """
                # self.cutoff_lower = 0.01
                self.cutofffrequency = 0.8
                if self.cutofffrequency is not None:
                    window_Gw0 = 1 - (0.5 + 0.5 *
                                      special.erf((abs(omegafft) -
                                                   self.cutofffrequency) * 25))
                if self.cutoff_lower is not None:
                    window_Gw0_L = (0.5 + 0.5 *
                                    special.erf((abs(omegafft) -
                                                 self.cutoff_lower) * 250))
                    print("CAREFUL, you are using window function(s) for G0w",
                          "and Xw (only upper) with a cutoff-energy [eV] ",
                          str(self.cutofffrequency * Hartree),
                          str(self.cutoff_lower * Hartree))

                    print("CAREFUL, it could be that the cutoffs break",
                          "Kramers-Kroenig and lead to false imaginary",
                          "components in D.")

                Gw0[:, ii * shiftfac] = (GG_pj(omegafft) * window_Gw0 *
                                         window_Gw0_L *
                                         len(omegafft))
        return omegafft, Gw0, Gst, window_Gw0, g_omega

    def dyadicGt(self, timestep, maxtimesteps, print_output=False, *, itert):
        omegafft, Gw, Gst, window_Gw0, g_omega = self.greens_tensor_frequency(
            timestep, maxtimesteps, itert=itert)

        Gw = np.zeros((len(omegafft), 9), dtype=complex)
        Xw = np.zeros((len(omegafft), 9), dtype=complex)
        Dt = np.zeros_like(Gw)

        if (self.ensemble_loss == 0 or
            self.ensemble_omegap == 0 or
            self.ensemble_resonance == 0):
            print('Ab initio embedding using dm_ensemblebare_{x/y/z}.dat')
            kickx = read_td_file_kicks('dm_ensemblebare_x.dat')
            kickstrengthx = np.sqrt(np.sum(kickx[0]['strength_v']**2))
            kicky = read_td_file_kicks('dm_ensemblebare_y.dat')
            kickstrengthy = np.sqrt(np.sum(kicky[0]['strength_v']**2))
            kickz = read_td_file_kicks('dm_ensemblebare_z.dat')
            kickstrengthz = np.sqrt(np.sum(kickz[0]['strength_v']**2))
            impdip_x = np.loadtxt('dm_ensemblebare_x.dat', skiprows=5)
            impdip_y = np.loadtxt('dm_ensemblebare_y.dat', skiprows=5)
            impdip_z = np.loadtxt('dm_ensemblebare_z.dat', skiprows=5)
            timedm = impdip_z[:, 0]
            dm_x = ((impdip_x[:, 2:] - impdip_x[0, 2:]) /
                    (2. * np.pi * kickstrengthx))
            dm_y = ((impdip_y[:, 2:] - impdip_y[0, 2:]) /
                    (2. * np.pi * kickstrengthy))
            dm_z = ((impdip_z[:, 2:] - impdip_z[0, 2:]) /
                    (2. * np.pi * kickstrengthz))
            omegafftdm = 2. * np.pi * np.fft.fftfreq(len(timedm),
                                                     timedm[1] - timedm[0])
            # One could move the calculation of the polarizability matrix
            # into some tool and let the user specify directly the location
            # of that matrix. This would make it necessary to have a tool
            # for alpha_ij and the user would need to run it. However, this
            # would also allow to get alpha_ij from another code.
            # Maybe the best option would be to provide both input
            # alternatives, careful with the FFT norms.

            lastsize = np.sum(np.abs(dm_x[-1, :]) + np.abs(dm_y[-1, :]) +
                              np.abs(dm_z[-1, :]))
            dmp = np.ones(len(dm_x[:, 0]))
            if lastsize > self.dmpval and self.dmpval != 0:
                print("Adding additional damping to smoothen Xw to ",
                      self.dmpval)
                decayrate = -np.log(self.dmpval) / len(dm_x[:, 0])
                dmp = np.exp(-decayrate * np.arange(len(dm_x[:, 0])))

            pol_matrix = np.array([np.fft.fft(dm_x[:, 0] * dmp),
                                   np.fft.fft(dm_x[:, 1] * dmp),
                                   np.fft.fft(dm_x[:, 2] * dmp),
                                   np.fft.fft(dm_y[:, 0] * dmp),
                                   np.fft.fft(dm_y[:, 1] * dmp),
                                   np.fft.fft(dm_y[:, 2] * dmp),
                                   np.fft.fft(dm_z[:, 0] * dmp),
                                   np.fft.fft(dm_z[:, 1] * dmp),
                                   np.fft.fft(dm_z[:, 2] * dmp)],
                                  dtype=complex)

            print('Embedding RR, particle density is ',
                  self.ensemble_number / self.cavity_volume,
                  'with particle number',
                  self.ensemble_number,
                  'and volume of',
                  self.cavity_volume)
            # alpha_ij is the polarizablity (local response)
            alpha_ij = np.zeros((len(omegafft), 9), dtype=complex)
            for ii in range(9):
                pol_interp = interpolate.interp1d(omegafftdm,
                                                  pol_matrix[ii, :],
                                                  kind="cubic",
                                                  fill_value="extrapolate")
                alpha_ij[:, ii] = (pol_interp(omegafft)
                                   * timestep / (timedm[1] - timedm[0])
                                   * window_Gw0)
            print("Increase in frequency resolution",
                  len(omegafft) / len(omegafftdm))

            if self.claussius == 1:
                print('Using Claussius-Mossotti for Xw')
                skippedCM = False
                for el in range(len(omegafft)):
                    if is_invertible(np.reshape(alpha_ij[el, :], (3, 3))):
                        # Xw[el, :] = np.reshape((4. * np.pi * self.ensemble_number / self.cavity_volume) * np.reshape(alpha_ij[el, :], (3, 3)) @ (np.linalg.inv(np.eye(3)) - 1. / 3. * (4. * np.pi * self.ensemble_number / self.cavity_volume) * np.reshape(alpha_ij[el, :], (3, 3))) , (-1, ))
                        Xw[el, :] = np.reshape(np.linalg.inv(np.linalg.inv(np.reshape(alpha_ij[el, :], (3, 3))) / (
                            4. * np.pi * self.ensemble_number / self.cavity_volume) - 1. / 3. * np.eye(3)), (-1, ))
                    else:
                        Xw[el, :] = 4. * np.pi * self.ensemble_number / \
                            self.cavity_volume * alpha_ij[el, :]
                        skippedCM = True
                if skippedCM:
                    print("Polarizability was not invertible at some point! This might be due to the window-function, consider REMOVING the window from alpha_ij -- CHRISTIAN!")

            if print_output:
                for ii in range(9):
                    plt.figure()
                    if self.claussius == 1:
                        plt.plot(omegafft * Hartree, np.real(Xw[:, ii]))
                        plt.plot(omegafft * Hartree, np.imag(Xw[:, ii]))
                    else:
                        plt.plot(omegafft * Hartree,
                                 (4. * np.pi
                                  * self.ensemble_number
                                  / self.cavity_volume
                                  * np.real(alpha_ij[:, ii])))
                        plt.plot(omegafft * Hartree,
                                 (4. * np.pi
                                  * self.ensemble_number
                                  / self.cavity_volume
                                  * np.imag(alpha_ij[:, ii])))
                    plt.xlabel("Energy (eV)")
                    plt.ylabel(r"$\chi(\omega)$, i=" + str(ii)
                               + 'CM=' + str(self.claussius))
                    if self.cutofffrequency is not None:
                        plt.xlim(0, self.cutofffrequency * 1.5 * Hartree)
                    else:
                        plt.xlim(0, 14)
                    plt.savefig('Xw_' + str(ii) + '.png')
                    plt.close()
        else:
            print('Drude-Lorentz model for polarizablity')
            for ii in range(3):
                alpha_ij[:, ii * 4] = (self.ensemble_omegap**2
                                       / (self.ensemble_resonance**2 -
                                          omegafft**2 + (1j * self.ensemble_loss
                                                         * omegafft)))
        if self.precomputedG is None:
            print("Dressing G for simplified cavity")
            for ii in range(3):
                for jj in range(3):
                    Gw[:, 3 * ii + jj] = (
                        self.cavity_polarization[ii] *
                        self.cavity_polarization[jj] *
                        np.reciprocal(
                            np.reciprocal(g_omega) -
                            (
                                alpha ** 2 * omegafft ** 2 * 4 * np.pi *
                                self.ensemble_number * alpha_ij[:, 3 * ii + jj] *
                                self.cavity_polarization[ii] *
                                self.cavity_polarization[jj]
                            )
                        )
                    )
            if self.claussius == 1:
                Gwibar = np.zeros((len(omegafft), 9), dtype=complex)
                for ii in range(3):
                    for jj in range(3):
                        Gwibar[:, 3 * ii + jj] = (np.reciprocal(g_omega) *
                                                  self.cavity_polarization[ii] *
                                                  self.cavity_polarization[jj])
                for el in range(len(omegafft)):
                    if not is_invertible(np.reshape(Gwibar[el, :], (3, 3)) - (4 * np.pi * alpha**2 * omegafft[el]**2 * self.ensemble_number * np.reshape(alpha_ij[el, :], (3, 3)))):
                        shift = 1e-8
                    else:
                        shift = 0
                    Gw[el, :] = np.reshape((np.linalg.inv((np.linalg.inv(np.eye(3) + 1. / 3. * np.reshape(Xw[el, :], (3, 3))) @
                                                           (np.reshape(Gwibar[el, :] + shift, (3, 3)) @ (np.eye(3) - 1. / 3. * np.reshape(Xw[el, :], (3, 3)))))
                                                          - (4 * np.pi * alpha**2 * omegafft[el]**2 * self.ensemble_number * np.reshape(alpha_ij[el, :], (3, 3))))
                                            @ np.linalg.inv(np.eye(3) + 1. / 3. * np.reshape(Xw[el, :], (3, 3)))
                                            @ (np.eye(3) + 1. / 3. * np.reshape(Xw[el, :], (3, 3)) @ np.linalg.inv(np.eye(3) + np.reshape(Xw[el, :], (3, 3))))), (-1, ))
                    # Gw[el, :] = np.reshape(np.linalg.inv(np.eye(3) - 1. / 3. * np.reshape(Xw[el, :], (3, 3))) @
                    #                        np.reshape(Gw[el, :], (3, 3)) @ (np.eye(3) + 1. / 3. * np.linalg.inv(np.eye(3) + np.reshape(Xw[el, :], (3, 3))) @ np.reshape(Xw[el, :], (3, 3)) )
                    #                        ,(-1, ))
            elif self.claussius == -1:
                Gwibar = np.zeros((len(omegafft), 9), dtype=complex)
                for ii in range(3):
                    for jj in range(3):
                        Gwibar[:, 3 * ii + jj] = (np.reciprocal(g_omega) *
                                                  self.cavity_polarization[ii] *
                                                  self.cavity_polarization[jj])
                for el in range(len(omegafft)):
                    Xw[el, :] = 4. * np.pi * self.ensemble_number / \
                        self.cavity_volume * alpha_ij[el, :]
                    if not is_invertible(np.reshape(Gwibar[el, :], (3, 3)) - (alpha**2 * omegafft[el]**2 * self.cavity_volume * np.linalg.inv(np.eye(3) + 1. / 3. * np.reshape(Xw[el, :], (3, 3))) @ np.reshape(Xw[el, :], (3, 3)))):
                        shift = 1e-8
                    else:
                        shift = 0
                    Gw[el, :] = np.reshape((np.linalg.inv((np.linalg.inv(np.eye(3) + 1. / 3. * np.reshape(Xw[el, :], (3, 3))) @
                                                           (np.reshape(Gwibar[el, :] + shift, (3, 3)) @ (np.eye(3) - 1. / 3. * np.reshape(Xw[el, :], (3, 3)))))
                                                          - (alpha**2 * omegafft[el]**2 * self.cavity_volume *
                                                              np.linalg.inv(np.eye(3) + 1. / 3. * np.reshape(Xw[el, :], (3, 3))) @ np.reshape(Xw[el, :], (3, 3))))
                                            @ np.linalg.inv(np.eye(3) + 1. / 3. * np.reshape(Xw[el, :], (3, 3)))
                                            @ (np.eye(3) + 1. / 3. * np.reshape(Xw[el, :], (3, 3)) @ np.linalg.inv(np.eye(3) + np.reshape(Xw[el, :], (3, 3))))), (-1, ))
        else:
            print("Dressing G using the provided Gw0,",
                  "this may take a few minutes.")
            bg_correction = False
            if not bg_correction:
                print(
                    "Using new version with local field correction but ignoring free-space dressing.")
            if bg_correction:
                G_freespace = np.zeros((len(omegafft), 9), dtype=complex)
                Gbg = np.zeros((len(omegafft), 9), dtype=complex)
                for ii in range(9):
                    G_freespace[:, ii] = (1j * omegafft * alpha /
                                          (6 * np.pi) * krondelta[ii])
                Gw0 = Gw0 + G_freespace * len(omegafft)
                """
                The factor len(omegafft) is again added to compensate the
                normalization via the FFT.
                """
            for el in range(len(omegafft)):
                """
                For each frequency, reshape Gw0 and Xw into matrix and
                build Gw via taking the inverse. Notice, that for Xw=0,
                Gw=Gw0 and the singular case is also set to Gw=Gw0.
                """
                if self.claussius == 1:
                    if is_invertible(np.reshape(Gw0[el, :], (3, 3))):
                        Gw[el, :] = np.reshape((np.linalg.inv((np.linalg.inv(np.eye(3) + 1. / 3. * np.reshape(Xw[el, :], (3, 3))) @
                                                               (np.linalg.inv(np.reshape(Gw0[el, :], (3, 3))) @ (np.eye(3) - 1. / 3. * np.reshape(Xw[el, :], (3, 3)))))
                                                              - (4 * np.pi * alpha**2 * omegafft[el]**2 * self.ensemble_number * np.reshape(alpha_ij[el, :], (3, 3))))
                                                @ np.linalg.inv(np.eye(3) + 1. / 3. * np.reshape(Xw[el, :], (3, 3)))
                                                @ (np.eye(3) + 1. / 3. * np.reshape(Xw[el, :], (3, 3)) @ np.linalg.inv(np.eye(3) + np.reshape(Xw[el, :], (3, 3))))), (-1, ))
                        # Gw[el, :] = np.reshape(np.linalg.inv(np.eye(3) - 1. / 3. * np.reshape(Xw[el, :], (3, 3))) @
                        #                        (np.linalg.inv(np.linalg.inv(np.reshape(Gw0[el, :], (3, 3))) - (4 * np.pi * alpha**2 * omegafft[el]**2 * self.ensemble_number * np.reshape(alpha_ij[el, :], (3, 3)))) @
                        #                         (np.eye(3) + 1. / 3. * np.linalg.inv(np.eye(3) + np.reshape(Xw[el, :], (3, 3))) @ np.reshape(Xw[el, :], (3, 3)) )
                        #                         )
                        #                        ,(-1, ))
                    else:
                        Gw[el, :] = Gw0[el, :]
                else:
                    if is_invertible(np.reshape(Gw0[el, :], (3, 3))):
                        Gw[el, :] = np.reshape(
                            np.linalg.inv(
                                np.linalg.inv(
                                    np.reshape(Gw0[el, :], (3, 3))
                                ) - (alpha**2 * omegafft[el]**2 *
                                     self.ensemble_number * 4 * np.pi *
                                     np.reshape(alpha_ij[el, :], (3, 3)))
                            ),
                            (-1, )
                        )
                    else:
                        Gw[el, :] = Gw0[el, :]
            if bg_correction and self.cutofffrequency is not None:
                """
                    -- adding local field correction
                                   derived by Frieder
                print("Setting up Gbar0 correction -- only ISO yet")
                iso_eps = 1 + (Xw[:, 0] + Xw[:, 4] + Xw[:, 8])/3
                for ii in range(3):
                    for jj in range(3):
                        Gbar0[:,ii,jj] = (
                            krondelta[ii,jj] *
                            1/(3.* V_e * iso_eps * omegafft ** 2)
                --- missing: define V_e somewhere
                """
                for ii in range(3):
                    [Gbg_re, Gout_re] = l_fit(omegafft,
                                              np.real(Gw[:, 4 * ii]),
                                              self.cutofffrequency)
                    [Gbg_im, Gout_im] = l_fit(omegafft,
                                              np.imag(Gw[:, 4 * ii]),
                                              self.cutofffrequency)
                    if (np.abs(Gbg_im / len(omegafft)
                        / (alpha / (6 * np.pi)) - 1) > 0.1):
                        Gbg[:, 4 * ii] = 1j * Gbg_im * omegafft
                        self.Ggbamp = (Gbg_im / len(omegafft) /
                                       (alpha / (6 * np.pi)))
                        print('background emission amplification: ',
                              self.Ggbamp)
                    else:
                        Gbg[:, 4 * ii] = (1j * alpha / (6 * np.pi) *
                                          omegafft * len(omegafft))
                        self.Ggbamp = 1.
                    if np.abs(Gbg_re) > 1e6:
                        print("# WARNING: The linear part of G has a",
                              "sizeable real-part, which is not take into",
                              "account Gbg_re = ", str(Gbg_re))
                Gw = Gw - Gbg
                Gw0 = Gw0 - Gbg
            elif bg_correction and self.cutofffrequency is None:
                Gw = Gw - G_freespace
                Gw0 = Gw0 - G_freespace

        if self.precomputedG is not None:
            # We can move the sign to Dt later but the previous version
            # was missing a sign in g and for test-suit reasons I added this
            Gw = Gw * (-1)
            Gw0 = Gw0 * (-1)
            # if self.quantization_plane * Bohr**2 != 1.:
            #     Gw = Gw * self.quantization_plane * Bohr**2
            #     print("CAREFUL, you amplify the strength of G by a factor",
            #           "self.quantization_plane=",
            #           self.quantization_plane * Bohr**2)
        for ii in range(9):
            Dt[:, ii] = - \
                np.gradient(np.fft.ifft(Gw[:, ii].flatten()), timestep)
            # For some reason the explicit derivative works best, version
            # below also possible but seems to have issues sometimes.
            # Notice that I moved a minus from down here to the sine G0
            # because the poles had been defined up side down
            # NOTE - I moved the sign back because there seem to be minor
            # numerical deviations otherwise -- check that again later
            # Dt[:,ii] = np.fft.ifft(1j * omegafft * Gw[:,ii].flatten())
            if print_output:
                if ii == 8:
                    plt.figure()
                    plt.plot(omegafft * Hartree, np.real(Gw[:, ii]), 'k-',
                             label='Real Gw-scatter Element: ' + str(ii))
                    plt.plot(omegafft * Hartree, np.imag(Gw[:, ii]), 'r-',
                             label='Imag Gw-scatter Element: ' + str(ii))
                    plt.plot(omegafft * Hartree, np.real(Gw0[:, ii]), 'k:',
                             label='Real Gw0 Element: ' + str(ii))
                    plt.plot(omegafft * Hartree, np.imag(Gw0[:, ii]), 'r:',
                             label='Imag Gw0 Element: ' + str(ii))
                    plt.xlabel("Energy (eV)")
                    plt.ylabel(r"$G^{(1),no static}_i(\omega)$, i=" + str(ii))
                    if self.cutofffrequency is not None:
                        plt.xlim(0, self.cutofffrequency * 1.5 * Hartree)
                    else:
                        plt.xlim(0, 14)
                    plt.legend(loc="upper right")
                    plt.savefig('Gw_' + str(ii) + '.png')
                    plt.figure()
                    plt.plot(range(maxtimesteps), np.real(Dt[:maxtimesteps, ii]),
                             label='Real Dt Element: ' + str(ii))
                    plt.plot(range(maxtimesteps), np.imag(Dt[:maxtimesteps, ii]),
                             label='Imag Dt Element: ' + str(ii))
                    plt.xlabel("time step")
                    plt.ylabel(r"$D^{(1),no static}_i(t)$, i=" + str(ii))
                    plt.legend(loc="upper right")
                    plt.savefig('Dt_' + str(ii) + '.png')
                if self.ensemble_number is not None:
                    np.savez("Xw",
                             energy=omegafft[:int(
                                 len(omegafft) / 2)] * Hartree,
                             alpha_ij=alpha_ij[:int(len(omegafft) / 2), :],
                             Xw=Xw[:int(len(omegafft) / 2), :])
                    np.savez("dyadicD",
                             Dt=Dt[:int(len(omegafft) / 2), :],
                             energy=omegafft[:int(
                                 len(omegafft) / 2)] * Hartree,
                             Gw=Gw[:int(len(omegafft) / 2), :],
                             Gst=Gst)
        return [Dt, Gst]
