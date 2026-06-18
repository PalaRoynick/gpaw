Extra exercise - Vibrational energy
===================================

The energy calculated with DFT is the electronic energy at 0 K; however, to
model catalytic reactions we are usually intereseted in the energy landscape
at finite temperature. In this exercise we will calculate the energy
contributions from vibrations and see how they affect the splitting of
:mol:`Ni_2` on Ru.

We calculate the vibrational energy in the harmonic approximation using the
finite displacement method. For further reading, see for example the following sources:

[1]: :doi:`Stoffel et al, Angewandte Chemie Int. Ed., 49, 5242-5266 (2010) <10.1002/anie.200906780>`

[2]: :doi:`Togo and Tanaka, Scripta Materialia 108, 1-5 (2015) <10.1016/j.scriptamat.2015.07.021>`

Vibrational energy of the initial and final states
--------------------------------------------------

a) Calculating vibrations requires tighter convergence than normal energy
   calculations. Therefore you should first take your already optimised initial
   and final state geometries from the NEB calculations and relax them further
   to `f_\mathrm{max}=0.01` eV/Å with the QuasiNewton optimiser and an energy cutoff of
   500 eV. Converge the eigenstates to 1e-8. (Note that for other systems you
   might need even tighter convergence!)

Submit the structures to the queue. The optimisation should take 10-15 mins
for each structure on 8 cores.

b) Once you have done this you can calculate the vibrations using the
   `vibrations module in ASE <https://ase-lib.org/ase/vibrations/vibrations.html>`__ following
   the template below. We only calculate the vibrations of the adsorbate and
   assume that the frequencies of the substrate are unchanged - this is a
   common assumption. Use 4 displacements to fit the frequencies and the same
   calculator parameters as in a).

Submit the calculations for the initial and the final state to the queue. It
will take a while to run, but you can start preparing your analysis (tasks c
and d) or read some of the references in the meantime.

You cannot run this code as is - use it as a starting point for your python script!

.. literalinclude:: vibrations_initial.py
   :start-after: snippet-vibrations-start
   :end-before: snippet-vibrations-end

The final three lines write out the frequencies and :code:`.traj` files with
animations of the different modes in order of their energy. Take a look at
the vibrational modes in the files. Do you understand why the first mode has
a low energy while the last one has a high energy?

c) Analyse the frequencies in the harmonic approximation:

.. code:: python

    from ase.thermochemistry import HarmonicThermo

    T = 300  # Kelvin
    energies = [0.01, 0.05, 0.10]  # An example only - insert your calculated energy in eV levels here
    vibs = HarmonicThermo(energies)
    Efree = vibs.get_helmholtz_energy(T, verbose=True)
    print('Free energy at 300 K: ', Efree)

The :code:`verbose` keyword gives a detailed description of the different
contributions to the free energy. For more information on what the different
contributions are, see the
`ASE background webpage <https://ase-lib.org/ase/thermochemistry/thermochemistry.html#harmonic-limit>`__.

Now try to calculate how the different contributions change with temperature.
You can for example make a :code:`for` loop and use the :code:`get_entropy()` and
:code:`get_internal_energy()` methods (see description
`here <https://ase-lib.org/ase/thermochemistry/thermochemistry.html#ase.thermochemistry.IdealGasThermo.get_enthalpy>`__).

d) Calculate how the vibrational energy affects the overall reaction energy.
   How does it change with temperature? Which contribution is important for the
   change in reaction energy?


e) To make sure that your NEB is converged you should also calculate the
   vibrational energy of the transition state. Again, this requires tighter
   convergence than we have used in the NEB exercise. This takes a while to run
   so to save time, we provide the transition state geometry from a reasonably
   converged NEB (i.e. `f_\mathrm{max}=0.01`, a cutoff energy of 800eV and 6x6
   k-points) in the file :code:`TS.xyz`. Calculate the vibrations with these
   parameters. How many imaginary modes do you get and how do they look? What
   does this mean?
