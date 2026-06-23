.. _magnetism1:

============================================
Part 1: Critical temperature of :mol:`CrI_3`
============================================

In 2017, ferromagnetic order was observed in a monolayer of :mol:`CrI_3` below 45 `K` [*Nature* **546** 270 (2017)]. It comprises the first demonstration of magnetic order in a 2D material and has received a lot of attention due to the peculiar properties of magnetism in 2D. The physics of magnetic order in 2D is rather different than in 3D and in order to understand what is going on we will need to introduce a bit of theory. But before we get to that let us get started with the calculations.


In this turorial you will set up and relax a monolayer of
:mol:`CrI_3` after which you will calculate its exchange contants and critical
temperature using different models

The procedure will be as follows:

1) Set up the atomic structure and optimize the geometry of :mol:`CrI_3`
2) Calculate the nearest neighbor Heisenberg exchange coupling based on a total
   energy mapping analysis
3) Show that the magnetic ground state is thermodynamically unstable when
   anisotropy is neglected (The Mermin-Wagner theorem)
4) Calculate the single-ion magnetic anisotropy and estimate the critical
   temperature



DFT calculation - finding the atomic structure of :mol:`CrI_3`
==============================================================

We start by setting up the atomic structure of a :mol:`CrI_3` monolayer and optimizing the atomic positions and unit cell. There are two formula units in the minimal unit cell and only the Cr atoms bear significant magnetic moments. A spin-polarized calculation is initiated by specifying the initial magnetic moments of the all the atoms in units of `\mu_B`.

1. What do you expect the ionic value for the magnetic moment of the Cr atoms to be? (Hint: Use Hund's rule. The electronic configuration of a Cr atom is [Ar]3d`^5`4s`^1` and each of the iodine atoms will steal one electron). Use your answer to fill in the spin state `S` in the code below.

Try to understand the individual lines in the code below, copy it to a file and run the script. The calculation will open the ase gui to show the initial atomic structure. You may, for example, try to repeat the structure (under view) to get a better feeling for the material. Then look at the text output from the calculation and answer the following questions:

2.  How many electrons are used in the calculation? Which valence states of Cr and I are included?
3.  What is the total magnetic moment after the first DFT calculation and on which atoms are the magnetic moments located?
4.  What is the number of irreducible k-points (k-points not related by symmetry) in the calculation?
5.  What is the maximum force on the atoms after the first DFT calculation? Does it become smaller after subsequent calculations?

In order to get the script running fast, we have set a few of the computational parameters to values which are expected to produce a somewhat inaccurate result. Can you identify what parameters one would need to converge or modify in order to produce more accurate results?

Leave the script running and continue with the theory section below.

.. literalinclude:: CrI3_gs_student.py


A bit of theory
===============

The Heisenberg model
--------------------

If we want to calculate the Curie temperature of :mol:`CrI_3` it is, in principle, clear what we have to do. We should calculate the magnetization as a function of temperature using standard methods from statistical physics, and record the temperature where the magnetization vanishes. Unfortunately, this requires knowledge of all the excited states of the system, which we do not have access to. In particular, the magnetization will be dominated by collective magnetic exciations, and these are not directly accessible from the Kohn-Sham spectrum produced by our DFT calculations.

Instead we will consider the Heisenberg Hamiltonian, which captures the basic physics of typical spin systems. It is given by

.. math::

  H = -\frac{1}{2}\sum_{i,j}J_{ij}\mathbf{S}_i\cdot \mathbf{S}_j,

where `\mathbf{S}_i` denotes the spin operator at site `i` in units of `\hbar` and `J_{ij}` are magnetic exchange coupling constants. If we want to model a real material with the Heisenberg model we then need to identify a set of magnetic sites and calculate the exchange coupling constants `J_{ij}`.

1.   What are the magnetic sites of :mol:`CrI_3`?
2.   How many nearest neighbors does each magnetic site have?
3.   What are the possible values of `S_i^z` for a magnetic site in :mol:`CrI_3`?
4.   What is the unit of `J_{ij}`?

In the following, we will assume that the physics is dominated by neareast neighbor interactions such that `J_{ij}\equiv J` if atoms `i` and `j` are nearest neighbors and zero otherwise. In 3D systems a reasonable estimate of the Curie temperature can be obtained from mean-field theory as

.. math::

  T_c^{\mathrm{MF}}=\frac{NJS(S+1)}{3k_B},

where `k_B` is Boltzmann's constant, `N` is the number of nearest neighbors, and `S` is the maximum value of `S^z_i`



DFT calculation of `J`
======================

We now want to make a first principles calculation of the nearest neighbor exchange coupling constant `J`. Since the exchange coupling parametrizes the energy difference between aligned and anti-aligned spin configurations, we can obtain `J` by considering the energy difference between a ferromagnetic and an antiferromagnetic calculation. Note that both can be obtained as collinear DFT ground states subject to different spin constraints. For the :mol:`CrI_3` system, `J` can calculated as

.. math::

  J=\frac{E_{\mathrm{AFM}}-E_{\mathrm{FM}}}{3S^2},

where `E_{\mathrm{FM}}` and `E_{\mathrm{AFM}}` are the energies *per magnetic atom* of the ferromagnetic and antiferromagnetic configurations respectively.

1.   Derive the expression for `J` from the classical Heisenberg model yourself. In particular, how does the factor of 3 arise?

We have compiled a database of various 2D materials at https://cmrdb.fysik.dtu.dk/c2db/, which are relaxed with the PBE functional. We will therefore refrain from doing a full coverged geometry optimization and simply download the optimized structure from the database. Search the database for :mol:`CrI_3` (it appears as :mol:`Cr_2I_6` on the webpage). If you like, you can take a look at various properties of the material like band structure and stability. Download the ``.xyz`` file and save it as ``CrI3.xyz``. You can also setup the atomic structure similar by simply defining an atoms object with positions and lattice parameters stated at the webpage. With this input structure, run the code below to obtain a ``.gpw`` file containing a converged ferromagnetic calculation. The calculation will take about 30 minutes. To speed up the process, you can copy the code to a python script and submit it as a batch job to the DTU computers with multiple CPU cores. Try for example to send it to 16 cores for 30 minutes. If the relaxation in the code above did not finish you may kill it. It is not crucial to complete in order to proceed with the rest of the exercise. Continue with the theory below while you wait for the calculations to finish.


.. literalinclude:: CrI3_fm.py


Now run the code again (copy it into a new script with a different name) but change the initial magnetic configuration such that it becomes antiferromagnetic (remember to change the name of the ``.gpw`` file so you do not overwrite the ``CrI3_fm.gpw`` file). In practise, we do not need to constrain the antiferromagnetic calculation because it comprises a local minimum, but check the magnetic moments in the output to verify that we do indeed end up with an antiferromagnetic state!

Finally, we can calculate `J` and `T_c^{\mathrm{MF}}` by extracting the *ab initio* energy difference from the ``.gpw`` files. Fill in the formulas for `J` and `T_c` below and run the code.


.. literalinclude:: get_Tc_mf_student.py



More theory
===========


The Mermin-Wagner theorem
-------------------------
Completing the previous calculations, you should have obtained a value of `T_c`, which is on the order of 100 K. This is much larger than the experimental value.  However in 2D materials mean-field theory fails miserably and the results cannot be trusted. In fact, at finite temperatures the Heisenberg model stated above does not exhibit magnetic order in two dimensions. The reason is that entropy is dominant over enthalpy, such that the free energy is always minimized by disordered configurations at finite temperatures. This is summarized by the Mermin-Wagner theorem, which states that:

*Continuous symmetries cannot be spontaneously broken at finite temperature for systems with short range interactions in dimensions* `d\le2`.

The Heisenberg model above has a continuous rotational symmetry in the spin degrees of freedom and magnetic order is obtained by choosing a certain direction for all the spins. This means that the spin rotation symmetry is spontaneously broken in the magnetically ordered state. However, the direction of magnetization is arbitrary and can still be rotated without any energy cost, as long as the spins remain aligned with respect to each other.

In the Heisenberg model it is straightforward to calculate the collective magnetic excitations of the system, yielding a quadratic spin wave dispersion for ferromagnetic systems

.. math::

  \varepsilon(q)=Dq^2

in the limit of `q\rightarrow 0`. The spin wave excitations are bosons lowering the total spin along the magnetized direction by a single unit. Hence, the magnetization at finite temperatures `T` can be calculated from

.. math::

  M=M_0 - \int_0^\infty\frac{g(\epsilon)d\varepsilon}{e^{\varepsilon/k_B T} - 1}

where `M_0` is the ground state magnetization and `g(\varepsilon)` is the density of states for the spin waves.

1.   Calculate the spin wave density of states `g(\varepsilon)` as a function of dimensionality `d`. (Answer: `g(\varepsilon)=a_d D^{-d/2} \varepsilon^{(d-2)/2}`, where `D` is the spin wave stiffness defined above and `a_d` is a geometric constant depending on the dimension `d`)
2.   Compute the bosonic occupation and show that it diverges for `d\le2`. (Hint: You can convince yourself that it does by Taylor expanding the `\varepsilon\rightarrow 0` limit of the integral)

The divergence of the integral for `d\le2` shows that the ferromagnetic ground state is thermodynamically unstable and comprises an example of the Mermin-Wagner theorem: In `d\le2`, the free energy is always dominated by entropy at finite temperatures and magnetic order cannot be maintained if a material preserves the rotational symmetry of the spin degrees of freedom. The instability in two dimensions is closely related to the vanishing energy of magnetic excitations in the limit of `q\rightarrow 0`, which in turn is a consequence of the rotational symmetry.


Magnetic anisotropy
-------------------
Thanks to the Mermin-Wagner theorem, magnetic order is only possible in two dimensions if the rotational symmetry of the spins is *explicitly broken*. That is, there must be an additional term in the Hamiltonian that breaks the symmetry. Such a term can be provided by spin-orbit coupling which couples the spin to the lattice through the electronic orbitals.
We assume that :mol:`CrI_3` is isotropic in the plane of the monolayer and introduce an anisotropy term in the Heisenberg Hamiltonian of the form

.. math::

  H_{\mathrm{ani}}=A\sum_i(S_i^z)^2,

where we have chosen the `z`-direction to be orthogonal to the plane.

3.   Describe the physics of this term in the cases of `A<0` and `A>0`. Does the term fully break rotational symmetry of the ground state in both cases?



Magnetic anisotropy from DFT
============================

In the code below, the magnetic anisotropy is calculated for the ferromagnetic ground state. The function ``calculate_band_energy()`` will return the sum of Kohn-Sham eigenvalues for the occupied states. This energy will depend on the direction of the spins, which is specified by the polar and azimuthal angles `\theta` and `\varphi` respectively.

1.   What is the sign of `A` in the Hamiltonian above? (Fill in the formula for `A` and run the code)
2.   Does spin-orbit coupling break the rotational symmetry of the ground state?


.. literalinclude:: CrI3_anisotropy_student.py

We can also plot the total energy of the ground state as a function of the polar angle `\theta`.

3.   Run the following below and inspect the plot. Does it look like you would expect?


.. literalinclude:: CrI3_plot.py

.. image:: CrI3.svg

Now that we have calculated the anisotropy constant `A`, we are finally in a position to improve our estimate of the Curie temperature of :mol:`CrI_3`. But how do we get the critical temperature if we cannot apply mean-field theory? One way is to perform Monte-Carlo simulations of the classical Heisenberg model as a function of temperature and find the point where the total magnetization vanishes. The results of such simulations are well approximated by the expression [*2D Mater.* **6** 015028 (2019)] (https://iopscience.iop.org/article/10.1088/2053-1583/aaf06d)

.. math::

  T_c=T_c^{\mathrm{Ising}}\tanh^{1/4}\Big[\frac{6}{N}\log\Big(1-0.033\frac{A}{J}\Big)\Big],

where `T_c^{\mathrm{Ising}}=1.52\cdot S^2J/k_B` is the critical temperature of the ferromagnetic Ising model on a honeycomb lattice.

4.   Fill in the Monte-Carlo formula for `T_c` in the code below and calculate the Curie temperature using the values of `A` and `J` found above.


.. code::

   from numpy import tanh, log

   A = 0
   J = 0

   T0 = 1.52 # Ising limit
   T_c = ???
   print(f'T_c = {T_c:.1f} K')


The result for `T_c` should be in reasonable agreement with the experimental value of 45 K (expect to obtain a value around 30 K). Of course one should carefully check the convergence of all calculations in the present tutorial. In fact a converged calculation yields `J = 3.1` meV and `A=-0.38` meV, which results in `T_c = 37` K.
