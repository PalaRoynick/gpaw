.. _magnetism2:

=============================================
Part 2: Noncollinear magnetism in :mol:`VI_2`
=============================================

In materials where the dominant magnetic exchange coupling is antiferromagnetic,
or in cases where different exchange couplings compete, the ground state may
have a complicated noncollinear magnetic order. In this part of the tutorial
you will examine a prototypical monolayer with a noncollinear ground state,
namely :mol:`VI_2`. Starting from the structure file :download:``VI2.xyz``,
you will:

1) Relax the atomic structure using LDA
2) Compare a collinear antiferromagnetic structure with the ferromagnetic state
3) Obtain the noncollinear ground state
4) Calculate the magnetic anisotropy and discuss whether or not the material
   will exhibit magnetic order at low temperatures

Our basic reference point will still be the Heisenberg Hamiltonian given by:

.. math::

  H = -\frac{1}{2}\sum_{i,j}J_{ij}\mathbf{S}_i\cdot \mathbf{S}_j+A\sum_i(S_i^z)^2

but here we consider the antiferromagnetic case where `J<0`.


Optimizing the atomic structure
===============================

Since we will need to do LDA calculations later on, we will start of this part of the project by relaxing the atomic structure of :mol:`VI_2`
using the LDA functional. Usually, the difference in crystal structure between different magnetically ordered states is small, so we will
perform the relaxation in the ferromagnetic state, which has a smaller unit cell.

1.   First you should download the relaxed PBE crystal structure. Either, browse the C2DB at https://cmrdb.fysik.dtu.dk/c2db and download
     the ``.xyz`` file for :mol:`VI_2`.
2.   Fill in the expected ionic value for the V spins `S` below and run the code below to relax the crystal structure. The calculation takes about
     17 minutes. (Hint: V has the electronic configuration [Ar]3d\ `^3`\ 4s\ `^2`).

.. literalinclude:: VI2_gs_student.py


Magnetic anisotropy
===================

In the code above, we not only performed a structural optimization, but we also took the relaxed structure, switched off the `k`\ -point
symmetries and did a final DFT calculation with all the `k`\ -points in the Brillouin zone. We did this in order to be able to evaluate the
the magnetic anisotropy arising from the spin-orbit coupling.

1.   Adapt the code you used for :mol:`CrI_3` to calculate the magnetic anisotropy of :mol:`VI_2`.
2.   Is the easy-axis in-plane or out-of-plane?
3.   Do you expect to the :mol:`VI_2` to exhibit magnetic order at finite temperatures?


DFT calculations in a repeated cell
===================================

To realize that :mol:`VI_2` is in fact an antiferromagnetic material, we need to do a calculation starting from an antiferromagnetic alignment of
the spins. To do so, we need more than a single V atom in the unit cell. In the code below, the atomic structure is repeated once to obtain
two V atoms in the unit cell and it is shown how do a DFT calculation for the antiferromagnetic state.

1.   Fill in the `kpts` for the repeated unit cell.
2.   Replace the `...` with code to calculate the ferromagnetic state.
3.   Run the script once you have finalized the missing part for the ferromagnetic state. The calculation takes about 7 minutes.

.. literalinclude:: VI2_afm_student.py
   :end-before: ferromagnetic


Calculating J
=============

Finally, we are in a position to calculate the nearest neighbour Heisenberg exchange coupling `J`. Before doing so, please compare the
output of the antiferromagnetic and ferromagnetic calculations from the code above.

1.   Which state has the lowest energy?
2.   What will the sign of `J` be?

You should find that the antiferromagnetic state has a lower energy than the ferromagnetic one, but does this also mean that the calculated
configuration is the correct magnetic ground state? Rather, it implies that the spins prefer to be antialigned, i.e. that the exchange
coupling `J` is negative.

3.   Draw the structural arrangement of the V atoms on a piece of paper. Which type of magnetic lattice do they form?
4.   Fill in the spin configuration of the magnetic lattice and convince yourself that all spins cannot be antialigned to their nearest
     neighbours.

The latter finding means that the antiferromagnetic system is frustrated and the antiferromagnetic configuration we have computed will not
be the true ground state of the system.

Leaving the magnetic frustration aside for the moment, we will first calculate `J`, which we can still do even though that we have not found
the ground state yet.

5.   Use the classical Heisenberg model with nearest neighbor interaction only (and `A=0`) to derive the energy per magnetic site of the
     ferromagnetic and antiferromagnetic configurations respectively.

You should obtain the following:

.. math::

  E_{\mathrm{FM}} = E_0 - 3JS^2

and

.. math::
  E_{\mathrm{AFM}} = E_0 + JS^2

where `E_0` is some reference energy.

6.   Use these expressions to eliminate `E_0` and express `J` in terms of the energy difference per magnetic site of the two configurations.
7.   Write code to extract `E_\mathrm{fm}` and `E_\mathrm{afm}` as outlined below.
8.   Fill in the formula for `J` and evaluate it.

You should get a value for `J` around -1.4 meV.


.. code::

    E_fm = E_fm = ???
    E_afm = E_afm = ???

    J = ???
    print(f'J = {J * 1000:1.2f} meV')



Noncollinear configuration using a super cell
=============================================

As it turn out, the optimal spin structure on a trigonal lattice with antiferromagntice exchange coupling is to place all spins at
120\ `^\circ` angles with respect its neighbors.

1.   Draw this structure and convince yourself that it is indeed possible to put every spin at a 120\ `^\circ` angle with respect to its
     neighbors.
2.   What is the minimal number of magnetic atoms required in the magnetic unit cell to represent such a state?
3.   Verify that the Heisenberg model with classical spins gives a lower energy with this configuration than in the anti-aligned structure
     calculated above.

You should obtain the following energy for the 120\ `^\circ` noncollinear configuration:

.. math::

  E_{\mathrm{NC}}=E_0+\frac{3}{2}JS^2.

We will now check if we can verify this prediction within the LSDA. To do that we need to perform a noncollinear DFT calculation, which is
done by the code below.

4.   Read and try to understand the code to perform a noncollinear LSDA calculation in the 120\ `^\circ` noncollinear configuration.
5.   Replace the `...` with code to make a noncollinear LSDA calculation for the ferromagnetic state as well (we will need this for a later
     comparison).

Run the cell and verify that the energy per magnetic atom is lower in the 120\ `^\circ` noncollinear configuration compared to both of the
previous calculated states.


.. literalinclude:: VI2_noncol.py
   :end-before: snippet-stop


Noncollinear configuration using spin spirals
=============================================

The noncollinear spin configuration can also be represented in a single unit cell using the generalized Bloch's theorem. While the usual
boundary conditions from Bloch's theorem yields that translation by a lattice vector simply results in the multiplication of a phase factor
onto the wave function, the boundary conditions from the generalized Bloch's theorem yields that translation by a lattice vector can
additionally result in the rotation of spins. The rotation can be characterized by the spin spiral vector `\mathbf{q}` where, for example,
`\mathbf{q}=[1/2,0,0]` represents that translating by the first lattice vector rotates the spins by 1/2 * 360\ `^\circ`\ =180\ `^\circ` whereas
translating by the second and third lattice vectors leaves the spins unchanged.
(See [this page](https://gpaw.readthedocs.io/tutorialsexercises/magnetic/spinspiral/spinspiral.html) for further information.)

1.   For :mol:`VI_2`, the spins should rotate by 120\ `^\circ` when translating by the first and second lattice vector. Write down the `\mathbf{q}`
     vector that represents these boundary conditions.
2.   Consider the block of code below. Verify that you can get the same potential energy pr. atom using a single unit cell with the
     generalized Bloch theorem as you can using the usual Bloch theorem and a super cell.

You can generate a path of `\mathbf{q}`\ -points between the ferromagnetic boundary conditions and the 120\ `^\circ` spin rotating boundary
conditions using the command ``path = atoms.cell.bandpath('GK', npoints=7).kpts``.

3.   Wrap the ground state calculation in a for-loop over `\mathbf{q}`\ -points (``for i, q_c in enumerate(path):``) and obtain the energy as a
     function of `\mathbf{q}`. Plot the energy vs. `\mathbf{q}` curve. Where is the minimum?



.. code::

    import numpy as np
    from ase.io import read
    from gpaw import GPAW

    atoms = layer.copy()
    magmoms = np.zeros((3, 3), float)
    m = 3
    magmoms[0] = [m, 0, 0]

    q_c = [?, ?, ?]

    calc = GPAW(mode={'name': 'pw',
                      'ecut': 400,
                      'qspiral': q_c},
                xc='LDA',
                symmetry='off',
                parallel={'domain': 1, 'band': 1},
                magmoms=magmoms,
                soc=False,
                kpts={'size': (3, 3, 1)})
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    print(energy)


Anisotropy and exchange coupling from noncollinear DFT
======================================================

In the super cell calculation above we could have included spin-orbit coupling in a self-consistent way by setting 'soc'=True'. However, it
is more convenient for us to exclude it such that we can explicitly compute the single-ion anisotropy parameter `A` afterwards. If the
localized spin Hamiltonian with nearest neighbor exchange interactions is a good model, we should be able to obtain both `J` and `A` from
the noncollinear calculation as well.

1.   Make a script that computes the nearest neighbour exchange coupling `J` based on the noncollinear calculations performed above.
2.   Then make a script (could be the same one) that computes the spin-orbit coupling corrected energies of the ferromagnetic state. Check that
     you get the same either using a collinear configuration in the minimal unit cell or the non-collinear ferromagnetic configuration.
     You should calculate the energies of spins directed along the `x`, `y` and `z` directions (Hint: the anisotropy is calculated by rotating
     the entire initial spin configuration first by `\theta` and then by `\varphi`).
3.   Repeat point 2, but for the 120\ `^\circ` noncollinear configuration. In this case, you cannot align all spins to one direction, but
     takes as a reference the first V atom.

How does the computed exchange constant compare to the one obtained from the collinear FM and AFM calculations above?


Critical temperature?
=====================

Based on the noncollinear calculations above:

1.   What is the easy axis of the system?
2.   Does the easy axis agree with your initial findings for the simple ferromagnetic state?

You should find that the energy of the 120\ `^\circ` noncollinear configuration does not depend strongly on whether the first V atom is
aligned to the `x` or the `y` direction.

3.   If we assume full in-plane isotropy is there any rotational freedom left in the noncollinear ground state?
4.   What implications does this have for the critical temperature of the monolayer?

You might be able to convince yourself that some degree of in-plane anisotropy is required to obtain a finite critical temperature for the
120\ `^\circ` noncollinear magnetic order. Again, bear in mind that all the calculations in the present tutorialk ought to be properly
converged with respect to `k`\ -points, plane wave cut-off etc. to achieve an accurate estimate of e.g. the in-plane anisotropy.

Clearly the noncollinear spin state of :mol:`VI_2` is more difficult to describe than the ferromagnetic state in :mol:`CrI_3` and we do not yet have
a simple theoretical expression for the critical temperature as a function of anisotropy and exchange coupling constants. However, with the
rapid development of experimental techniques to synthesize and characterize 2D materials it does seem plausible that such a noncollinear 2D
material may be observed in the future.
