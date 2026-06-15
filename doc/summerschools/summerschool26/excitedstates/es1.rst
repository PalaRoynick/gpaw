

=============================================
Introduction to band gaps and band structures
=============================================


In this exercise we study some of the key properties of materials for photovoltaic applications. In the first part of the exercise you will be given a semiconductor material and you are requested to investigate:

* atomic structure
* band gap
* band gap position
* band structure
* compare how different exchange correlation functionals perform

We will use ASE and GPAW packages, and at the end of this excercise, you will be requested to write your own scripts and submit them to the supercomputer. You will be asked to compare your results to each other and to discuss your results with other groups studying different materials.

Atomic structure
================

As you have already learnd in the previous sesion, when investigating the electronic structure of a material, the first thing to be done is to find the atomic positions by relaxing the atoms, i.e. minimizing the forces.

Here is some information to help you to build the ase.Atoms object:

* Silicon crystalizes in the diamond structure with lattice constant a=5.43 Å
* Germanium crystalizes in the diamond structure with lattice constant a=5.66 Å
* Diamond has diamond structure (!) with lattice constant a=3.56 Å
* CdTe crystalizes in the zincblende structure with lattice constant a=6.48 Å
* GaAs crystalizes in the zincblende structure with lattice constant a=5.65 Å
* Monolayer BN centered in a hexagonal unit cell with a=2.5 Å (and 7 Å of vacuum at each side to prevent it from interacting with its periodic copies) and a basis of (0,0) and (0, :math:`a / \sqrt{3}`)

The first thing you should do is to create an :class:`ase:ase.Atoms` object (click the link to see ways to build atoms objects). In order to do so, you might find useful to use one of the crystal structures included in ase.build.bulk (hint, if you have an element of the IV group you might be interested on this link :func:`ase:ase.build.bulk`) or you might have to create a list/array for the atomic positions and another one for the unit cell and then create an atoms object (hint: see above). 


.. literalinclude:: solution1.py
   :start-after: # snippet-basic-imports-start
   :end-before: # snippet-basic-imports-end



.. literalinclude:: solution1.py
   :start-after: # snippet-structures-start
   :end-before: # snippet-structures-end


We are now going to relax the structure. To do so, we need to add a calculator, GPAW, to get DFT energies, and forces. We are going to use PBE exchange correlation functional.

Since we are going to relax the unit cell, we need to use the plane wave mode, since it is the only that includes the stress-tensor. In order to do so, remember this mode requires you to specify the plane wave cut-off
(hint: We recommend plane wave cut-off of 600 eV and the k-point mesh size could be (6,6,6) if you want it to run reasonably fast and to get a reasonable result). We will discuss convergence further in the next section.

The materials we are looking at are semiconductors. Thus, the default value for the Fermi-Dirac smearing (i.e. occupations function) is too high (it is set up to 0.1 eV to work with metals). We recommend setting it to 0.01eV

These links might be helpful for you:

* :ref:`manual_mode`
* :ref:`lattice_constants`


.. literalinclude:: solution1.py
   :start-after: # snippet-calculator-start
   :end-before: # snippet-calculator-end

We are going to relax the atomic positions and the unit cell at the same time. To do so, we are going to use the :class:`ase:ase.filters.UnitCellFilter` and the BFGS (or QuasiNewton) optimizer.


.. literalinclude:: solution1.py
   :start-after: # snippet-optimizer-start
   :end-before: # snippet-optimizer-end


Make sure that you have understand the difference of optimizing a bare atoms object and using a filter!
**Bonus**: Would you like to visualize the trajectory using ase gui, you can attach a trajectory file now by creating a new cell. After you execute the op.run cell, you can create another cell saying
! ase gui filename.traj
and execute it

.. literalinclude:: solution1.py
   :start-after: # snippet-run-optimization-start
   :end-before: # snippet-run-optimization-end





Band gap and band structure
===========================


We are now going to illustrate how to compute the band gap and obtain the band structure for a toy example with GPAW. You will be writing and submitting scripts doing your own meaningful calculations in the next section of this exercise, so do not worry now about the parameters, we know they are not a good choice and band structures look ugly :).

The starting point for this section (and you might also want to use it in your own scripts) will be the PBE relaxed structure from the previous section:


.. literalinclude:: solution1b.py
   :start-after: # snippet-restart-from-relaxed-start
   :end-before: # snippet-restart-from-relaxed-end


We are now going to restart the calculator and recompute the ground state, saving it to a new gpw file. As we are dealing with small bulk system, plane wave mode is the most appropriate here.
It is generally a good idea to choose a finer kpoint mesh for the band structure, but we are going to make the opposite choice here.
We are also going to use LDA, which is faster but not very good at predicting bandgaps (yes, we know, you are going to get a silly value here).


.. literalinclude:: solution1b.py
   :start-after: # snippet-lda-calculator-start
   :end-before: # snippet-lda-calculator-end


Lets use this calculator to get the energy, the *valence band maximum*, the *conduction band minimum*, and the *band gap*, as the difference of the two of the VBM and the CBM.

For the VBM and CBM, we are going to use the get_homo_lumo method of the calculator. This method returns the energy of the highest Kohn-Sham occupied orbital (called HOMO here) and the energy lowest Kohn-Sham unoccupied orbital (the LUMO). We are going to compute the band gap at this level of theory from the difference between both.


.. literalinclude:: solution1b.py
   :start-after: # snippet-homo-lumo-doc-start
   :end-before: # snippet-homo-lumo-doc-end


.. literalinclude:: solution1b.py
   :start-after: # snippet-band-gap-start
   :end-before: # snippet-band-gap-end


.. literalinclude:: solution1b.py
   :start-after: # snippet-save-lda-start
   :end-before: # snippet-save-lda-end



Band structure:
---------------
Next, we calculate eigenvalues along a high symmetry path in the Brillouin zone. You can find the definition of the high symmetry k-points for the fcc lattice in :data:`ase:ase.dft.kpoints.special_points`.

If your system is in the fcc or the diamond structures, then, your path may look something like 'GXWKL'. For BN, 'GMKG'.

For the band structure calculation, the density is fixed to the previously calculated ground state density, and as we want to calculate all k-points, symmetry is not used (symmetry='off').


.. literalinclude:: solution1b.py
   :start-after: # snippet-fixed-density-start
   :end-before: # snippet-fixed-density-end


Finally, we compute the band structure using ASE's :class:`ase:ase.spectrum.band_structure.BandStructure`.


.. literalinclude:: solution1b.py
   :start-after: # snippet-band-structure-doc-start
   :end-before: # snippet-band-structure-doc-end


.. literalinclude:: solution1b.py
   :start-after: # snippet-plot-band-structure-start
   :end-before: # snippet-plot-band-structure-end


.. literalinclude:: solution1b.py
   :start-after: # snippet-save-band-structure-start
   :end-before: # snippet-save-band-structure-end


.. image:: Si_bandstructure_LDA.png

Convergence (optional but recommended)
======================================


What happened to the results in the previous section? Did they look reasonable, or can you tell something went wrong?
In this section, we study the convergence of the results with the parameters that improve the completeness of the basis.

**Note**: If your are running out of time to complete the exercise (i.e., you are left 20 minutes), contact us, we will help you to jump to the next section. You might be able to come back to discuss convergence in DFT in the last day of the summer school.


Numerical convergence of DFT calculations should always be checked to avoid obtaining spurious results that are caused by a very coarse discretization. In this tutorial you can find an example on how to find a converged lattice constant for aluminum:
:ref:`lattice_constants`

The k-point mesh and the plane wave energy cut-off in the previous section were too low.

We suggest that you play around with the number of k-points and the plane wave cutoff rerunning the previous cells. Increasing their value produces better results, but also increases the computation time.

Finally, we suggest you to explore the convergence of the band gap as in the tutorial for the lattice constant. To do so, You will have to write a script. You may want to use the tutorial and the previous cells as a guide.



The band gap with different exchange correlation functionals
============================================================

You are now about to complete the last part of the exercise. Now that you know how to do ground state plane wave calculations and to find the band structure of a semiconductor, we ask you to discuss the effect of choosing a functional at a given level of theory.
We propose you to study the results with the following functionals:

* LDA (the one you have just used)
* PBE
* RPBE
* mBEEF

mBEEF is an meta GGA exchange correlation functional inspired from Bayesian statistics. An essential feature of these functionals is an ensemble of functionals around the optimum one, which allows an estimate of the computational error to be easily calculated in a non-self-consistent fashion. Further description can be found in:

* \J. J. Mortensen, K. Kaasbjerg, S. L. Frederiksen, J. K. Nørskov, J. P. Sethna, and K. W. Jacobsen (2005). Phys. Rev. Lett. 95, 216401
* Wellendorff, J., Lundgaard, K. T., Jacobsen, K. W., & Bligaard, T. (2014). The Journal of Chemical Physics, 140(14), 144107.

To complete this part of the exercise, we suggest that you write scripts and submit them (i.e. write one script for each functional) so that they can run in parallel. As a guide, you can use the LDA calculations you have already done.

Remember to choose a k-point mesh and an plane energy cut-off that make sense. In case of doubt, just ask.

You do not need to relax the structure again, the PBE relaxed structure you have saved to a gpw file in the beginning is good enough. Your script can just read it. :)

Your script should contain a ground state calculation with the functional you are studying. We suggest you print E, VBM, CBM and the gap to a file, together with the functional, so that you can use them for the discussion the last day of the summer school. You should save the ground state to a gpw file (it is a good idea to give different names to the files of different xc functionals) and restart the calculator from that file to compute the band structure. Save the plot to a png file and the data to a .json.
