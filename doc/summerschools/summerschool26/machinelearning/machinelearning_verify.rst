.. _machinelearning_verify:


===========================
Verifying the MLIP with DFT
===========================

It is now time to verify that what the MLIP calculates actually makes sense.  Note that the zero point of the energy may differ between the MLIP and GPAW calculations, as the MLIP is trained to reproduce VASP calculations and the atomic reference energies differ between DFT programs.  We can therefore only compare energy differences, not absolute energies.  The forces on the atoms are not affected by this.

As DFT calculations are expensive, it is necessary to run the verification with the smallest possible system.  That implies only a single layer in the 2D material, and preferably as small a system in the *x-y* directions without making the physics completely absurd.  In the introductory section you probably started off with a 4x4 supercell, for this part a 3x3 supercell is more appropriate.

You could follow a procedure like the following:

1. Set up a suitable system a bit smaller than the one used initially
2. Thermalize the system.
3. Add a projectile and run an MD simulation of the collision.
4. Select interesting frames from MD simulation.
5. Run DFT calculations on those frames.

**Steps 1 to 3** entails modifying the simulation scripts you have already used.  **Computational scientist survival tip:** Never ever edit old scripts because you need them for a new purpose.  Make a copy of the script and edit the copy.  Otherwise you lose the ability to reproduce your earlier results!  More than one scientific paper have not been written because of this!

**Step 4:**  Select frames where "something happens", e.g where there are significant forces on some atoms.  You can view the forces in ``ase gui`` and/or collect information about the magnitude of the largest force on any atom during the MD and plot it.

Running DFT calculations
========================

We run DFT calculations on individual frames of the MD trajectory.  Suitable parameters for the DFT calculations are a plane-wave cutoff of 350 eV and a k-point sampling of around 20 Å, meaning that the number of k-point multiplied by the size of the system should be this number in both in-plane directions.

The necessary plane-wave cutoff and k-point sampling depend on the chemical elements in the system, and a convergence test would be appropriate if time permits.

You can then compare the DFT forces with the machine learning forces.  Note that the forces vary dramatically in magnitude, that makes comparison difficult - the large forces may have large absolute errors but moderate relative errors, while for small forces it is the opposite.

The following lines set up the GPAW calculator for the DFT calculations.  You can build a script around it that calculates the DFT energy and forces for a single frame in the trajectory file, or for a selected list of frames.

.. literalinclude:: machinelearning_dft_multi.py
   :start-after: snippet-gpaw-start
   :end-before: snippet-gpaw-end

You can access both the MLIP and the GPAW forces in the script: the MLIP forces are available in the trajectory files, and can be obtained from the atoms *before* the GPAW calculator is attached, then the DFT forces can be calculated and obtained *after* it is attached.

The code below also includes a sanity check: that no atom has passed outside the simulation cell in the *z*-direction.  This is because we run the MLIP without periodic boundary conditions in the *z*-direction, but GPAW will (at least in plane-wave mode) ignore the boundary conditions specified on the atoms and always assume periodic boundaries.

.. literalinclude:: machinelearning_dft_multi.py
   :start-after: snippet-oldforces-start
   :end-before: snippet-oldforces-end

Be sure to save the DFT forces to one or more trajectory files.  Now you can try to analyze the difference between the forces calculated by the MLIP and by DFT.  Try to critically evaluate if the MLIP is doing reasonably well.

When you save the DFT results, you need to prevent GPAW from saving the dipole moment of the system, as that would trigger a bug in NequIP when you later try to use the calculation for fine tuning the MLIP (the symptom of the bug is a crash when reading the file).  To prevent saving the dipole moment, you need to explicitly state what you want saved in the Trajectory file:

.. literalinclude:: machinelearning_dft_multi.py
   :start-after: snippet-traj-start
   :end-before: snippet-traj-end

You can run the DFT code on the interactive node where you are logged in, but it will be significantly faster to submit it to the batch system on e.g. 8 CPUs.  You can see instructions for submitting Python jobs with MyQueue :ref:`here <gbar submitting>`.

You may want to "dry-run" GPAW and see how many k-points it actually uses in the reduced Brilloin zone.  Choosing the number of CPUs to be an integer multiple (or divisor) of this number gives the best performance.  You dry-run a GPAW script with this command::

   gpaw python --dry-run=1 script.py

This will run your script, but abort as soon as GPAW would start the actual calculation.  It will write the same ``.txt`` file as would otherwise be written (but it is cut short, of course).  In this file, you should look at the line ``Number of IBZ points:``, the context would be like this::

   BZ-sampling:
     Number of BZ points: 4
     Number of IBZ points: 2

     Monkhorst-Pack size: [2, 2, 1]
     Monkhorst-Pack shift: [0, 0, 0]

In this case, the irreducible Brillouin zone (IBZ) contains two k-points, and the number of CPUs should be an even number.  Preferably the remainder of the number should have small divisors, so a number of CPUs like 2, 4, 6, 8, 12 etc would work reasonably well.  If the number of IBZ points is 5, then you need to choose 5 or 10 CPUs, not 4 or 8!

After running the DFT tests and evaluated the results, you can then proceed to :ref:`fine-tuning the MLIP <machinelearning_finetune>`.
