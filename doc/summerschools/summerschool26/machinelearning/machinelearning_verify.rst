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

We run DFT calculations on individual frames of the MD trajectory.  Suitable parameters for the DFT calculations are a plane-wave cutoff of 350 eV and a k-point sampling of XXX Å, meaning that the number of k-point multiplied by the size of the system should be this number in both in-plane directions.  If you get an odd number of k-points you may get slightly more accuracy at the same computational cost by increasing the number of k-point by one.

The necessary plane-wave cutoff and k-point sampling depend on the chemical elements in the system, and a convergence test would be appropriate if time permits.

