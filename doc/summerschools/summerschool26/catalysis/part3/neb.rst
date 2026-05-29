Introduction to Nudged Elastic Band (NEB) calculations
======================================================

This tutorial describes how to use the NEB method to calculate the diffusion
barrier for an Au atom on Al(001). If you are not familiar with the NEB
method, some relevant references are listed on ASE's
documentation page on the :mod:`ase.mep.neb` module.

The tutorial uses the EMT potential instead of DFT, as this is a lot faster.
It is based on a :ref:`tutorial found on the ASE
webpage <ase:selfdiffusion_example>`.

.. literalinclude:: neb.py
  :start-after: snippet-imports-start
  :end-before: snippet-imports-end


First we set up the initial state and check that it looks ok:


.. literalinclude:: neb.py
  :start-after: snippet-setup-surface-start
  :end-before: snippet-setup-surface-end


Then we optimise the structure and save it.

.. literalinclude:: neb.py
  :start-after: snippet-optimize-start
  :end-before: snippet-optimize-end


We make the final state by moving the Au atom by one lattice constant and
optimise again.

.. literalinclude:: neb.py
  :start-after: snippet-optimize-finalstate-start
  :end-before: snippet-optimize-finalstate-end


Now we make a NEB calculation with 3 images between the inital and final
states. The images are initially made as copies of the initial state and the
call ``interpolate()`` performs a linear interpolation between the initial and
final states. As always, we check that everything looks ok before we run the
calculation.

NOTE: The linear interpolation works well in this case but not for e.g.
rotations. In this case an improved starting guess can be made with the
:ref:`IDPP method <neb_idpp_tutorial>`.

.. literalinclude:: neb.py
  :start-after: snippet-setupneb-start
  :end-before: snippet-setupneb-end

After verifying the structure, we can run the optimization:

.. literalinclude:: neb.py
   :start-after: snippet-bfgsneb-start
   :end-before: snippet-bfgsneb-end

We visualize the final path with:

.. literalinclude:: neb.py
   :start-after: snippet-viewfinalimages-start
   :end-before: snippet-viewfinalimages-end


You can find the barrier by selecting Tools->NEB in the gui,
or you can make a script using :class:`ase.mep.neb.NEBTools`, e.g.:

.. literalinclude:: neb.py
  :start-after: snippet-nebtools-start
  :end-before: snippet-nebtools-end


Exercise
--------

Now you should set up your own NEB using the configuration with :mol:`N_2`
lying down as the initial state and the configuration with two N-atoms
adsorbed on the surface as the final state.

Parallelisation
===============

The NEB should be parallelised over images. An example can be found in
:ref:`this GPAW tutorial <neb>`. The
script enumerates the CPUs and uses this number (``rank``) along with the
total number of CPUs (``size``) to distribute the tasks.
Note that the NEB class now needs to be initialized by using
``NEB(images, parallel=True)``.

.. literalinclude:: neb.py
   :start-after: snippet-mpiranks-start
   :end-before: snippet-mpiranks-end


For each image, we assign a set of CPUs identified by their rank. The rank
numbers are given to the calculator associated with this image.

.. literalinclude:: neb.py
   :start-after: snippet-gpaw-start
   :end-before: snippet-gpaw-end

When running the parallel NEB, you should choose the number of CPU cores
properly.  Let ``Ncore = N_im * Nk`` where ``N_im``
is the number of images, and ``Nk``
is a divisor of the number of k-points; i.e., if there are 6 irreducible
k-points, ``Nk`` should be 1, 2, 3 or 6.  Keep the total number of cores to 24 or
less, or your job will wait too long in the queue.

Input parameters
================

Some suitable parameters for the NEB are given below:

* Use the same calculator and constraints as for the initial and final images, but remember to set the ``communicator`` as described above
* Use 6 images. This gives a reasonable description of the energy landscape and can be run e.g. on 12 cores.
* Use a spring constant of 1.0 between the images. A lower value will slow the convergence
* Relax the initial NEB until ``fmax = 0.1`` eV/Å, then switch on the climbing image and relax until ``fmax = 0.05`` eV/Å.

Once the calculation is done, you should check that the final path looks
reasonable. What is the N—N distance at the saddle point? Use NEBTools to
calculate the barrier. Is :mol:`N_2` likely to dissociate on the surface
at room temperature?
