
.. _machinelearning_intro:


===================================
Selecting and testing a 2D material
===================================

For this exercise, you should select a 2D material to work with.  The material should not have a huge unit cell, and it should be well described by the foundational model we start to work with.  The latter has to be checked.

Select a 2D material from the `Computational 2D-Materials Database (C2DB) <https://c2db.fysik.dtu.dk>`_.  It contains around 17000 2D materials, many of them hypothetical.  All the well-known ones are there, such as C (graphene), BN (hexagonal boron nitride), MoS<sub>2</sub> (molybdenum disulphide, a transition metal dichalchogenide), Ti<sub>2</sub>C (titanium carbide, an MXene).  There are also more exotic materials, like CrBrS - still experimentally observed but with a more complicated structure.

Search for a composition in C2DB.  In this example, we look for MoS2 you write MoS2 in the search box.  Look for the material with the lowest "Energy above hull", if that is zero the material is thermodynamically stable, if it is low it is possibly metastable.  In this case we find two structures, the stable so-called 2H phase and the metastable 1T phase with twice as many atoms in the unit cell.

.. image:: c2db_MoS2_search.png
   :width: 80%

Once we click on the stable phase, we get an overview:

.. image:: c2db_MoS2_view.png
   :width: 80%

On the overview, please note a few features.  The field **Experimental realizations** shows that this material actually exists.  And most importantly, in the upper right corner you can download the structure in a few file formats.  The XYZ format is particularly simple. Clicking on the box gives you the file ``1MoS2-1.xyz`` which we will need in the following.

**PLEASE CHOOSE YOUR OWN MATERIAL!**  If you work together in a group, you should choose different materials.

Note that clicking the download link will download the file to your laptop, not to the DTU system.  If you use MobaXTerm to log in, you can just transfer it using the remote file viewer in the left part of your MobaXTerm window.  If you log in in another way, it may be easiest to download the file directly to the DTU system: Right-click on the link and copy the link, then copy it into this command::

    curl -o MoS2.xyz XXXXX

where the XXXX is replaced with the link you paste, probably something like ``https://c2db.fysik.dtu.dk/material/1MoS2-1/download/xyz``

Look at the material
====================

You can examine the material with the ASE gui::

    ase gui MoS2.xyz

Note that this shows you a single unit cell.  On the menu, choose ``View`` then ``Repeat ...`` and repeat it in the x and y direction to get a better idea of the structure.

Energy minimization test of the Machine Learning potential
==========================================================

We need to test if the MLIP is at all sensible for this structure.  Does the structure remain the same if we minimize the energy?  Is the structure stable during a short molecular dynamics (MD) simulation?  If not, that particular material is badly described by the potential, and you should choose another one (in a scientific project, you would of course choose another MLIP instead).

First, we test the energy minimization.  We work on a 4x4 unit cell as that allows us to discover if the MLIP incorrectly wants to dimerize the structure or something like that.

We start by loading the structure.

.. literalinclude:: machinelearning_intro_student.py
   :start-after: snippet-initialize-start
   :end-before: snippet-initialize-end

The we set up the machine learning potential.  We need to tell it which elements are present in the simulation.  Potentially, the elements could also have different names in the model, but that is not the case here, so the dictionary maps existing chemical symbols to themselves.

.. literalinclude:: machinelearning_intro_student.py
   :start-after: snippet-mlip-start
   :end-before: snippet-mlip-end

Now we test if running an energy minimization changes something dramatical.  We check if the unit cell changes significantly or if any atom moves significantly during the energy minimization.  The ``FrechetCellFilter`` presents atomic degrees of freedom and unit cell degrees of freedom in a unified way, so they can be minimized together.  It is important to mask out the non-periodic z-axis from the unit cell, as there is no energy change associated with changing it.  The order of the unit cell components are the commonly used Voigt ordering: ``xx``, ``yy``, ``zz``, ``yz``, ``xz``, ``xy``.

.. literalinclude:: machinelearning_intro_student.py
   :start-after: snippet-minimization-start
   :end-before: snippet-minimization-end

Hopefully the unit cell changes little (compared to their size), and the scaled positions also very little (compared to 1.0).  You check the scaled positions because the Cartesian position *will* change as the unit cell changes.

If something has changed, you can inspect the minimization trajectory with

::

    ase gui minimization.traj

IMPORTANT: The energy minimization will have changed the unit cell slightly, and is likely to have slightly broken the symmetry due to numerical noise.  This will cause trouble later, so **do not use this minimized structure in the following.**

Molecular dynamics test of the Machine Learning potential
=========================================================

Now we can run a molecular dynamics (MD) simulation and see if the structure remains stable over a short simulation.  This example also illustrates how data can be collected during the simulation and then plottet - most of that information is also stored in the trajectory file.

Spend the time needed to understand the following code.  We use the Langevin dynamics to couple the atoms to a heat bath.  This may cause the center of mass to fluctuate, which is annoying when visualizing, so we fix those three degrees of freedom.

.. literalinclude:: machinelearning_intro_student.py
   :start-after: snippet-md-start
   :end-before: snippet-md-end

We can then plot the temperature and see that it has stabilized.

.. literalinclude:: machinelearning_intro_student.py
   :start-after: snippet-plotT-start
   :end-before: snippet-plotT-end

**IMPORTANT:** You now have to visually inspect the MD trajectory and verify that nothing wild happens - if the structure is stable the atoms should be vibrating but not moving around.

::

    ase gui md.traj


