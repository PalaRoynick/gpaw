.. _machinelearning_projectile:


=================================
Implanting atoms in a 2D material
=================================

We now want to let a projectile atom interact with our 2D material.  You can choose freely which kind of atom you use as a projectile, which gives you a number of choices.

* The projectile atom can be a component of the material, or it can be foreign.
* If foreign,
  
  - The projectile atom can be *lighter*, *similar*, or *heavier* than the atoms in the material.
  - It can be *reactive* (like oxygen or early transition metals) or relatively *inert* (like helium or gold).

The energy of the impacting atom matters, there are roughly three regimes:

* The energy can be *small* compared to the chemical bond energy, e.g. a few eV or less ("soft landing").
* The energy can be *comparable to the bond energy*, perhaps in the range 2-20 eV.
* The energy can be *large*, perhaps 50 eV or above.

But as you will see if you experiment, both momentum and energy matters so light atoms behave very differently from heavy atoms.

Adding a projectile to the simulation
=====================================

You should start with a thermalized 2D material, where the atomic vibrations have a well-defined temperature.  That means starting with the output of the :ref:`MD simulation you previously made <md_test>`.  You can read the last timestep of the simulation like this::

    from ase.io import read
    atoms = read('md.traj')
    atoms.set_constraint(None)

(assuming the file name is ``md.traj``).  If you want to read another configuration from the file, give the frame number as the second argument.  The last line removes the constraint on the center of mass that was used during the thermalization.

You need to place your atoms in a reasonable position.  It is easiest to visualize the simulations if it is not too near the simulation edges, so somewhere near the center of the simulation cell, and perhaps 5Å above the material.  You can find the center of the unit cell using ``atoms.cell``.  This unit cell object can be indexed to find the three unit cell vectors, so the *x-y* coordinate of the center of the unit cell can be found from ``atoms.cell[0]`` and ``atoms.cell[1]``.  You also need to place the atom a controlled distance above (or below) the material, here it may be useful to use ``atoms.positions[:,2].max()`` to find the maximal *z*-coordinate of the atoms in the material.  Then calculate the momentum of the atom corresponding to the desired energy, and direct it towards the material.

The projectile atom can then be added to the system like this:

.. literalinclude:: machinelearning_projectile_student.py
   :start-after: snippet-addprojectile-start
   :end-before: snippet-addprojectile-end


Molecular dynamics of the impact
================================

The MD simulation should now be done without coupling to a heat bath, as the system is already at the correct temperature - most thermostatting algorithms would interpret the high kinetic energy of the projectile as the system being way out of thermal equilibrium, and try to pull that energy out.  Some algorithms will merely slow the projectiles, others will scale down the kinetic energy of all atoms, freezing the 2D material!

We therefore use the simplest MD algorithm available, ``VelocityVerlet``, which merely solves Newton's 2nd law numerically without coupling to external quantities.  As the impacting atom is fast, a low time step is necessary, probably 1 fs or smaller.  The snippet below uses 1 fs, but a smaller timestep is probably necessary if the impact energy is high and the impacting atom is light.

The Velocity Verlet integrator is set up like this:

.. literalinclude:: machinelearning_projectile_student.py
   :start-after: snippet-mdsetup-start
   :end-before: snippet-mdsetup-end

It is important to check that the time step is not too large, this is done by plotting the total energy of the system and checking that it is not varying significantly (i.e. by more than a few meV/atom).  The total energy is stored in the trajectory file and can be plotted by ``ase gui``, or you can use the ``Datacollector`` object defined previously and plot the total energies it collects.

Experiment
==========

You can plot the resulting simulation with ``ase gui md_impact.traj``

Now you can experiment with various projectiles and collision energies.  Also, it will matter which atom in the 2D material you hit, and whether you hit it off-center.

