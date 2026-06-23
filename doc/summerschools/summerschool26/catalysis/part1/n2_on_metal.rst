Catalysis: Dissociative adsorption of :mol:`N_2` on a metal surface
===================================================================

This is the rate limiting step for ammonia synthesis.

**Scientific disclaimer:**  These calculations are done on a flat surface.
In reality, the process takes place at the foot of an atomic step on the
surface.  Doing calculations on this more realistic system would be too slow
for these exercises.  For the same reason, we use a metal slab with only two
layers, a realistic calculation would require the double.


:mol:`N_2` Adsorption on a metal surface
----------------------------------------

This tutorial shows how to calculate the adsorption energy of an
:mol:`N_2` molecule on a closepacked Ru surface. The first cell imports
some modules from the ASE and GPAW packages

.. literalinclude:: n2_on_metal.py
   :start-after: snippet-imports-header-start
   :end-before: snippet-imports-header-end


Setting up the metal surface
----------------------------

Ru crystalises in the hcp structure with a lattice constants a = 2.706 Å and
c = 4.282 Å.  It is often better to use the lattice constants corresponding
to the DFT variant used (here PBE with PAW).  We get this from
https://oqmd.org.

We model the surface by a 2 layer slab of metal atoms, and add 5Å vacuum on
each side.

We visualize the system with ASE GUI, so you can check that everything looks
right.  This pops up a new window.

.. literalinclude:: n2_on_metal.py
   :start-after: snippet-setup-slab-start
   :end-before: snippet-setup-slab-end

To optimise the slab we need a calculator. We use the GPAW calculator in
plane wave (PW) mode with the PBE exchange—correlation functional. The
convergence with respect to the cutoff energy and k-point sampling should
always be checked — see :ref:`catalysis_convergence`
for more information on how this
can be done. For this exercise an energy cutoff of 350eV and 4x4x1 k-point
mesh is chosen to give reasonable results with a limited computation time.

.. literalinclude:: n2_on_metal.py
   :start-after: snippet-calc-define-start
   :end-before: snippet-calc-define-end

The bottom layer of the slab is fixed during optimisation. The structure is
optimised until the forces on all atoms are below 0.05 eV/Å.

.. literalinclude:: n2_on_metal.py
   :start-after: snippet-optimize-slab-start
   :end-before: snippet-optimize-slab-end

The calculation will take ca. 5 minutes. While the calculation is running you
can take a look at the output. How many k-points are there in total and how
many are there in the irreducible part of the Brillouin zone? What does this
mean for the speed of the calculation?

What are the forces and the energy after each iteration? You can read it
directly in the output above, or from the saved .traj file like this:

.. literalinclude:: n2_on_metal.py
   :start-after: snippet-read-traj-start
   :end-before: snippet-read-traj-end

Often you are only interested in the final energy which can be found like this:

.. literalinclude:: n2_on_metal.py
   :start-after: snippet-final-energy-start
   :end-before: snippet-final-energy-end


Making a Nitrogen molecule
--------------------------

We now make an :mol:`N_2` molecule and optimise it in the same unit cell
as we used for the slab.

.. literalinclude:: n2_on_metal.py
   :start-after: snippet-optimize-n2-start
   :end-before: snippet-optimize-n2-end


We can calculate the bond length like this:

.. literalinclude:: n2_on_metal.py
   :start-after: snippet-bond-length-start
   :end-before: snippet-bond-length-end

How does this compare with the experimental value?


Adsorbing the molecule
----------------------

Now we adsorb the molecule on top of one of the Ru atoms.

Here, it would be natural to just add the molecule to the slab, and minimize.
However, that takes 45 minutes to an hour to converge, **so we cheat to speed
up the calculation.**

The main slowing-down comes from the relaxation of the topmost metal atom
where the :mol:`N_2` molecule binds, this atom moves a quarter of an
Ångström out.  Also, the binding length of the molecule changes when it is
adsorbed, so we build a new molecule with a better starting guess.

.. literalinclude:: n2_on_metal.py
   :start-after: snippet-setup-adsorption-start
   :end-before: snippet-setup-adsorption-end

We optimise the structure.  Since we have cheated and have a good guess for
the initial configuration we prevent that the optimization algorithm takes
too large steps.

.. literalinclude:: n2_on_metal.py
   :start-after: snippet-optimize-adsorption-start
   :end-before: snippet-optimize-adsorption-end

The calculation will take a while (10-15 minutes). While it is running please
follow the guidelines in the **Exercise** section below.

Once the calculation is finished we can calculate the adsorption energy as:
`E_{\mathrm{ads}} = E_{\mathrm{slab+N}_2} - (E_{\mathrm{slab}} + E_{\mathrm{N}_2})`.



.. literalinclude:: n2_on_metal.py
   :start-after: snippet-adsorption-energy-start
   :end-before: snippet-adsorption-energy-end

Try to calculate the bond length of :mol:`N_2` adsorbed on the surface.
Has it changed?  What is the distance between the :mol:`N_2` molecule and
the surface?

Exercise
--------

1) Make a script and set up an adsorption configuration where the
   :mol:`N_2` molecule is lying down with the center of mass above a
   three-fold hollow site as shown below. Use an adsorption height of 1.7 Å.

.. image:: ../N2Ru_hollow.png

Remember that you can read in the `traj` files you have saved, so you don't
need to optimise the surface again.

View the combined system before you optimize the structure to ensure that you
created what you intended.

.. literalinclude:: n2_on_metal.py
   :start-after: snippet-adsorption-hollow-start
   :end-before: snippet-adsorption-hollow-end


Note that when viewing the structure, you can find the index of the
individual atoms in the ``slab`` object by clicking on them.

You might also find the
:meth:`~ase.Atoms.get_center_of_mass`
and
:meth:`~ase.Atoms.rotate`
methods useful.

Now you should optimize the structure as you did before with the
:mol:`N_2` molecule standing.  The calculation will probably bee too long
to run interactively.  Prototype it in the terminal, then interrupt the
calculation.

Check the number of irreducible k-points and then submit the job as a batch
job running on that number of CPU cores.

3) Make a configuration where two N atoms are adsorbed in hollow sites on the
   surface as shown below

.. image:: ../2NadsRu.png

Note that here the two N atoms sit on next-nearest hollow sites.  An
alternative would be to have them on nearest neighbor sites.  If you feel
energetic you could investigate that as well.  Also, there are two different
kinds of hollow sites, they are not completely equivalent!

Optimise the structure and get the final energy. Is it favourable to
dissociate :mol:`N_2` on the surface? What is the N—N distance now? What
does that mean for catalysis?
