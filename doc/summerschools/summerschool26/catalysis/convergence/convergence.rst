.. _catalysis_convergence:

Convergence checks
==================

In this notebook we look at the adsorption energy and height of a nitrogen
atom on a Ru(0001) surface in the hcp site.  We check for convergence with
respect to:

* number of layers
* number of k-points in the BZ
* plane-wave cutoff energy


Nitrogen atom
-------------

First step is an isolated nitrogen atom which has a magnetic moment of 3.
More information:
:class:`ase.Atoms` and :ref:`GPAW parameters <parameters>`.

.. literalinclude:: eads.py
   :start-after: web-page
   :end-before: snippet-slab


Clean slab
----------

We use the :func:`ase.build.hcp0001` function to build the Ru(0001) surface.

.. literalinclude:: eads.py
   :start-after: snippet-slab
   :end-before: snippet-nslab


N/Ru(0001)
----------

Now, let's add a nitrogen atom in the "HCP" site:

.. literalinclude:: eads.py
   :start-after: snippet-nslab
   :end-before: snippet-nslab

Alternatively, you can just use the
:func:`ase.build.add_adsorbate` function:

.. literalinclude:: eads.py
   :start-after: snippet-hcp
   :end-before: snippet-hcp-end

.. image:: nru2.png

Now, calculate the total energy and the unrelaxed adsorption energy:

.. literalinclude:: eads.py
   :start-after: snippet-eads
   :end-before: snippet-eads-end

If you also calculate for forces (``f = nslab.get_forces()``),
you will see that the force on the N-atom is quite big (``print(f[-1])``).
Let's freeze the surface and relax the
adsorbate.  We use
:class:`ase.optimize.BFGSLineSearch` and
:class:`ase.constraints.FixAtoms`
for this task.

.. literalinclude:: eads.py
   :start-after: snippet-eads-end


Convergence
-----------

In order to make it easy to check for convergence of the adsorption energy
and height we write functions (called ``adsorp()`` and ``atom_energy()``)
that does all of the stuff above taking
``nlayers``, ``nkpts`` and ``ecut`` as input parameters.

Take a look at the :download:`convergence.py` script and try to
understand what it does:

.. literalinclude:: convergence.py

If you run the script, the ``main()`` function will run and all the
convergence calculations will be run.  You can run the script yourself or
you can just download the results here: :download:`results.json`. We will
now analyse the results.

Try this:

.. literalinclude:: plot.py
   :start-after: creates
   :end-before: layers.svg

This should produce this:

.. image:: layers.svg

Here are the plots for **k**-point and `E_{text{cut}}` convergence:

.. image:: kpts.svg
.. image:: ecut.svg


Conclusion
----------

For accurate calculations you would need:

* a plane-wave cutoff of 600 eV
* 5 layers of Ru
* 9x9 Monkhorst-Pack grid for BZ sampling (for a 1x1 unit cell)

For our quick'n'dirty calculations we will use 350 eV, 2 layers and a
4x4 `\Gamma`-centered Monkhorst-Pack grid (for a 2x2 unit cell).
