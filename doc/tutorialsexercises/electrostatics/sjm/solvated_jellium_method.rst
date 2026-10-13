.. _solvated_jellium_method:

=============================
Solvated Jellium Method (SJM)
=============================

Please familiarize yourself with the theory of the SJM approach before starting this tutorial, by reading the :ref:`SJM documentation page<sjm>`.

A simple Au(111) slab
=====================

As a simple example, we'll examine the calculation of a Au slab with a water overlayer calculated at a potential of -0.2 V versus SHE.
To speed the computation, we'll use a tiny slab with a single water molecule over the top.

To use the solvated jellium approach, we use the regular GPAW calculator and attach a :class:`~gpaw.extensions.electrochemistry.SJM` object to it through the ``extensions`` keyword.
The SJM-specific settings are the keys of its ``potentiostat`` dict.
In our example script, we set only one of them---the target potential---and keep the rest at their default values.
Keep in mind that the *absolute* potential has to be provided, where the value of the SHE potential on an absolute scale is approximately 4.4 V.

Since SJM utilizes an implicit solvent to screen the field, we will need to specify some solvent parameters as well.
SJM relies on GPAW's :ref:`continuum_solvent_model` to provide this screening, and the ``cavity``, ``dielectric`` and ``interactions`` parameters are given to the :class:`~gpaw.extensions.electrochemistry.SJM` object as its ``solvation`` dict.
Reasonable values to simulate room-temperature water are used in the script below.
In practice, since the purpose of the implicit solvent is just to screen the field, the net results---such as binding energies and barrier heights---will be relatively insensitive to the choice of implicit solvent parameters, so long as they are kept consistent.

.. note::

   Both ``solvation`` and ``potentiostat`` can, in principle, be left out.
   Without ``solvation``, SJM uses the water of the script below---the same dielectric constant, surface tension, temperature and ``SJMPower12Potential``---except that it does not clear the solvent from the water layer (no ``H2O_layer``) and gives hydrogen the radius of 1.09 Å.
   Without ``potentiostat``, it runs at constant charge with no excess electrons; give ``target_potential`` for a constant-potential calculation, as the script does.

Try running the script below to find the number of electrons necessary to equilibrate the work function to 4.2 eV:

.. literalinclude:: run-Au111.py

If you examine the output in 'Au111.txt', you'll see that the code varies the number of excess electrons until the potential is within tolerance of the 4.2 V target; the tolerance defaults to 0.005 V, and can be changed with the ``tol`` key of the ``potentiostat`` dict.
This process usually takes a few steps on the first image, but is faster on subsequent images of a trajectory, since the changes are less dramatic and the potential--vs--charge slope is retained from previous steps.
You should see that in net the routine added only about 0.006 electrons to the simulation cell, as compared to a charge-neutral simulation, in order to achieve the desired work function.

You'll notice that the output in 'Au111.txt' contains additional information, as compared to a charge-neutral simulation.
Once the calculation has converged, the equilibration is reported in a single block, which states the potential and the number of excess electrons it settled on, the slope of potential versus charge regressed over the last ``slope_regression_depth`` attempts (four by default, or as many as the equilibration took), and the apparent capacitance that slope corresponds to.
It is followed by the Legendre-transformed energies::

    Potential found to be 4.19521 V (with +0.00645 excess electrons, attempt 4/10).
    Slope estimated from the last 4 attempts is -67.9098 V/electron,
    or apparent capacitance of 3.2731 muF/cm^2
    Potential is within tolerance. Equilibrated.

    Legendre-transformed energies (grand potential, Omega = E - N mu):
      N (excess electrons):      +0.006453
      mu (-potential):           -4.195210  # eV
      (Grand) free energy:      -21.289829
      (Grand) extrapolated:     -21.267235

These Legendre-transformed energies, `\Omega = E - N \mu`, are written into any ASE trajectories created, and are the quantity returned by ``atoms.get_potential_energy()``.
As discussed in :ref:`grand-potential-energy`, these grand-potential energies are consistent with the forces in the grand-canonical scheme, and are thus compatible with methods such as saddle-point searches (*e.g.*, NEB) and energy optimizations (*e.g.*, BFGSLineSearch).
If you'd rather have it output the traditional canonical energies to the ``get_potential_energy`` methods, set ``grand_output`` to ``False`` in the ``potentiostat`` dict.
The number of excess electrons and the electrode potential are also available as ASE properties, in ``calc.results['excess_electrons']`` and ``calc.results['electrode_potential']``.
The script keeps the ``SJM`` object in the variable ``sjm``, which gives access to the running calculation: ``sjm.excess_electrons`` and ``sjm.electrode_potential`` give the same two numbers directly---the latter in V, on the vacuum scale, for the last converged SCF loop.

In the last line of the script above, ``sjm.write_sjm_traces(path='sjm_traces.out')`` wrote out some additional information in the form of 'sjm traces', into the directory ``sjm_traces.out``.
These are traces of the electrostatic potential, the background charge, and the solvent cavity across the `z` axis (that is, `xy`-averaged).
It's a good idea to take a look at these, and perhaps make a plot for each, when running a simulation with a new system so you can be sure it is behaving as you expect.
The three files that are created are:

potential.txt:
 Electrostatic potential averaged over `xy` and referenced to the system's Fermi Level.
 The outer parts should correspond to the respective work functions and the potential at the top boundary should correspond to the potential set in the input.

cavity.txt:
 The shape of the implicit solvent cavity averaged over `xy`.
 The `xy`-averaged dielectric constant in the unit cell is 1 + (epsinf - 1) times this cavity function.

background_charge.txt:
 The jellium counter charge averaged over `xy`, as a charge density in e/Bohr^3; it integrates to minus the number of excess electrons.
 It is therefore orders of magnitude smaller than the cavity function in this example, and the plot below scales it to the same range to show its shape.

.. Note:: Alternatively, ``sjm.write_sjm_traces(style='cube')``
          can be used to write cube files on the 3-D grid, instead of traces.

Both ``write_sjm_traces()`` and ``electrode_potential`` need a calculation to have been run, and ``write_sjm_traces()`` must be called on all MPI ranks, as any other calculator method.
After reading a ``.gpw`` file, the ``SJM`` object is ``calc.params.extensions[0]`` again.

Our traces showing the solvent region and the jellium region are plotted below.
The black dots indicate the atoms' `z` positions.
We see that both the solvent and the jellium are starting above the highest atom, so all is good.
If you do see solvent appearing in a region where you do not want it, you can block it with ghost atoms or a boundary plane, as described in the :class:`~gpaw.solvation.sjm.SJMPower12Potential` docstring.
If you see solvent in the metal region, you could increase the atomic radii as described in the same docstring.

.. image:: traces-Au111.png
           :width: 600 px
           :align: center

(The script used to generate this plot is here: :download:`plot-traces.py`.)

Relaxing the slab and adsorbate
===============================

Next, let's perform a structural optimization at a constant potential.
Use the same system as the previous example, but add an H atom to the surface as an adsorbate.

.. literalinclude:: run-Au111-H-seq.py

If you examine the output, you should see this behaves the same as any other relaxation job, except the potential is constant, rather than the number of electrons.

A faster route: simultaneous potential and structure optimization
=================================================================

It is often faster to optimize the structure and the potential simultaneously.
That is, instead of waiting until the potential is perfectly equilibrated before taking an ionic step, you can take an ionic step and adjust the number of electrons simultaneously.
To do this, set a loose potential tolerance and set the ``always_adjust`` key to ``True``.
An example follows.

.. literalinclude:: run-Au111-H-sim.py

Note that at the end of the script, we tightened the criteria back up, in order to guarantee precise results consistent with a normal, sequential calculation (assuming it found the same local minimum).

The plot below compares the "sequential" and "simultaneous" optimization approaches.
The open circles are potential-equilibration steps, while the filled circles represent ionic steps (in which the atoms are moved).
In sequential mode, it always makes sure the potential is within tolerance before taking an ionic step, and you can see that it typically takes an extra DFT calculation (or two) before moving the atoms.
In simultaneous mode, it moves the atoms with much higher frequency, and only when the potential gets very far from the target does it pause to equilibrate.
Generally, this results in faster total convergence.
At the end of the simultaneous optimization, we switched back to sequential mode to ensure that the final result was at the expected potential; this can add a few extra steps---here the structure was already converged within both of the tightened tolerances, so it added none---but overall the computational savings are typically worth it.

.. image:: simultaneous.png
           :width: 600 px
           :align: center

(Plot script: :download:`plot-seq-sim.py`.)

Finding a barrier (NEB/DyNEB)
=============================

Perhaps the most-desired use of constant-potential calculations is to find barriers at a specified potential.
This avoids a well-known problem in canonical electronic structure calculations, where the work function (that is, the electrical potential) can change by 1--2 V over the course of an elementary step.
Such a potential change is obviously much greater than the experimental situation, where potentiostats hold the potential to a tolerance many orders of magnitude smaller.
The SJM method allows one to calculate a reaction barrier in a manner where all images in the trajectory have identical work functions, down to a user-specified tolerance.

The discrepancy between a canonical and grand-canonical simulation is most pronounced in reactions involving the creation or destruction of an ion, as the ion collides with the constant-potential surface.
Unfortunately, simulating even a simple reaction like the Volmer reaction---where a proton in the solution reacts to form an adsorbed hydrogen atom---is not possible in this tutorial, as a relatively large water layer---that is, larger than the one molecule we are using!---is needed in order to stabilize the solvated proton in the initial state.
Therefore, in this tutorial, we'll do a much simpler reaction: the diffusion of our adsorbed H atom from one type of surface site to another.
Note that we should expect very little difference in the number of excess electrons in order to equilibrate the initial, transition, and final state; nevertheless, it allows us to demonstrate the use of the nudged elastic band (NEB) method with SJM.

You will need to create the final state before you run the NEB; you can go through the exercise of creating this yourself, or you can use our script (:download:`run-Au111-H-hollow.py`).
After you have created the final state, you can run the NEB as below; note here we use the DyNEB method which tends to save a large amount of computational time for these sorts of barriers.
Note also that we reduce the degrees of freedom in this example, holding the water molecule fixed along with the bottom two layers of gold, so that the band follows the diffusing hydrogen rather than the soft rotations of the solvent molecule.

.. literalinclude:: run-neb.py

.. Note:: Each image needs its own ``SJM`` object, with its own ``cavity``,
          ``dielectric`` and ``interactions``, which is why ``make_calculator()``
          creates them all afresh for every image.
          Do not hand one ``SJM(...)`` object---or one set of solvent objects---to
          several calculators: each calculation keeps its state in them (the
          cavity on its grid, for instance), so the images would silently
          overwrite each other's, and ``sjm.change()``, ``sjm.potentiostat``
          and the like would only reach the calculator built last.

After a few dozen iterations, the band completes and we can make a plot of the barrier, using ``NEBTools``.
(Plot script: :download:`plot-neb.py`.)

.. image:: band.png
           :width: 600 px
           :align: center

Although the reaction modeled here is very simple and doesn't involve much need for a constant-potential method, it nonetheless allows us to demonstrate that SJM plays well with standard barrier-search methods, like the NEB.

You'll note that we used ``DyNEB`` instead of the regular ``NEB``.
Although ``NEB`` will also work fine, ``DyNEB`` is more efficient as it skips re-calculating individual images whose maximum force is already below the cutoff (of ``fmax=0.05``).
Note also that parallelizing a NEB over images is in general not a great strategy when using a grand-canonical method, because typically only one of the images might need potential equilibration on any individual force call of the NEB---and in parallel-image mode all of the images would need to wait for that individual image.
However, running in serial allows us to take advantage of the computational efficiency of the ``DyNEB`` method, which is not possible when parallelizing over images.

Constant-charge mode
====================

The SJM code can also run in constant-charge mode, where the user specifies the total number of electrons in the simulation.
This can be a fast way to calculate a system at several potentials---that is, if one does not care about the specific potentials, but just wants to span a range of potentials.
To use constant-charge mode, just specify the number of ``excess_electrons``---and no target potential---and calculate as normal::

    calc = GPAW(..., extensions=[SJM(solvation=...,
                                     potentiostat={'excess_electrons': 0.5})])


If your intent is to run in the constant-charge ensemble---like in the use of the charge-extrapolation scheme---you may want to output the canonical, rather than the grand-potential energies.
The canonical energies are consistent with the forces in constant-charge mode.
To accomplish this, set ``grand_output=False``, like::

    calc = GPAW(..., extensions=[SJM(solvation=...,
                                     potentiostat={'excess_electrons': ...,
                                                   'grand_output': False})])

To change the SJM settings, keep the ``SJM(...)`` object in a variable, ``sjm = SJM(...)``, and call its ``change()``: ``sjm.change(potentiostat={'excess_electrons': ...})`` changes the number of excess electrons, before the first calculation or later on without throwing away the wave functions, and the same method switches between constant-charge and constant-potential mode by setting or clearing ``target_potential``.
The settings are read off the potentiostat that runs in the calculation---``sjm.potentiostat.optimizer.target_potential``, ``sjm.potentiostat.optimizer.slope`` (the current estimate) and so on, which are its live values, and the number of excess electrons is ``sjm.excess_electrons``---but set them through ``change()``, which also makes the next calculation run the SCF loop again.
