======================================
Transport barriers and Voltage profile
======================================

You will calculate the energy barriers for transport of Li intercalated in the
graphite anode. You will examine how sensitive this barrier is to the
interlayer distance in graphite. You will also examine the energy of
intermediate states during the charge/discharge process. This will allow some
basic discussion of the voltage profile of the battery.

* Create initial and final structures for a NEB calculation, that will
  determine the transition state

  - If time permits you can study the influence of changing the interlayer
    graphite distance on the energy barrier.

* Create structures for a Li vacancy in :mol:`LiFePO_4` and a single Li
  in :mol:`FePO_4`

* Calculate the Li vacancy/insertion energies and compare them to the
  equilibrium potential

  - What can they tell you about the charge/discharge potential curves?

You will in general be provided less code than yesterday, especially
towards the end of this notebook.  You will have to use what you have
already seen and learned so far.

There will be some natural pauses while you wait for calculations to
finish.  If you do not finish this entire notebook today, do not despair.


Transport barrier of Li in graphite
===================================

You will now calculate the energy barrier for Li diffusion in the graphite
anode.  You will do this using the :class:`Nudged Elastic Band (NEB)
method <ase.mep.neb.NEB>`.

You can use your work from Day 2, but for simplicity you are advised to
load in the initial atomic configuration from file.

.. literalinclude:: li_barrier.py
   :end-before: snippet-final

You will now make a final structure, where the Li atom has been moved to a
neighbouring equivalent site.  The
:meth:`~ase.Atoms.get_positions`,
:meth:`~ase.Atoms.set_positions` and
:meth:`~ase.Atoms.get_cell` methods
are highly useful for such a task.  HINT: Displace the Li atom
`\frac{1}{n} (\vec{a}+\vec{b})`

.. code::

    final = initial.copy()
    ...
    ...

Visualize that you have made the final strcuture correctly.

.. code::

    from ase.visualize import view
    view(final)

Make a band consisting of 7 images including the initial and final.

.. literalinclude:: li_barrier.py
   :start-after: snippet-neb1
   :end-before: snippet-neb2

It this point ``images`` consist of 6 copies of ``initial`` and one entry of
``final``. Use the ``NEB`` method to create an initial guess for the minimum
energy path (MEP). In the cell below a simple interpolation between the
``initial`` and ``final`` image is used as initial guess.

.. literalinclude:: li_barrier.py
   :start-after: snippet-neb2
   :end-before: snippet-constraint-gpaw

Add a ``view(images)`` line to your script and run it so that you can
visualize the NEB images.

It turns out, that while running the NEB calculation, the largest amount
of resources will be spend translating the carbon layer without any
noticeable buckling.  You will thus
:mod:`constrain <ase.constraints>` the
positions of the carbon atoms to save computational time.

Each image in the NEB requires a unique calculator.

This very simple case is highly symmetric.  To better illustrate how the
NEB method works, the symmetry is broken using the
:meth:`ase.Atoms.rattle` method.

.. literalinclude:: li_barrier.py
   :start-after: snippet-constraint-gpaw
   :end-before: snippet-initial-final

Start by calculating the energy and forces of the first (``initial``) and
last (``final``) images as this is not done during the actual NEB
calculation.

Note, that this can take a while if you opt to do it inside the notebook.

.. literalinclude:: li_barrier.py
   :start-after: snippet-initial-final
   :end-before: snippet-optimize

You can run the NEB calculation by running an optimization on the NEB
object the same way you would on an atoms object.  Note the ``fmax`` is
larger for this tutorial example than you would normally use.

.. literalinclude:: li_barrier.py
   :start-after: snippet-optimize

Submit the calculation to the HPC cluster.  Do this by first building a
complete script in the cell below using the cells above (minus the
``view()`` commands). Make sure the cell runs and then interrupt the kernel.

.. code:: bash

    $ mq submit NEB.py -R 8:1h  # submits the calculation to 8 cores, 1 hour
    $ ...
    $ mq ls
    $ ...
    $ tail neb.log

You can move on while you wait for the calculation to finish.

Once the maximum force (``fmax``) in the log is below 0.1, the calculation
is finished.  Load in the full trajectory.

You will use the ``ase gui`` to inspect the result.  The below line reads in
the last 7 images in the file.  In this case the MEP images.

.. code::

    $ ase gui neb.traj@-7:

In the GUI use :menuselection:`Tools --> NEB`.

Now inspect how the TS image has developed.

.. code::

   $ ase gui neb.traj@3::7

For more complicated MEP's, use the :ref:`ase:climbingimage` method to
determine the transition state.  Why is it not required here?


Bonus
=====

You will now study the influence of changing the interlayer graphite
distance on the energy barrier.  Due to the high degree of symmetry, this
can be done easily in this case.  Load in the initial state (IS) and
transition state (TS) images from the converged MEP.

.. literalinclude:: li_barrier_2.py
   :end-before: snippet-barrier

Now calculate the energy of the initial state (IS) image and the
transition state (TS) image using the
:meth:`~ase.Atoms.get_potential_energy` method.

.. literalinclude:: li_barrier_2.py
   :start-after: snippet-barrier
   :end-before: snippet-strain

New change the graphite layer distance by changing the the size of the
unit cell in the *z* direction by ±3 %. and use the same calculator
object as you did above and calculate the potential energy of the
compressed initial and final state.

.. literalinclude:: li_barrier_2.py
   :start-after: snippet-strain
   :end-before: snippet-gpaw

How does the energy barrier change?


:mol:`FePO_4` with one Li
=========================

You will now calculate the energy gain of adding a single Li atom into
the :mol:`FePO_4` cell you made on Day 3. This corresponds to a charge of 25
%. You can compare this energy to the equilibrium potential.

Start preparing a new Python script (say, ``fepo4_1li.py``) and load in the :mol:`FePO_4` structure you wrote to file on in a previous
exercise and add Li.  Assume that the cell dimension remain unchanged.

.. literalinclude:: fepo4_1li.py
   :end-before: snippet-li

Add a Li atoms at `(x,y,z)=(0,0,0)`:

.. literalinclude:: fepo4_1li.py
   :start-after: snippet-li
   :end-before: snippet-gpaw

Now finish the script:

* calculate the energy with the BEEF functional
* write result to a ``.traj`` file:
  ``from ase.io import write`` and
  ``write('fepo4_1li_out.traj', fepo4_1li)``
* calculate a BEEF-ensemble

and submit the job to the queue.

Once the calculation is finished, you are ready to calculate the energy
gained by intercalating a single Li ion into the cathode.  Start by
loading in the relevant reference structures and obtain the potential
energies.  This should not require any new DFT calculations.

.. literalinclude:: fepo4_1li.py
   :start-after: snippet-results
   :end-before: snippet-results-end

Calculate the energy of intercalting a single Li in the :mol:`FePO_4` cell.
How does this energy compare with the equilibirum potential?  What can it
tell you about the charge/discharge potential curves?


Bonus: :mol:`LiFePO_4` with one vacancy
=======================================

If time permits, you will now do a similar calculation but this time with
:mol:`LiFePO_4` contraining one vacancy.  Once again you should assume that
the cell dimension remain unchanged compaired to :mol:`LiFePO_4`.

There are numerous ways to obtain this structure.  You can get
inspiration from the way :mol:`LiFePO_4` was made on Exercise day 3.  Use
``del atoms[index]``, the :meth:`ase.Atoms.pop` method
or even the GUI to delete an atom and save the structure afterwards.

When you have made your script (say ``lifepo4_vac.py``), submit it to the
HPC cluster.  Once the calculation has finished you are ready to
calculate the energy cost of creating a Li vacancy in the fully lithiated
:mol:`LiFePO_4`. Start by loading in the relevant reference structures and
obtain the potential energies.  This should not require any calculations.

.. literalinclude:: lifepo4_vac.py
   :start-after: snippet-results
   :end-before: snippet-results-end

How does this energy compare with the equilibirum potential?  What can it
tell you about the charge/discharge potential curves?


Bonus
=====

Calculate the error estimates of the energy for the added Li atom and
vacancy formation using the ensembles.
