.. _batteries:

=========
Batteries
=========

In this exercise we will study the anode and cathode material of a Li-ion
battery.  The cathode material will be :mol:`LiFePO_4` a typical cathode
material in rechargeable Li-ion batteries.  The anode will be graphite.

The first day we start out soft by calculating the intercalation energy of Li
in graphite while we learn the methods and workflow using ASE and GPAW. The
second day will be about determining the equilibrium potential of a
:mol:`LiFePO_4/C` battery, we will also use a Bayesian approach to estimate the DFT
error we expect on this important value. On the final day we will determine
important battery characteristics such Li transport barriers and the voltage
profile.

Tools used:

* Structure creation and modification with ASE

* Unit cell relaxation

* Bayesian error estimation

* Nudged Elastic Band (NEB) calculations for estimating Li migration barriers

.. toctree::

   batteries1
   batteries2


Part 3: Transport barriers and voltage profile
==============================================

:download:`batteries3.ipynb`, :download:`NEB_init.traj`

You will calculate the energy barriers for transport of Li intercalated in the
graphite anode. You will examine how sensitive this barrier is to the
interlayer distance in graphite. You will also examine the energy of
intermediate states during the charge/discharge process. This will allow some
basic discussion of the voltage profile of the battery. The notebook is
``batteries3.ipynb``.

* Create initial and final structures for a NEB calculation, that will
  determine the transition state

  - If time permits you can study the influence of changing the interlayer
    graphite distance on the energy barrier.

* Create structures for a Li vacancy in :mol:`LiFePO_4` and a single Li
  in :mol:`FePO_4`

* Calculate the Li vacancy/insertion energies and compare them to the
  equilibrium potential

  - What can they tell you about the charge/discharge potential curves?
