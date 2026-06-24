.. _excited states:

==============
Excited States
==============

This exercise goes through how to calculate some of the most important 
properties for photovoltaics. The first exercise is about how to 
calculate the bandstructure and how different functionals affect it. 
In the second exercise the GW approximation is used to get more accurate 
bandgaps at a higher computational cost. The third part involves 
calculating the absorption spectrum with the random phase approximation (RPA). 
In the final exercise the excitonic effects are include in the absorption 
spectrum calculations.


Part 1: Setup of the structure and bandstructure calculations
=============================================================

The tutorial shows how to set up the material and how to calculate a converged band structure.

* Set up the atomic structure and optimize its geometry

* Calculate the band gap, band gap position,
  and band structure

* Compare the performance of different exchange correlation
  functionals

.. toctree::
    es1


Part 2: Quasiparticle bandgap
=============================

.. toctree::
   es2

This tutorial teaches how to set up
calculations to find the quasiparticle band gap using GW
approximation.

* Understanding the GW approximation.

* First basic tests on the convergence of quasiparticle
  spectrum.

* Write/submit batch jobs


Part 3: Absorption spectrum
===========================

.. toctree::
    es3

The tutorial teaches how to set up
calculations of the dielectric function to find the absorption spectrum.

* Understanding the dielectric function

* First basic tests on the convergence of absorption
  spectra in the random phase approximation

* Write/submit batch jobs


Part 4: Excitonic effects and Discussion
========================================

.. toctree::
    es4

In the final tutorial the results for different
materials are plotted and discussed. Next the absorption spectra
is calculated including the excitonic effects (Bethe-Salpeter formalism).

* Understanding the excitonic effects and Bethe-Salpeter formalism.

* Basic tests on convergence of the absorption spectra.

* Write/submit batch jobs

Special care is taken of the convergence.
