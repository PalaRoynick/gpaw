

==============================
Calculating absorption spectra
==============================


In this exercise we are going to calculate the absorption spectra of the material you have considered so far.
In general, the absorption spectrum is given by the imaginary part of the macroscopic dielectric function. This means that our computational task is to calculate the macroscopic dielectric function for our material, and then plot the imaginary part. For more information about the dielectric function please consult:

https://gpaw.readthedocs.io/tutorialsexercises/opticalresponse/dielectric_response/dielectric_response.html

To calculate the dielectric function, we will use the Random Phase Approximation (RPA) correlation energy (so we say we calculate the absorption spectrum within the random phase approximation).
(Details about RPA to calculate the total energy can be found here: https://gpaw.readthedocs.io/documentation/xc/rpa.html#rpa)

As discussed earlier, it is of greatest importance to make sure our calculations are converged. Since the convergence parameters is not necessarily the same for band gaps and RPA absorption spectra (and any other material property) with respect to for instance plane-wave cut-off or k-points, we need to do do a new ground state calculation with parameters. For RPA absorption spectra we will here look at the number of bands included in the calculation and the k-point mesh. We will therefore restart from the previous ground state file, and update some of the parameters. First we will look at a too rough k-point mesh and then do more calculations with a finer k-point mesh. In this way we can see the importance of converging the calculations.

The new ground state is calculated below. For BN use (24,24,1) k-points while for the bulk materials use (12,12,4) k-points.


.. code::

   kpts_grid = (..., ..., ...)


.. literalinclude:: solution3.py
   :start-after: # snippet-rpa-groundstate-start
   :end-before: # snippet-rpa-groundstate-end


Now that we have obtained the ground state, it is time to calculate the dielectric function. This is done below where we first initialize the parameters needed for an RPA calculation, then calculate the dielectric function, and finally obtain the polarizability. Note that one also would have to converge the parameters initialized below, for a fully converged study, however in the interest of time and computational power we will not consider this here. The principle is however exactly the same, as you will see with the different k-point meshes we will employ here.

These calculations are both time-wise and memory-wise heavier than what you previously encountered. Therefore we will need to submit these calculations to the databar so they can run over night. Open a new SSH terminal, edit and copy the below code into a script format (.py file), and submit the calculations from the terminal using the following command:

:code:`mq submit -R 8:15h script.py`

This will submit the script with the name "script.py" to 8 cores with a maximum time of 15 hours.


.. literalinclude:: solution3.py
   :start-after: # snippet-rpa-polarizability-start
   :end-before: # snippet-rpa-polarizability-end


Now it is time to do more calculations with finer k-point mesh to see the importance of converging our calculations. Repeat the above calculations with the following parameters:

For BN try: (24,24,1), (40,40,1), and (60,60,1) k-points
For bulk materials try e.g.: (12,12,4), (18,18,6), and (24,24,8) k-points
(Remember to use a meshes according to the length of lattice vectors.)

For the calculations with more k-points you probably want to combine the (new) ground state calculation and the calculation of the absorption spectrum into one script which you will submit over night.
