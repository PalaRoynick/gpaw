
=============================================
Inclusion of excitonic effects and Discussion
=============================================


Now, we will finally visualize the absorption spectra we have calculated. The script below is setup for plotting and saving the absorption spectra for BN with 24x24x1 k-points. First we load the x, y, and z components and plot all three curves in different colors.


.. literalinclude:: solution4.py
   :start-after: # snippet-plot-rpa-start
   :end-before: # snippet-plot-rpa-end


Now open the saved figure and inspect the absorption spectra. Does it look as you expected? Can you guess the band gap from the absorption spectra and which of the band gaps calculated on day 2 does it match?

Now plot the absorption spectra for the other k-point meshes you calculated and compare. You could for instance modify the script above to plot, in different colors, all spectra in the same plot for comparison. Would you say that the calculation is converged?

Finally, talk with the other groups and compare your absorption spectra with their absorption spectra. Is the same number of k-points needed to obtain same degree of convergence for the different materials? Is there any difference for the 2D material (Boron-Nitride)?

For a more realistic description one would have to include excitonic effects (i.e. the electron-hole interaction).
In the next example, we will calculate the absorption spectrum using the Bethe-Salpeter equation (BSE). For theoretical understanding you can read the brief description provided as in the link below: https://gpaw.readthedocs.io/documentation/bse/bse.html

We start by calculating the ground state density and diagonalizing the resulting Hamiltonian. The last line in the script creates a .gpw file which contains all the informations of the system, including the wavefunctions.


.. code::

   atoms = bulk(...)
   pw_cut_off = ...
   kpts_grid = (..., ..., ...)
   gamma = ...


.. literalinclude:: solution4.py
   :start-after: # snippet-bse-groundstate-start
   :end-before: # snippet-bse-groundstate-end


Below we will set up the Bethe-Salpeter Hamiltonian in a basis of valence bands and conduction bands (hint: the number of valence and conduction bands should be in the single digits).
However, the screened interaction that enters the Hamiltonian needs to be converged with respect the number of unoccupied bands. Next we calculate the dynamical dielectric function using the Bethe-Salpeter equation. The imaginary part is proportional to the absorption spectrum. We will calculate the dielectric function within the Random Phase Approximation (with the same convergence parameters for comparison).


.. code::

   ecut = ...
   nbands = ...
   valence_bands = np.range(..., ...)
   conduction_bands = np.range(..., ...)
   nbands = ...

Hint: The end of the valence band range should be the same as the start of 
conduction band range.

.. literalinclude:: solution4.py
   :start-after: # snippet-bse-spectrum-start
   :end-before: # snippet-bse-spectrum-end


Note the .csv output files that contains the spectre. The script also generates a .dat file that contains the
BSE eigenvalues. The spectrum essentially consists of a number of peaks centered on the eigenvalues. Below we will
plot it along with the RPA spectrum for comparison.


.. literalinclude:: solution4.py
   :start-after: # snippet-plot-bse-start
   :end-before: # snippet-plot-bse-end


Note: The parameters that need to be converged in the calculation are the k-points in the initial ground state calculation. In addition the following keywords in the BSE object should be converged: the plane wave cutoff, the numbers of bands used to calculate the screened interaction, the list of valence bands and the list of conduction bands included in the Hamiltonian. You can find an example calculation for Silicon in the link below: https://gpaw.readthedocs.io/tutorialsexercises/opticalresponse/bse_tutorial/bse_tutorial.html
