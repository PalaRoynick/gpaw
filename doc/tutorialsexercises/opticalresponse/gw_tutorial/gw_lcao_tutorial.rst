.. _gw_lcao_tutorial:

==================================================================================
Quasi-particle spectrum in the GW approximation from LCAO wave functions: tutorial
==================================================================================

For a brief introduction to the GW theory and the details of its
implementation in GPAW, see :ref:`gw_theory`.

More information can be found here:

    \F. Hüser, T. Olsen, and K. S. Thygesen

    `Quasiparticle GW calculations for solids, molecules, and
    two-dimensional materials`__

    Physical Review B, Vol. **87**, 235132 (2013)

    __ https://prb.aps.org/abstract/PRB/v87/i23/e235132
    
Additionally, before getting started using LCAO wave functions in the GW approximation,
it would be beneficial to become familiar with G0W0 using GPAW, see :ref:`gw tutorial`.

Quasi-particle spectrum of bulk diamond
=======================================


Groundstate LCAO calculation
----------------------------

First, we need to do a regular LCAO groundstate calculation, and save all of the
wave functions. The basis-set is chosen using the ``basis`` keyword, and  
``nbands`` is set to ``nao``, reflecting the maximum nbands value that can be used in LCAO,
which is the same number of bands as there are atomic orbitals. 

.. literalinclude:: C_lcao_groundstate.py

G0W0 calculation using LCAO basis functions
-------------------------------------------

We can now set up the G0W0 calculation similar to the PW case, passing the LCAO groundstate 
calculation ``C_lcao_groundstate.gpw`` to the G0W0 calculator, which is then internally converted
to PW. 
However, there are a few major differences. ``ecut_extrapolation`` is disabled,
because the convergence wrt. G-bands and chi0 bands is unknown with LCAO-basis set,
whereas in PW mode, the number of bands can be just directly chosen from
the plane wave cut off.

.. literalinclude:: C_lcao_gw.py

The results are stored in ``C-g0w0-lcao_results.pckl``.  

Two-dimensional G0W0+SOC band structure
----------------------------------------

For a two-dimensional system, truncate the Coulomb interaction in the
non-periodic direction and use the two-dimensional treatment of the
Gamma-point integral.  The analytic ``q0_correction`` substantially improves
k-point convergence.  Cutoff extrapolation is disabled for an LCAO ground
state because the number of unoccupied states is limited by the basis.

The following standalone example calculates the C2DB monolayer MoS2 structure
in LCAO mode with a ``dzp`` basis.  A copy of the wave functions is converted
to their plane-wave representation for G0W0, while the resulting corrections are Fourier
interpolated with the original LCAO Hamiltonian.  Finally, the atomic SOC
Hamiltonian is added and diagonalized.  Corrections for *all* LCAO bands are
required by the interpolation.

The numerical settings shown are representative rather than universally
converged.  In particular, the k-point mesh, response cutoff, basis and vacuum
spacing must be converged for the desired observable.

One script calculates and converts the three ground states:

.. literalinclude:: MoS2_lcao_gs.py

Three separate scripts then run G0W0.  The 12x12 script includes both response
cutoffs, while the 18x18 and 24x24 scripts use 200 eV:

.. literalinclude:: MoS2_gw_12.py

.. literalinclude:: MoS2_gw_18.py

.. literalinclude:: MoS2_gw_24.py

The four resulting points are 12x12, 18x18, and 24x24 at 200 eV, together
with 12x12 at 300 eV.

The interpolation and plotting are performed separately, using the ground
state and G0W0 result written by the calculation above:

.. literalinclude:: MoS2_lcao_gw_soc_plot.py

.. image:: MoS2-gap-convergence.png
   :width: 100%
   :align: center

.. image:: MoS2-24x24-200-bands.png
   :width: 100%
   :align: center

The PBE+SOC and G0W0+SOC bands are referenced to the G0W0 SOC valence-band
maximum.  Notice that a spin-paired calculation produces twice as many SOC
spinor bands as scalar bands; consequently the number of occupied SOC bands
equals the number of valence electrons, not half that number.

GW self-energy for Diamond (C)
------------------------------

We plot the results stored in ``C-g0w0-lcao_results.pckl`` as the imaginary and real part of the 
GW self-energy :math:`\Sigma(\omega)` depending on the frequency using the script 
:download:`plot_C_lcao_gw.py`. We compare the LCAO results to PW ones computed 
using :download:`C_pw_groundstate.py` and :download:`C_pw_gw.py`. 

.. figure:: C_Im.png
   :align: center
   :width: 50%

.. figure:: C_Re.png
   :align: center
   :width: 50%
