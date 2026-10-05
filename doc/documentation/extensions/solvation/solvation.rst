.. module:: gpaw.solvation
.. _continuum_solvent_model:

=============================
Continuum Solvent Model (CSM)
=============================

Theoretical Background
======================

A continuum solvent model (CSM), as described in [HW14]_, can be used as a GPAW extension.
The model is based on a smooth cavity `g(\br)` represented on a real-space grid, although the CSM now works in both grid and plane-wave modes.
The electrostatic screening effect of the solvent is modeled as a spatially varying permittivity (relative dielectric constant) `\epsilon(\br)`, which is related to the cavity by

.. math:: \epsilon(\br) = 1 + (\epsilon_{\infty} - 1) g(\br).

The bulk static dielectric constant `\epsilon_{\infty}` is taken from experiment for the solvent in use (e.g., `\epsilon_{\infty} \approx 80` for water at room temperature).
The electrostatic energy for the total charge distribution `\rho(\br)` of the solute (electron density + atomic nuclei) is calculated as

.. math:: W_{\mathrm{el}} = \frac{1}{2} \int \rho(\br) \phi(\br) \mathrm{d}\br

where `\phi(\br)` is the electrostatic potential in the presence of the dielectric.
It is obtained by solving a generalized Poisson equation including the spatially varying permittivity:

.. math:: \vec{\nabla} \cdot \Big(\epsilon(\br) \vec{\nabla} \phi(\br)\Big) = -4\pi \rho(\br)

In grid mode, this is solved numerically with a finite-difference multi-grid solver as outlined in [SSS09]_.
In plane-wave mode, a preconditioned conjugate-gradient Poisson solver is used.

As not only the electrostatic interaction but also cavity formation and short-range interactions between the solute and the solvent contribute to the solvation Gibbs energy, additional short-range interactions can be included in the calculation.
The ansatz chosen in [HW14]_ is to describe the cavity formation energy and all short-range interactions as a term proportional to the surface area of the cavity.
The surface area is not well defined for a smooth cavity `g(\br)`.
The approach of [IBR98]_ is taken, where the surface area `A` is calculated from the gradient of the cavity:

.. math:: A = \int \| \vec{\nabla} g(\br) \| \mathrm{d}\br

The cavity formation energy and short-range interaction are then given as `G_{\mathrm{cav+sr}} = \gamma A` with a proportionality constant `\gamma` that has the dimensions of a surface tension.

The cavity is constructed from an effective repulsive potential `u(\br)` via the Boltzmann equation

.. math:: g(\br) = \exp \Big(-\frac{u(\br)}{k_{\mathrm{B}} T}\Big).

The effective potential is taken as the repulsive part of the Lennard-Jones potential and constructed from the positions `\mathbf{R}_a` of the atomic nuclei and a set of atomic radii `R_a^{\mathrm{vdW}}` as

.. math:: u(\br) = u_0 \sum_a \Big(\frac{R_a^{\mathrm{vdW}}}{\|\br - \mathbf{R}_a \|} \Big)^{12}.

The single free parameter `u_0` describes the strength of the repulsion at the atomic radii.
It controls the size of the cavity and can be fitted to experimental partial molar volumes (per atom) `\bar{v}_M^{\infty}` at infinite dilution via the compressibility equation

.. math:: \int (1 - g(\br)) \mathrm{d}\br = \bar{v}_M^{\infty} - \kappa_T k_{\mathrm{B}} T,

where `\kappa_T` is the isothermal compressibility of the bulk solvent (usually negligible compared to `\bar{v}_M^{\infty}` except for very small molecules).
For water as a solvent, Bondi van der Waals radii with a modified value for hydrogen (1.09 Å) are a good choice for the atomic radii of the effective repulsive potential [HW14]_.

Altogether, the model has three parameters.
The static dielectric constant `\epsilon_{\infty}` of the solvent is taken directly from experimental data.
The parameter `u_0` is fitted to experimental volumes.
Afterwards, the parameter `\gamma` can be fitted to experimental solvation Gibbs energies.
Parameters for water (at room temperature) as a solvent are given in [HW14]_.
A preliminary parameterization for dimethylsulfoxide (DMSO) is also given there.
The accuracy for aqueous solvation Gibbs energies compared to experimental values is about 5 kJ per mole for neutral molecules and 13 kJ per mole for cations.
It is not recommended to apply the model to anions in water without further modification as short-range interactions between anions and water are not well represented using the parameters optimized for neutral molecules [HW14]_.

A lot of the parts that make up the model (cavity, dielectric function, Poisson solver, non-electrostatic interactions) can be replaced easily by alternative implementations as they are represented as Python classes in the source code.
Some alternative models already exist in the GPAW source code, yet they are not well tested and therefore not recommended for production use.
Nevertheless, they can serve as an example of how to add your own solvation interactions/models to GPAW.
Also refer to Section III of [HW14]_ for a more precise description of the general framework of continuum solvent models with a smooth cavity employed in the GPAW implementation.
In the following section, the usage of the model described above is documented in terms of a use case.

Example: Ethanol in Water
=========================

The calculation of the solvation Gibbs energy of ethanol in water is demonstrated.
For simplicity, the solvation Gibbs energy is calculated as the total energy difference of a calculation with the continuum solvent model and a gas-phase calculation using a fixed geometry of the ethanol molecule.
In principle, a relaxation of the solute can be done also with the continuum solvent model.
The following annotated script demonstrates how to perform the calculation using the continuum solvent model:

.. literalinclude:: ethanol_in_water.py

The calculated value for the solvation Gibbs energy should be about -4.5 kcal per mole.
The experimental value is -5.0 kcal per mole [KCT06]_.

There is also a helper function to use the solvation parameters for water as in the above example:

.. literalinclude:: kwargs_factory.py

.. _solvation_dipole:

Dipole Corrections
==================

Dipole corrections work with the variable-dielectric field that is inherent to continuum solvation calculations, in both plane-wave and grid mode.
When an implicit solvent is present, the correction is made by creating a neutrally-charged region in the shape of a Gaussian derivative, then solving the generalized Poisson equation for the potential field resulting from this charge, with the same variable-dielectric field as the actual problem.
The potential from this dipole-layer calculation is superimposed on top of the potential field produced from the generalized Poisson equation that was run on the atomic system, to produce a decoupled, dipole-corrected potential profile with flat slopes in both the top and bottom regions of the cell, and meaningful top and bottom work functions.
This also enables the potential to be inferred in the :ref:`Solvated Jellium Method<sjm>`.
The corrected profile is propagated through the SCF cycle, ultimately producing an electronic structure that is dipole corrected.
Note the dipole correction is only implemented in the z direction; e.g., the typical slab + vacuum system where the slab extends in the x and y directions, and the vacuum padding is in the z direction.

An example is shown below for the same sodium-on-aluminum-slab system used in the :ref:`dipole-correction tutorial<dipole>`.
The figure below shows the atomic system in use.
The blue shaded region shows where the implicit solvent exists, with the color intensity proportional to solvent cavity.

.. image:: csm-overview.png
           :width: 600 px
           :align: center

If we run a simulation without a dipole correction, we get a potential profile that has a non-zero slope in the vacuum region, as shown below.

.. image:: csm-potential.png
           :width: 600 px
           :align: center

The dipole correction works by creating a charge distribution in the shape of a Gaussian derivative, as shown by the blue-shaded region below.
The potential profile that the Poisson equation produces from this charge profile, in the same solvent cavity as the atoms, is also shown in the figure.
This potential field is added to the physical atomic potential field, and fed through the SCF cycle.

.. image:: csm-layer.png
           :width: 600 px
           :align: center

The final, corrected potential profile is shown below, giving flat slopes in the vacuum region and unambiguous values for the top and bottom work functions, shown on the plot.

.. image:: csm-corrected.png
           :width: 600 px
           :align: center

To turn on the dipole correction, use the ``dipolelayer`` keyword::

    calc = GPAW(...,
                extensions=[Solvation(...,
                                      dipolelayer=True)])

The cell must be non-periodic along *z* (``atoms.pbc = (True, True, False)``), even in plane-wave mode.
(This is the same requirement as in the standard dipole correction.)

You can adjust the geometry of the Gaussian derivative by instead supplying a :class:`~gpaw.solvation.GaussianDerivativeDipole`, like ``dipolelayer=GaussianDerivativeDipole(center=1.1, width=0.4)``, or a dictionary of the same keywords, like ``dipolelayer={'center': 1.1, 'width': 0.4}``.
To use a different dipole correction, which can be written in your own script, supply an object with ``build()`` and ``todict()`` methods; see :meth:`~gpaw.solvation.GaussianDerivativeDipole.from_param`.

Legacy GPAW
===========

The legacy-GPAW calculator, ``SolvationGPAW``, is still available from ``gpaw.old.solvation``, but new calculations should use the ``Solvation`` extension described above.

Docstrings
==========

.. autoclass:: gpaw.solvation.Solvation

.. autoclass:: gpaw.solvation.GaussianDerivativeDipole
   :members: from_param

References
==========

.. [HW14] A. Held and M. Walter,
           :doi:`Simplified continuum solvent model with a smooth cavity based on volumetric data <10.1063/1.4900838>`,
           *J. Chem. Phys.* **141**, 174108 (2014).

.. [SSS09] V. M. Sanchez, M. Sued and D. A. Scherlis,
            :doi:`First-principles molecular dynamics simulations at solid-liquid interfaces with a continuum solvent <10.1063/1.3254385>`,
            *J. Chem. Phys.* **131**, 174108 (2009).

.. [IBR98] W. Im, D. Beglov and B. Roux,
            :doi:`Continuum solvation model: Computation of electrostatic forces from numerical solutions to the Poisson-Boltzmann equation <10.1016/S0010-4655(98)00016-2>`,
            *Comput. Phys. Commun.* **111**, 59 (1998).

.. [KCT06] C. P. Kelly, C. J. Cramer and D. G. Truhlar,
            :doi:`Aqueous Solvation Free Energies of Ions and Ion-Water Clusters Based on an Accurate Value for the Absolute Aqueous Solvation Free Energy of the Proton <10.1021/jp063552y>`,
            *J. Phys. Chem. B* **110**, 16066 (2006).
