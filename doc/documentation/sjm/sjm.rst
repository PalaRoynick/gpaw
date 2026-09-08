.. module:: gpaw.extensions.electrocatalysis.sjm
.. _sjm:

======================================================
Solvated Jellium (constant-potential electrochemistry)
======================================================

Overview
========

The Solvated Jellium method (:class:`~gpaw.extensions.electrocatalysis.sjm.SJM`) is a simple method for the simulation of electrochemical interfaces in DFT.
A full description of the approach can be found in [Kastlunger2018]_.
The method allows you to control the simulated electrode potential (manifested as the topside work function) by varying the number of electrons in the simulation; calculations can be run in either constant-charge or constant-potential mode.
SJM is attached to a regular GPAW calculator as an extension---:literal:`GPAW(..., extensions=[SJM(...)])`---and the calculator is then used just like any other; it returns energy and forces, but can do so at a fixed potential.
The number of excess electrons and the electrode potential are returned alongside them, as the :literal:`excess_electrons` and :literal:`electrode_potential` properties.
(Please see the note below on the Legendre-transform of the energy.)
The potential is controlled by a simple iterative technique; in practice if you are running a trajectory (such as a relaxation or nudged elastic band) the first image will take longer than a conventional calculation as the potential equilibrates, but the computational cost is much less on subsequent images; practically, we estimate the extra cost to be <50% compared to a traditional DFT calculation of a full trajectory.
For a practical guide on the use of the method, please see the :ref:`solvated_jellium_method` tutorial.


Theoretical background
======================

The philosophy of the solvated jellium method is to construct the simplest model that captures the physics of interest, without introducing spurious effects.
The solvated jellium approach consists of two components: jellium and an implicit solvent.
A schematic is shown below:

.. image:: overview.png
           :width: 600 px
           :align: center

In this figure, the jellium is shown by the hashed marks, while the implicit solvent is shown in the blue shaded region. Note that an explicit solvent (the water molecules) is also conventionally used in this approach, as the major purpose of the implicit solvent is not to simulate the solvation of individual species but rather to screen the net field.
A more detailed discussion of both of these components follows.


The jellium slab: charging
--------------------------

In a periodic system we cannot have a net charge; therefore, any additional, fractional electrons that are added to the system must be compensated by an equal amount of counter charge.
SJM accomplishes this by adding a smeared-out background charge in a fixed region of the simulation cell; in the figure above it is shown as the dashed region to the right of the atoms.
By default that region is a slab perpendicular to `z`, controlled by the :literal:`jelliumregion` keyword described under `Usage`_ below.
The number of electrons in the simulation is increased by an amount equal to the total charge of that region.
When you run a simulation, you should see that these excess electrons localize on the `+z` side (the right side in these figures) of the metal atoms that simulate the electrode surface, and not on the `-z` (left) side, which simulates the bulk.
This is accomplished by only putting the jellium region on one side of the simulation, and employing a dipole correction (included by default when you run SJM) to electrostatically decouple the two sides of the cell.

The figure below shows the difference in two simulations, one run at 4.4 V and one run at 4.3 V.
The orange curve shows where the potential drops off, and the blue curve shows where the electrons localize.

.. image:: traces.png
           :width: 600 px
           :align: center

The jellium region is conventionally thought of as a region of smeared-out positive charge, accompanied by a positive number of electrons.
However, the signs can readily be reversed, making the jellium region a smeared-out negative region accompanied by a reduction in the total number of electrons.
In this way, the same tool can be used to perturb the electrons in either a positive or negative direction, and thus vary the potential in either direction in order to find its target.
Note also that the jellium region does not overlap any atoms, separating this from approaches that employ a homogeneous background charge throughout the unit cell (in which spurious interactions can occur).
This is important to not distort the electronic structure of the atoms and molecules being simulated.
Additionally, note that the jellium is enclosed in a regular slab geometry in the figure above, but this need not be the case in general; other shapes---for example one following the cavity of the implicit solvent---are a planned extension of the :code:`jelliumregion` keyword, though only the slab is available in this version (the legacy :class:`gpaw.solvation.sjm.SJM` calculator does offer a cavity-shaped counter charge).


The solvation: screening
------------------------

By itself, the excess electrons and the jellium counter charge would set up an artificially high potential field in the region of the reaction.
To screen this large field, an implicit solvent is added to the simulation in the region above the explicit solvent, completely surrounding the jellium counter charge.
For this purpose, the solvated jellium method employs the implicit solvation model of Held and Walter [Held2014]_, which changes the dielectric constant of the vacuum region.
(You can learn more about the solvation method in the :ref:`continuum_solvent_model` tutorial.)

Here, the primary purpose of the implicit solvent is *not* to solvate the species reacting at the surface; explicit solvent (shown by the water molecules above) is typically employed in SJ simulations for this purpose.
The implicit solvent is located above the explicit solvent (and therefore may provide some solvent stabilization to the explicit solvent molecules).
This can be seen in the figure above, where the implicit solvent is shown as the blue shaded region.
In this figure, the small amount of solvent that is apparent at a `z` coordinate corresponding to the water layer is just the result of the implicit solvent penetrating slightly into the cavity at the center of a hexagonal ice-like water structure.
It is important that the implicit solvent not be present in the region of the reaction, as this would be "double"-solvating those parts.
If this occurs, "ghost" atoms can be added to exclude the solvent from specific regions.

Generalized Poisson equation
----------------------------

In net, the SJ method is manifested as two changes to the generalized Poisson equation,

.. math:: \vec{\nabla} \cdot \Big(\epsilon(\br) \vec{\nabla} \phi(\br)\Big) = -4\pi \Big[ \rho_\mathrm{explicit}(\br) + \rho_\mathrm{jellium} (\br) \Big],

where `\epsilon(\br)` accounts for the solvation; that is, the dielectric constant is spatially variant, and the spatially-resolved charge density is modified by the presence of the `\rho_\mathrm{jellium}(\br)` term, which contains the smeared-out counter charge in a region away from all of the atoms (and electronic density) of the system.
`\rho_\mathrm{explicit} (\br)` contains the standard charge density of the system; that is, due to the electrons and nuclei.
Since the changes to the Poisson equation are relatively simple, it can be solved without relying on linearization.

When the electrode potential is defined as the work function, the generalized Poisson equation is solved with Neumann boundary conditions. When the inner potential is used to define the electrode potential, the generalized Poisson solver is solved with Dirichlet+Neumann boundary conditions as discussed in [Melander2024]_. The Dirichlet boundary condition sets the electrostatic potential to zero at the top of the cell while Neumann boundary conditions set the electrostatic potential gradient to zero.


.. _the_electrode_potential:

The electrode potential
-----------------------

The electrode potential can be defined in two ways: as the work function (wf) or as the electrode inner potential (CIP).

**Work function:**
This is SJM's way of computing the electrode potential. It is computed from the (\phi_\mathrm{e}) defined as the Fermi-level energy (\mu) referenced to a point deep in the solvent (\Phi_\mathrm{w}), where the whole charge on the electrode has been screened and no electric field is present.
(This is equivalently the topside work function of the slab.)
This is divided by the unit electronic charge `e` to convert from energy (typically in eV) to potential (typically in V) dimensions.

.. math:: \phi_\mathrm{e} = \frac{\Phi_\mathrm{w} - \mu}{e} .

Note that this gives the potential with respect to vacuum; if you would like your potential on a reference electrode scale, such as SHE, please see the :ref:`solvated_jellium_method` tutorial.

**Inner potential:**
This electrode potential scale uses the constant inner potential DFT (CIP-DFT) developed in [Melander2024]_. CIP-DFT uses the electrode inner potential (`\phi`) as the electrode potential. This method is always  used with the Dirichlet boundary condition. To use this method one needs to provide the Fermi-level (`\mu^{PZC}_e`) and the inner potential (`\phi^{PZC}`) of the uncharged simulation cell corresponding a potential of zero charge (PZC) at surface charge (`\sigma=0`). The applied electrode potential with respect to PZC at a given surface charge is then given as


.. math:: \phi_\mathrm{e} = \phi[\sigma] - \phi[\sigma=0]

.. _grand-potential-energy:

Legendre-transformed energy
---------------------------

The energy used in the analysis of electrode reactions is the grand-potential
energy

.. math:: \Omega \equiv E_\mathrm{tot} +  N_\mathrm{e}e \phi_\mathrm{e} .

Whereas :math:`E_\mathrm{tot}` is consistent with the forces in traditional
electronic structure calculations, the grand-potential energy :math:`\Omega`
is consistent with the forces in electronically grand-canonical (that is,
constant-potential) simulations. This means that relaxations that follow forces
will find local minima in :math:`\Omega`, and generally methods that rely
on consistent force and energy information (such as BFGSLineSearch or NEB)
will work fine as long as :math:`\Omega` is employed. Thus, this calculator
returns :math:`\Omega` by default, rather than :math:`E_\mathrm{tot}`.

Potential control
=================

The below figure shows both the localization of excess electrons and the local change in potential, when the total number of electrons in an example simulation are changed.

.. image:: delta-ne-phi.png
           :width: 600 px
           :align: center

As mentioned above, the excess electrons localize only on the top side of the slab, which is meant to represent the electrode surface, and not on the bottom side, which is mean to represent the bulk.
The potential drop is seen to localize in the Stern layer where the reaction takes place.
Over reasonable deviations, the relationship between the number of excess electrons and the potential :math:`\phi` is approximately linear:

.. image:: charge-potential.png
           :width: 600 px
           :align: center

Due to the simple relationship between the excess electrons and the potential, reaching a desired potential is typically a fast process.
If you are running a trajectory---for example, a relaxation, a molecular dynamics simulation, or a saddle-point search---the first image will often take a few repetitions (that is, sequential constant-electron calculations) until the desired potential is reached.
Atoms typically move relatively little from image-to-image in a trajectory; therefore, subsequent images are often already at the target potential and no equilibration steps are necessary; when equilibration steps are required, the slope (of potential vs. number of electrons) is recalled from the last adjustment, and it often only takes a single equilibration step.
Typically, over the course of a full trajectory, the added computational cost of working in the constant-potential ensemble is minimal, generally <50% greater computational time compared to a constant-charge calculation.
As described in the  :ref:`solvated_jellium_method` tutorial, this can sometimes be further improved by simultaneously optimizing the potential with the atomic positions; set a loose :literal:`tol` together with :literal:`always_adjust=True`, which keeps stepping the number of excess electrons---on to the next image, without re-converging the current one---even once the potential is within tolerance.
The slope itself is regressed over the last :literal:`slope_regression_depth` attempts and damped with the previous estimate by :literal:`mixer`; :literal:`slope` sets the initial guess, which is otherwise taken from an apparent capacitance of a few muF/cm^2.

**Constant inner potential (CIP) DFT:**

.. Note:: CIP-DFT is implemented only in the legacy
          :class:`gpaw.solvation.sjm.SJM` calculator, so this section uses that
          calculator and its :literal:`sj` dictionary throughout.

The SJM can also be ran in the constant inner potential mode where the
average electrostatic potential within the electrode, i.e.  the inner
potential, is controlled (see :ref:`the_electrode_potential` section). The
usage of CIP-DFT is similar to the standard work function-based SJM with a
few key changes.

To use CIP-DFT mode, one needs to specify the potential reference scale and :literal:`cip` dict in the :literal:`sj` dict::

    sj_cip = {'target_potential': pot, # potential on the absolute potential scale
              'pot_ref': 'CIP',
              'cip': {'autoinner': {'nlayers': 4},
                      'mu_pzc': mu_pzc, # Fermi level at zero charge
                      'phi_pzc': phi_pzc # inner potential at zero charge,
                      }
              }

The :literal:`autoinner` part automatically finds the inner region, in this case a 4-layer metal slab. The target electrode potential is given  with respect to the inner potential (:literal:`phi_pzc`) and Fermi level (:literal:`mu_pzc`) at the potential zero charge (PZC). For instance, if the PZC is 4.44 vs SHE, then :literal:`mu_pzc=-4.44`, and the targeted potential is 0.6 vs SHE, then :literal:`target_potential=3.84`. The needed :literal:`mu_pzc` and :literal:`phi_pzc` parameters are most easily obtained as follows::

        sj_calib = {'excess_electrons':0,
            'pot_ref': 'CIP',
            'cip': {'autoinner': {'nlayers': 4,
                    'threshold': 0.01}
                    }}
    calc = SJM(sj=sj_calib...)
    atoms.calc = calc
    atoms.get_potential_energy()
    phi_pzc = calc.get_inner_potential(atoms)
    mu_pzc = calc.get_fermi_level()

Usage
=====

The SJM parameters are keyword arguments of the :class:`~gpaw.extensions.electrocatalysis.sjm.SJM`
extension, given together with the implicit-solvent parameters
(:literal:`cavity`, :literal:`dielectric` and :literal:`interactions`)::

    from gpaw import GPAW
    from gpaw.extensions.electrocatalysis.sjm import SJM

    calc = GPAW(mode='fd', ...,
                extensions=[SJM(target_potential=4.4,  # absolute scale, V
                                excess_electrons=0.0,  # initial guess
                                tol=0.001,  # V
                                max_iters=10,
                                jelliumregion={'top': -1., 'bottom': -3.},
                                cavity=cavity,
                                dielectric=dielectric,
                                interactions=interactions)])

Leaving out :literal:`target_potential` gives a constant-charge calculation at
the given number of :literal:`excess_electrons`.  The
:literal:`jelliumregion` keyword takes :literal:`'top'`, :literal:`'bottom'`,
:literal:`'thickness'` and :literal:`'fix_bottom'`; a limit given as a negative
number is a distance above the highest atom, and the region then follows the
atoms during a relaxation unless :literal:`fix_bottom=True`.  This is the
default slab-shaped region; a :literal:`'shape'` key selects a different kind
(currently only :literal:`'cavity'`, a counter charge following the solvent
cavity, which is not yet implemented in this version --- the legacy
:literal:`bottom='cavity_like'` spelling is also accepted for it).  With
:literal:`grand_output=False`, the canonical rather than the grand-potential
energy is handed to ASE.

The SJM object is reached as :literal:`calc.sjm`, before as well as after a
calculation --- once one has been run, it is the same object as
:literal:`calc.dft.sjm`.  It offers

* :literal:`set(...)`, which changes :literal:`target_potential`,
  :literal:`excess_electrons`, :literal:`tol`, :literal:`max_iters`,
  :literal:`grand_output`, :literal:`jelliumregion`,
  :literal:`always_adjust`, :literal:`slope`, :literal:`mixer` or
  :literal:`slope_regression_depth` without throwing away the
  wave functions (this replaces the legacy :literal:`calc.set(sj={...})`).
  It can also be called before the first calculation, in which case it simply
  changes the parameters the calculation will be started with.  Note that
  assigning to the attributes instead---:literal:`calc.sjm.charge = ...`---does
  *not* work: only :literal:`set()` gets the charge to the jellium as well and
  empties the results, which is what makes the next
  :literal:`get_potential_energy()` run the SCF loop again.
* :literal:`write_sjm_traces(path='sjm_traces', style='z', props=...)`, which
  writes the electrostatic potential (in eV, referenced to the Fermi level),
  the solvent cavity and the jellium counter charge (in e/Bohr^3, integrating
  to minus the excess electrons) into the directory :literal:`path`.
  :literal:`style='z'` --- the default --- writes one :literal:`<prop>.txt`
  per quantity, averaged over `x` and `y` as a function of `z`, while
  :literal:`style='cube'` writes :literal:`<prop>.cube` on the full 3-D grid.
  :literal:`props` picks which of the three quantities to write.  This one
  needs a calculation to have been run; calling it earlier raises
  :literal:`RuntimeError`.

  Note that a :literal:`'z'` trace from the legacy
  :literal:`gpaw.solvation.sjm.SJM` calculator is shifted by one row relative
  to these: it writes no point at `z` = 0, so its first row is the point at
  one grid spacing and its row `i` is row `i` + 1 here.  Line the two up on
  the `z` column rather than on the row index.

  Note also that :literal:`background_charge` is a charge *density* in e/Bohr^3
  integrating to minus the excess electrons --- not the 0/1 jellium mask that
  the legacy :literal:`gpaw.solvation.sjm.SJM` calculator writes under the same
  name.  Scripts carried over from the legacy calculator have to be adjusted.

So a run that writes both kinds of trace looks like::

    atoms.get_potential_energy()  # traces need a converged calculation

    sjm = atoms.calc.sjm
    sjm.write_sjm_traces()  # sjm_traces/{potential,cavity,background_charge}.txt
    sjm.write_sjm_traces(path='traces-3d', style='cube')  # traces-3d/*.cube

Both calls are collective: run them on all ranks, as you would any other
calculator method.

References
==========

.. [Kastlunger2018] G. Kastlunger, P. Lindgren, A. A. Peterson,
                    :doi:`Controlled-Potential Simulation of Elementary Electrochemical Reactions: Proton Discharge on Metal Surfaces <10.1021/acs.jpcc.8b02465>`,
                    *J. Phys. Chem. C* **122** (24), 12771 (2018)
.. [Held2014] A. Held and M. Walter,
           :doi:`Simplified continuum solvent model with a smooth cavity based on volumetric data <10.1063/1.4900838>`,
           *J. Chem. Phys.* **141**, 174108 (2014).

.. [Melander2024] Marko M. Melander, Tongwei Wu, Timo Weckman, Karoliina Honkala,
           :doi:`Constant inner potential DFT for modelling electrochemical systems under constant potential and bias <10.1038/s41524-023-01184-4>`,
           *npj Computational Materials* **10** Article number: 5 (2024)

Class documentation
===================

.. autoclass:: gpaw.extensions.electrocatalysis.sjm.SJM

.. autoclass:: gpaw.extensions.electrocatalysis.sjm.SJMExtension
   :members: set, write_sjm_traces, get_electrode_potential

.. autoclass:: gpaw.solvation.sjm.SJMPower12Potential

The legacy calculator---:literal:`SJM(sj={...})` from
:literal:`gpaw.solvation.sjm`---is still available, and is what the CIP-DFT
section above uses:

.. autoclass:: gpaw.solvation.sjm.SJM
