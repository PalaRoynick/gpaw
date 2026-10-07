.. module:: gpaw.extensions.electrochemistry
.. _sjm:

======================================================
Solvated Jellium (constant-potential electrochemistry)
======================================================

Overview
========

The Solvated Jellium method (:class:`~gpaw.extensions.electrochemistry.SJM`) is a simple method for the simulation of electrochemical interfaces in DFT.
A full description of the approach can be found in [Kastlunger2018]_.
The method allows you to control the simulated electrode potential (manifested as the topside work function) by varying the number of electrons in the simulation; calculations can be run in either constant-charge or constant-potential mode.
SJM is attached to a regular GPAW calculator as an extension---``GPAW(..., extensions=[SJM(...)])``---and the calculator is then used just like any other; it returns energy and forces, but can do so at a fixed potential.
The number of excess electrons and the electrode potential are returned alongside them, as the ``excess_electrons`` and ``electrode_potential`` properties.
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
By default that region is a slab perpendicular to `z`, controlled by the ``jelliumregion`` key described under `Usage`_ below.
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
Additionally, note that the jellium is enclosed in a regular slab geometry in the figure above, but this need not be the case in general; other shapes---for example one following the cavity of the implicit solvent---are a planned extension of the :code:`jelliumregion` key, though only the slab is available in this version (the old :class:`gpaw.solvation.sjm.SJM` calculator (only working for `fd`-mode) does offer a cavity-shaped counter charge).


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
This is SJM's way of computing the electrode potential. It is the Fermi-level energy `\mu` referenced to a point deep in the solvent, `\Phi_\mathrm{w}`, where the whole charge on the electrode has been screened and no electric field is present.
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

The electrostatic potential is only defined up to a constant, so the traces in the upper panel have to be aligned somewhere before they can be compared, and where they are aligned decides where the fanning appears.
SJM itself references the potential deep in the solvent, at the top of the cell, which is where it reads off the electrode potential; the traces here are instead aligned at the vacuum *behind* the slab, on the bulk side, so that the difference between two charge states accumulates across the slab and the Stern layer.
The fanning on the solvent side is therefore a consequence of that alignment, not a separate physical effect: it is the same information as a rigid shift of each curve, displayed so that the region where the potential drop develops is the region where the curves separate.
Note that the backside vacuum is not the point `z = 0` in plane-wave mode, where the cell is periodic and `z = 0` is the same point as the top of the cell; the alignment is taken midway between the lower cell boundary and the lowest atom, and the sharp feature below `z \approx 2\,\mathrm{\AA}` is where each trace returns to the value SJM pins in the solvent.

Over reasonable deviations, the relationship between the number of excess electrons and the potential :math:`\phi` is approximately linear:

.. image:: charge-potential.png
           :width: 600 px
           :align: center

Due to the simple relationship between the excess electrons and the potential, reaching a desired potential is typically a fast process.
If you are running a trajectory---for example, a relaxation, a molecular dynamics simulation, or a saddle-point search---the first image will often take a few repetitions (that is, sequential constant-electron calculations) until the desired potential is reached.
Atoms typically move relatively little from image-to-image in a trajectory; therefore, subsequent images are often already at the target potential and no equilibration steps are necessary; when equilibration steps are required, the slope (of potential vs. number of electrons) is recalled from the last adjustment, and it often only takes a single equilibration step.
Typically, over the course of a full trajectory, the added computational cost of working in the constant-potential ensemble is minimal, generally <50% greater computational time compared to a constant-charge calculation.
As described in the  :ref:`solvated_jellium_method` tutorial, this can sometimes be further improved by simultaneously optimizing the potential with the atomic positions; set a loose ``tol`` together with ``always_adjust=True``, which keeps stepping the number of excess electrons---on to the next image, without re-converging the current one---even once the potential is within tolerance.
The slope itself is regressed over the last ``slope_regression_depth`` attempts and damped with the previous estimate by ``mixer``; ``slope`` sets the initial guess, which is otherwise taken from an apparent capacitance of 10 muF/cm^2.

All of this is done by a :class:`~gpaw.extensions.electrochemistry.Potentiostat`, which SJM builds from its ``potentiostat`` dict and which then owns the settings---``sjm.potentiostat.always_adjust``, ``sjm.potentiostat.optimizer.target_potential`` and the rest.
The potentiostat is a general constant-potential engine rather than an SJM-specific one: how the counter charge enters the cell (the *charging scheme*, a jellium slab here), how the potential is read off the converged state (the *potential probe*, the topside work function here) and how the charge is stepped toward the target (the *optimizer*, the slope regression described above) are separate, interchangeable pieces.
It also finishes a converged calculation: it reports the equilibration, Legendre-transforms the energies to the grand potential and writes the results, the grand-potential or the canonical energy as ``grand_output`` says.
SJM is the front end for the jellium-plus-implicit-solvent combination; a different constant-potential method can reuse the same potentiostat with another charging scheme, optimizer or probe, and SJM itself takes parts of your own (see `Usage`_).

**Constant inner potential (CIP) DFT:**

.. Note:: CIP-DFT is implemented only in the legacy
          :class:`gpaw.solvation.sjm.SJM` calculator, so this section uses that
          calculator and its ``sj`` dictionary throughout.

The SJM can also be ran in the constant inner potential mode where the
average electrostatic potential within the electrode, i.e.  the inner
potential, is controlled (see :ref:`the_electrode_potential` section). The
usage of CIP-DFT is similar to the standard work function-based SJM with a
few key changes.

To use CIP-DFT mode, one needs to specify the potential reference scale and ``cip`` dict in the ``sj`` dict::

    sj_cip = {'target_potential': pot, # potential on the absolute potential scale
              'pot_ref': 'CIP',
              'cip': {'autoinner': {'nlayers': 4},
                      'mu_pzc': mu_pzc, # Fermi level at zero charge
                      'phi_pzc': phi_pzc # inner potential at zero charge,
                      }
              }

The ``autoinner`` part automatically finds the inner region, in this case a 4-layer metal slab. The target electrode potential is given  with respect to the inner potential (``phi_pzc``) and Fermi level (``mu_pzc``) at the potential zero charge (PZC). For instance, if the PZC is 4.44 vs SHE, then ``mu_pzc=-4.44``, and the targeted potential is 0.6 vs SHE, then ``target_potential=3.84``. The needed ``mu_pzc`` and ``phi_pzc`` parameters are most easily obtained as follows::

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

The :class:`~gpaw.extensions.electrochemistry.SJM` extension takes two
keywords: ``solvation``, the implicit-solvent model (``cavity``,
``dielectric`` and ``interactions``), and ``potentiostat``, the SJM
settings::

    from gpaw import GPAW, PW
    from gpaw.extensions.electrochemistry import SJM

    calc = GPAW(mode=PW(400.), ...,
                extensions=[SJM(
                    solvation=dict(cavity=cavity,
                                   dielectric=dielectric,
                                   interactions=interactions),
                    potentiostat=dict(
                        target_potential=4.4,  # absolute scale, V
                        excess_electrons=0.0,  # initial guess
                        tol=0.005,  # V
                        max_iters=10,
                        jelliumregion={'top': -1., 'bottom': -3.}))])

Each of the two can also be given as an object instead of a dict: a
:class:`gpaw.solvation.Solvation` and a
:class:`~gpaw.extensions.electrochemistry.Potentiostat`, or subclasses of
them.  A solvation model without a dipole layer of its own gets
``GaussianDerivativeDipole(zero_vacuum=True)``, which references the
electrode potential to the vacuum at the top of the cell.

Leaving out ``target_potential`` gives a constant-charge calculation at
the given number of ``excess_electrons``.  The
``jelliumregion`` key takes ``'top'``, ``'bottom'``,
``'thickness'`` and ``'fix_bottom'``.  A negative ``'top'``
is a distance below the upper cell boundary, whereas a negative
``'bottom'`` is a distance above the highest atom, in which case the
region follows the atoms during a relaxation unless
``fix_bottom=True``.  This is the default slab-shaped region; a
``'shape'`` key selects a different kind
(currently only ``'cavity'``, a counter charge following the solvent
cavity, which is not yet implemented in this version --- the legacy
``bottom='cavity_like'`` spelling is also accepted for it).  With
``grand_output=False``, the canonical rather than the grand-potential
energy is handed to ASE.

**Parts of your own:** the potentiostat is built from three parts, each
with a default: the charging scheme
(:class:`~gpaw.extensions.electrochemistry.JelliumCharging`), the optimizer
(:class:`~gpaw.extensions.electrochemistry.DirectOptimizer`) and the
potential probe
(:class:`~gpaw.extensions.electrochemistry.WorkFunctionProbe`).  The keys
``excess_electrons`` and ``jelliumregion`` configure the default charging
scheme, and ``target_potential``, ``tol``, ``slope``, ``mixer``,
``slope_regression_depth``, ``max_potential_stepsize``, ``max_iters`` and
``max_reruns`` the default optimizer, which holds the target, decides when
the potential is converged (``PotentialOptimizer.converged()``) and sets
the limits of the equilibration.  To use a part of your own, subclass
:class:`~gpaw.extensions.electrochemistry.ChargingScheme`,
:class:`~gpaw.extensions.electrochemistry.PotentialOptimizer` or
:class:`~gpaw.extensions.electrochemistry.PotentialProbe` and give an
instance under the ``charging``, ``optimizer`` or ``potentialprobe`` key::

    sjm = SJM(solvation=...,
              potentiostat={'optimizer': MyOptimizer(target_potential=4.4)})

Its settings then go to the part itself: the keys of the default part it
replaces raise a ``ValueError``.  A ``.gpw`` file records a part of your
own by its class name only, so reading the file back raises
``ValueError: Unknown SJM part``.

To change the settings, keep the ``SJM(...)`` object in a variable
and call its ``change(potentiostat={...})``, which takes any key of the
``potentiostat`` dict but the parts::

    sjm = SJM(solvation=..., potentiostat={'target_potential': 4.4})
    atoms.calc = GPAW(..., extensions=[sjm])
    atoms.get_potential_energy()
    sjm.change(potentiostat={'target_potential': 4.3})
    atoms.get_potential_energy()

Before the first calculation this simply changes the settings the
calculation will be started with; afterwards it changes them without
throwing away the wave functions.  This replaces the legacy
``calc.set(sj={...})``.

The same object also offers

* ``sjm.potentiostat``, the potentiostat that runs in the calculation
  (``calc.dft.potentiostat`` is the same object), and ``sjm.solvation``.
  The settings are read off the potentiostat and its parts, as live
  values: ``sjm.potentiostat.excess_electrons`` (as the potentiostat
  adjusts it), ``sjm.potentiostat.optimizer.target_potential``,
  ``sjm.potentiostat.optimizer.slope`` (the current estimate),
  ``sjm.potentiostat.optimizer.tol`` and
  ``sjm.potentiostat.charging.region``.  ``todict()``, and so the ``.gpw``
  file, records these live values.  Change them through ``change()``
  rather than by assignment: only ``change()`` clears the results and
  recomputes the occupations, density and potential, which makes the next
  ``get_potential_energy()`` run the SCF loop again.
* ``get_electrode_potential()``, the electrode potential of the last
  converged SCF loop, in V on the vacuum scale, as the potential probe
  reads it---minus the Fermi level for the default ``WorkFunctionProbe``,
  since the Poisson solver zeroes the electrostatic potential in the
  vacuum.  It is the same number as
  ``calc.results['electrode_potential']``, and NaN until an SCF loop has
  converged.
* ``write_sjm_traces(path='sjm_traces', style='z', props=...)``, which
  writes the electrostatic potential (in eV, referenced to the Fermi level),
  the solvent cavity and the jellium counter charge (in e/Bohr^3, integrating
  to minus the excess electrons) into the directory ``path``.
  ``style='z'`` --- the default --- writes one ``<prop>.txt``
  per quantity, averaged over `x` and `y` as a function of `z`, while
  ``style='cube'`` writes ``<prop>.cube`` on the full 3-D grid.
  ``props`` picks which of the three quantities to write.  This one
  needs a calculation to have been run; calling it earlier raises
  ``RuntimeError``.  It is collective: call it on all ranks.

  Note that a ``'z'`` trace from the legacy
  ``gpaw.solvation.sjm.SJM`` calculator is shifted by one row relative
  to these: it writes no point at `z` = 0, so its first row is the point at
  one grid spacing and its row `i` is row `i` + 1 here.  Line the two up on
  the `z` column rather than on the row index.

  Note also that ``background_charge`` is a charge *density* in e/Bohr^3
  integrating to minus the excess electrons --- not the 0/1 jellium mask that
  the legacy ``gpaw.solvation.sjm.SJM`` calculator writes under the same
  name.  Scripts carried over from the legacy calculator have to be adjusted.

So a run that reports the potential and writes both kinds of trace looks
like::

    sjm = SJM(solvation=..., potentiostat={'target_potential': 4.4})
    atoms.calc = GPAW(..., extensions=[sjm])
    atoms.get_potential_energy()  # both need a converged calculation

    print(sjm.get_electrode_potential(), sjm.potentiostat.excess_electrons)
    sjm.write_sjm_traces()  # sjm_traces/{potential,cavity,background_charge}.txt
    sjm.write_sjm_traces(path='traces-3d', style='cube')  # traces-3d/*.cube

The ``write_sjm_traces()`` calls are collective: run them on all ranks, as
you would any other calculator method.

After reading a ``.gpw`` file, the ``SJM`` object is
``calc.params.extensions[0]`` again.  The running parts of the calculation
are ``calc.dft.potentiostat`` and ``calc.dft.solvation``.

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

.. autoclass:: gpaw.extensions.electrochemistry.SJM
   :members: change, write_sjm_traces, get_electrode_potential

.. autoclass:: gpaw.extensions.electrochemistry.Potentiostat
   :members: from_dict, todict, equilibrate, new_geometry,
             post_converged, write_results

.. autoclass:: gpaw.extensions.electrochemistry.ChargingScheme
   :members: initialize, todict, add_charge, add_charge_pw, move

.. autoclass:: gpaw.extensions.electrochemistry.JelliumCharging

.. autoclass:: gpaw.extensions.electrochemistry.PotentialOptimizer
   :members: set_target_potential, converged, next_excess_electrons,
             record_attempt, last_attempt, forget_attempts, reset, todict

.. autoclass:: gpaw.extensions.electrochemistry.DirectOptimizer

.. autoclass:: gpaw.extensions.electrochemistry.PotentialProbe
   :members: potential, todict

.. autoclass:: gpaw.extensions.electrochemistry.WorkFunctionProbe

.. autoclass:: gpaw.solvation.sjm.SJMPower12Potential

The legacy calculator---``SJM(sj={...})`` from
``gpaw.solvation.sjm``---is still available, and is what the CIP-DFT
section above uses:

.. autoclass:: gpaw.solvation.sjm.SJM
