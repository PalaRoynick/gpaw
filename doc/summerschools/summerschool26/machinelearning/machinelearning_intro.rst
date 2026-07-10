
.. _machinelearning_intro:


===================================
Selecting and testing a 2D material
===================================

For this exercise, you should select a 2D material to work with.  The material should not have a huge unit cell, and it should be well described by the foundational model we start to work with.  The latter has to be checked.

Select a 2D material from the `Computational 2D-Materials Database (C2DB) <https://c2db.fysik.dtu.dk>`_.  It contains around 17000 2D materials, many of them hypothetical.  All the well-known ones are there, such as C (graphene), BN (hexagonal boron nitride), MoS<sub>2</sub> (molybdenum disulphide, a transition metal dichalchogenide), Ti<sub>2</sub>C (titanium carbide, an MXene).  There are also more exotic materials, like CrBrS - still experimentally observed but with a more complicated structure.

Search for a composition in C2DB.  In this example, we look for MoS2 you write MoS2 in the search box.  Look for the material with the lowest "Energy above hull", if that is zero the material is thermodynamically stable, if it is low it is possibly metastable.  In this case we find two structures, the stable so-called 2H phase and the metastable 1T phase with twice as many atoms in the unit cell.

.. image:: c2db_MoS2_search.png
   :width: 80%

Once we click on the stable phase, we get an overview:

.. image:: c2db_MoS2_view.png
   :width: 80%

On the overview, please note a few features.  The field **Experimental realizations** shows that this material actually exists.  And most importantly, in the upper right corner you can download the structure in a few file formats.  The XYZ format is particularly simple. Clicking on the box gives you the file ``1MoS2-1.xyz`` which we will need in the following.

**PLEASE CHOOSE YOUR OWN MATERIAL!**  If you work together in a group, you should choose different materials.


