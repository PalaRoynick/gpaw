.. _machinelearning:

=======================================
Machine Learning interatomic potentials
=======================================

This exercise illustrates the use of Machine Learning Interatomic Potentials (MLIPs).  In particular, we explore foundational models, pretrained models that can be used out-of-the-box for moderate accuracy or fine-tuned for a specific application for greater accuracy.

We have chosen to base the simulation on the NequIP_ package and their small pretrained model NequIP-OAM-S.  This is their smallest model, chosen for performance over accuracy.  The medium sized model is more than twenty times slower and gives very similar results!
*For scientific purposes you probably want the medium or even the large model for anything beyond initial testing.*

Setting up your account
=======================

This project requires the Torch and NequIP packages, adding more than seven gigabytes to your installation. For that reason the packages are not installed for all students, but should only be installed if you plan to follow this project.

To install the machine learning tools, run this command in a terminal::

    setup ~jasc/setup2026ml

Now download the NequIP-OAM-S foundational model, and prepare to run it on the DTU computer system, using CPUs.  While GPUs give significant acceleration of the larger models, we gain little with the small model, and we only have access to very limited GPU power, so we will only use that for training.

::

    cd CAMD2026

    nequip-compile nequip.net:mir-group/NequIP-OAM-S:0.1 NequIP-OAM-S_cpu.nequip.pt2 --mode aotinductor --device cpu --target ase

Note that there are only two lines, the second is long and may wrap in your browser!


Theory
======

An MLIP is a mathematical function, taking an atomic structure as input and returning an approximation of the potential energy of the structure.  In the following we look at neural-network based MLIPs, although some of the theory is more general.

The potential energy of an atomic system has a number of symmetries that the MLIP must respect.  The most important are

* Invariance under *permutation of identical atoms*.
* Invariance under *translations*, *rotations* and *mirroring*, i.e. invariance under the elements of the 3-dimensional Euclidian group E(3).

Invariance under permutation is obtained by defining a per-atom energy: each atom has an energy that only depends on the relative positions of the surrounding atoms.  The total energy is then the sum of the atomic energies.  This is in contrast to Density Functional Theory, where atomic energies are not defined, and indeed cannot be defined, only the total energy exists in DFT.

As only relative positions enter the expression for the atomic energy, tranlsational invariance is trivially fulfilled.  Invariance under rotations and mirroring/inversion can be obtained in two ways.

* In an *invariant model,* a (large) number of invariant descriptors are calculated from the atomic positions, and are fed as input to the neural network (or another machine learning method).
* In an *equivariant model*, the interatomic distance vectors are fed into an *equivariant* neural network.  All elements of the network then operate on quantities that transform correctly under rotations and inversions (think of them as scalars, vectors and tensors, although in reality they are irreducible representations of the O(3) group, which transform like the spherical harmonics :math:`Y_{lm}` that you know from quantum mechanics).

Although the final potential energy is a scalar, and thus *invariant* under rotation, the equivariant models have a significant advantage, often requiring orders of magnitude less training data.  This appears to be due to a loss of information when invariant descriptors are constructed from the interatomic vectors, by keeping the vectors the network itself is able to construct more appropriate descriptors.

**Question:** The potential energy is a scalar and thus *invariant* under rotation.  But we also need the forces, they are vectors and are not invariant.  Discuss with your peers why an invariant model is nevertheless able to give correct forces.

You can read more about Equivariant Neural Networks in 

  Thomas *el al.*, Tensor field networks: Rotation- and translation-equivariant neural networks for 3D point clouds, `arXiv: 1802.08219 <https://arxiv.org/abs/1802.08219>`_.

The network described in the paper is implemented as the Python package ``e3nn``.  Several specific frameworks for equivariant MLIPs (as opposed to other applications of equivariant networks) have been build on top of ``e3nn``.  The one we use here is NequIP_, described in these publications

  S. Batzner *et al.*, E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials. *Nature Communications,* **13,** 2453 (2022). DOI: `10.1038/s41467-022-29939-5 <https://doi.org/10.1038/s41467-022-29939-5>`_.

  C. W. Tan *et al.*, High-performance training and inference for deep equivariant interatomic potentials.  *Digital Discovery* **5**, 1558–1567 (2026). DOI: `10.1039/d5dd00423c <https://doi.org/10.1039/d5dd00423c>`_

.. _NequIP: https://nequip.readthedocs.io/en/latest/



Exercises
=========

.. toctree::

   machinelearning_intro
