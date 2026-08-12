.. _summerschool26:

=======================
CAMD Summer School 2026
=======================

Announcement:
https://www.fysik.dtu.dk/english/research-sections/camd/events/summer-school-2026

.. highlight:: bash


Summer school exercises
=======================

The Summer School includes a number of :ref:`projects`, where you will
need to run
calculations on the DTU central computing facilities.
This requires some setup, which is described below.


The computer system
===================

The DTU computer system (known as the 'G-databar' for obscure reasons)
consists of a login node (named ``login.gbar.dtu.dk`` or ``gbarlogin``)
and a number of compute nodes.  Some of the compute nodes are reserved for
batch jobs, some allow interactive jobs.  You need to run on the interactive
compute nodes, running on the login node will overload it.



Instructions
============

There are two main ways to log in to the DTU system, *SSH* and *ThinLinc*.

**SSH** is probably the most convenient.  Programs on the remote system will be able to open windows on your computer just like local programs.

**ThinLinc** opens one big window with a Unix desktop running on the DTU computer.  Putting this window in full-screen mode makes your computer act as a graphical terminal.  Programs on the remote computer open their windows within this desktop window.

Which one you choose is a matter of taste.  We find ssh slightly more convenient once it is set up, but ThinLinc may be a bit easier to set up initially.


Setting up ssh for the first time
---------------------------------

The instructions depend on the operating system you use


.. toctree::
   :maxdepth: 1

   setupwin
   setuplinmac


Setting up ThinLinc for the first time
--------------------------------------

IMPORTANT: You should either set up SSH or ThinLinc, **no need to do both!**

.. toctree::
   :maxdepth: 1

   setupthinlinc


Using the DTU computers
-----------------------

.. toctree::
   :maxdepth: 1

   using_gbar
   submitting


iPad users (strongly discouraged)
---------------------------------

Working from an Ipad is *very* inconvenient, but possible.

.. toctree::
   :maxdepth: 1

   ipad


Introductory slides
-------------------

The slides from the introduction to the exercises (Sunday) can be
found here: :download:`Intro_ASE_Databar.pdf`


.. _projects:

Projects
========

Choose a project according to your interests.  The projects contain
brief descriptions of what aspects of GPAW you learn from them.

.. toctree::
   :maxdepth: 3

   intro/intro
   catalysis/catalysis
   magnetism/magnetism
   machinelearning/machinelearning
   excitedstates/excitedstates
   batteries/batteries
   workflows/workflows


Instructions for instructors
============================

Instructions for the people testing and developing the projects can be
found here:

.. toctree::
   :maxdepth: 3

   instructors
