.. _machinelearning_finetune:


====================
Fine-tuning the MLIP
====================

Fine-tuning the MLIP is the process of improving the MLIP by training it on configurations specific for your problem.

To fine tune you need a training set.  We will use a minimalistic training set containing 10 to 20 structures calculated by DFT.  A good starting point is the structures you calculated when :ref:`verifying the MLIP <machinelearning_verify>`.  If there are not enough configuration, go back and run a slightly different trajectory (change the projectile starting position and/or the starting momentum slightly).

If your training set is split over multiple ``.traj`` files, they should be combined into a single one.  This can be done with the ``ase`` command line tool::

    ase convert file1.traj file2.traj output.traj

Downloading the full MLIP data
==============================

To fine tune a model, you need extra data about the model.  That can all be downloaded from https://nequip.net, just select the model we are using (NequIP-OAM-S) and then click on the download link in the box on the right.

To download onto the DTU computer instead of you own laptop, left-click the link and copy it, then paste it into the ``curl`` command instead of URL::

    curl -o NequIP-OAM-S-0.1.nequip.zip URL



Writing a fine-tuning configuration file
========================================


Calculating the reference energies
==================================


Running the fine-tuning on GPUs
===============================
