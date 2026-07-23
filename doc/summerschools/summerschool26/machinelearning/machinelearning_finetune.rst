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



Calculating the reference energies
==================================

The original model was trained on VASP calculations, and the VASP reference energies differ from GPAW's.  For this reason, you need to calculate the energy of a single atom of each element that appears in the simulation.

You should do this with the same plane wave cutoff, same number of k-points and if possible the same unit cell size as for the actual DFT calculations you want to train on.

You can write a script that reads a configuration, extracts a list of elements present, and for each element calculates the energy of a single using the same unit cell as in the input file.  Or you can do it more manually.  You will need the energies in the next step.


Writing a fine-tuning configuration file
========================================

NequIP needs a configuration file in the YAML (Yet Another Markup Language) file format.  The following file is a suitable starting point.  You need to edit it, replacing:

* ``PATH_TO_ZIP`` with the actual path to the .zip file containing the foundational model.
* ``PATH_TO_TRAINING_TRAJFILE`` with the path to the training set.
* The atomic energies in the ``shifts`` section near the end with the energies calculated above for the elements that you are actually simulating.

Be very careful when editing the ``.yaml`` file that your editor does not replace spaces with TAB characters, that will break the parsing of the file.

:download:`finetuning.yaml <finetuning.yaml>`:

.. literalinclude:: finetuning.yaml


Running the fine-tuning on GPUs
===============================

The fine tuning script is a shell-script, not a Python script, so it cannot be submitted to the HPC cluster through MyQueue - instead to have to use the underlying LSF queue system.  You need a script that calls the Nequip training program with the YAML file as its argument.  The file needs to contain the correct instructions to the LSF queuing system as well.  Such a file can be found here:

:download:`train.bsub <train.bsub>`:

.. literalinclude:: train.bsub
    
