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


Make an output folder on a fast disk
====================================

The fine-tuning script will overload the normal disk server.  Please make an output folder on the fast parallel disk system.  In a terminal window, go to the folder where you run the script and make a soft-link to a folder on the fast disk::

    cd folder_where_you_run_your_scripts
    mkdir $BLACKHOLE/outputs
    ln -s $BLACKHOLE/outputs

(Do not be scared by the "black hole" name, your files will stay there during the summer school)


Running the fine-tuning on GPUs
===============================

The fine tuning script is a shell-script, not a Python script, so it cannot be submitted to the HPC cluster through MyQueue - instead to have to use the underlying LSF queue system.  You need a script that calls the Nequip training program with the YAML file as its argument.  The file needs to contain the correct instructions to the LSF queuing system as well.

Use this command line to submit the script::

    bsub < train.bsub

*Do not forget the less-than character, or you will submit a weird job that fails.*

A sample script file can be found here:

:download:`train.bsub <train.bsub>`:

.. literalinclude:: train.bsub

After the job has completed, you need to "package" and "compile" the trained model before it can be used.  Start by checking that the training job produce the expected output - the folder path in the ``ls`` command should match the output path appearing a few times in the ``finetuning.yaml`` file::

    ls -l outputs/finetuned/OAM_S

The output will look something like this::

    total 19392
    drwxr-xr-x 3 jasc fys      31 Jul 24 12:31 OAM-S
    -rw-r--r-- 1 jasc fys 9927963 Jul 24 12:35 best.ckpt      <--- Look for this file!
    -rw-r--r-- 1 jasc fys 9927963 Jul 24 12:35 last.ckpt

Then you can package and compile the new potential::

    nequip-package build outputs/finetuned/OAM_S/best.ckpt outputs/finetuned/OAM_S_finetuned.nequip.zip
    nequip-compile outputs/finetuned/OAM_S_finetuned.nequip.zip NequIP-OAM-S_finetuned_cpu.nequip.pt2 --mode aotinductor --device cpu --target ase

This should produce a new file ``NequIP-OAM-S_finetuned_cpu.nequip.pt2`` that can be used in your scripts.

Testing if fine-tuning helped
=============================

To test you need to produce a *test set* of configurations that have not been used during training.  For example, run the same script you produced to generate the training data, but with a slightly different position and/or energy of the projectile atom.  Select a number of frames, calculate their energies with DFT.  Then load those frames one at a time and recalculate the energies with both the old and the new MLIP, and compare.

Experimenting with fine-tuning
==============================

*If you feel inclined, you can experiment with the fine tuning hyperparameters*

The YAML file above provides a good starting
point for fine-tuning. Using these settings is
likely to significantly improve the accuracy of the
foundational model. The size of the training set
is still the main limiting factor, but there are a
few parameters that can be adjusted to squeeze
a little more performance out of the available data.

**Data set size:**  You can run one or two MD trajectories with the fine-tuned model, calculated DFT energies for selected frames, and add to the training set.

**Train/Validation/Test split:**
The dataset is divided into three parts. The training
set is used to update the model parameters, the
validation set is used to monitor performance
during training, and the test set is used for
final evaluation. Since we are testing the model
ourselves, we have per default removed the test set entirely.
Especially for small datasets, playing with the split between training and validation set might be beneficial.

**Batch size:**
The batch size used during training is specified
under train dataloader. This determines how many
structures are evaluated before the model updates
its weights. Larger batch sizes make more efficient
use of the GPU, but they also result in fewer
parameter updates per epoch. Very large batch
sizes can therefore reduce the effectiveness of
training, particularly for small datasets. The batch
size should always be smaller than the size of the
training set, and if you encounter GPU memory
limitations, reducing the batch size is usually the
first parameter to adjust.

**EMA decay:**
The ”ema decay” parameter controls the
exponential moving average (EMA) of the
model parameters during training. The EMA
keeps a separate, smoothed version of the model
weights that changes more slowly than the regular
model weights. These slowly changing weights are
used for validation, testing, and saving the final
model. A value chosen close to 1, means that the
EMA weights are updated slowly and average over
many previous training steps. Lower values make
the EMA follow the training weights more closely.

**Learning rate:**
The learning rate determines how large each
optimization step is, i.e. by how much the model
can update the weights. Since we are fine-tuning
a foundational model, this becomes a somewhat
tricky parameter. The benefit of fine-tuning a
foundational model is that it has been trained
on a large dataset, from which it has learned
transferable information. However, we also want
to adapt it to our specific material and use case.
If the learning rate is too high, you risk the
model ”forgetting” what it has already learned,
but setting the learning rate too low means it will
learn very little from the additional frames.

**Loss function:**
Lastly, the loss function can also be changed.
The loss function determines what the model
tries to optimize during training. The validation
metrics are used to assess performance during
validation. In the provided YAML file, the loss
function combines the RMSE of the forces and
the per-atom RMSE of the energies, with weights
4 and 1 respectively. NequIP provides a lot of
options for different metrics as can be seen here:
https://nequip.readthedocs.io/en/latest/api/metrics.html
It is even possible to use different metrics
for training and validation, although this is
generally not recommended.

Real scientific work requires larger data sets
==============================================

Obviously, you have been using a small foundational model and a ridiculously small training set for the fine tuning, so do not expect miracles.  Real scientific work requires larger models and larger data sets.

**This concludes the Machine Learning exercise.**  Note that it might be possible to try out some of the other exercises with a MLIP instead of DFT...

