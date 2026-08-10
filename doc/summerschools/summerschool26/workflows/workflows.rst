=======================
Computational workflows
=======================

In this exercise we will write and run computational workflows
with `TaskBlaster <https://taskblaster.readthedocs.io/en/latest/>`_.

The basic unit of computation in TaskBlaster is a *task*.  A task is a Python
function along with a specification of the inputs to that function.
The inputs can be either concrete values like lists, strings, or numbers,
or references to the outputs of other tasks.
Tasks that depend on one another and form a graph.  A
TaskBlaster workflow is a Python class which defines such a graph, along with
metadata about how the tasks should run and be stored.  Workflows can
then be parametrized so they run on a collection of materials,
for example.

When using TaskBlaster, we define workflows and tasks using Python.
However, the tools used to *run* workflows and tasks are command-line
tools.  Therefore, for this exercise we will be using the terminal
rather than the usual notebooks.  Basic knowledge of shell commands
is an advantage.

This exercise consists of multiple parts:

 * Introductory tutorials to TaskBlaster
 * Write a workflow which defines a structure optimization task
   followed by electronic ground-state and band structure computation
   using GPAW
 * Parametrize the workflow to comprise multiple materials
 * Submit multiple workers to execute the computational tasks at scale


Part 0: The TaskBlaster tutorials and documentation
===================================================

TaskBlaster comes with its own documentation and tutorials.
These are not related to physics or materials, but provide a generic
introduction to the framework:

 * To get started, go through the basic
   `TaskBlaster tutorials <https://taskblaster.readthedocs.io/en/latest/tutorial/module.html>`_,
   specifically
   `“Hello world” <https://taskblaster.readthedocs.io/en/latest/tutorial/hello_world/hello_world.html>`_ and
   `“My first workflow” <https://taskblaster.readthedocs.io/en/latest/tutorial/my_first_workflow/my_first_workflow.html>`_.
   Make sure you can run the examples and feel free to explore
   the command-line interface.

 * It may be illustrating to read the explanation page on
   `Basic Concepts <https://taskblaster.readthedocs.io/en/latest/explanation/basic_concepts/basic_concepts.html>`_.

We have now acquired a basic familiarity with TaskBlaster commands,
although we may still not see the full picture of how one runs
scientific high-throughput projects in practice.  We will deal with this
over the course of the next exercises.


Part 1: Simple materials workflow
=================================

In this exercise we will develop a first materials workflow using the
GPAW electronic structure code.
This exercise will be more open-ended.
Be sure to make good use of the different command-line tools'
:option:`--help` pages as well as TaskBlaster's online documentation.

TaskBlaster itself does not know anything about electronic structure,
materials, atoms, or ASE.
However we will want tasks to have ASE objects as input
and output, and that requires being able to save those objects.
TaskBlaster can be extended with the ability to encode and decode arbitrary
objects, and doing so requires a plugin.
Such a plugin is provided by asr-lib, the
`Atomic Simulation Recipes <https://gitlab.com/asr-dev/asr-lib>`_ library.
We will create a project using asr-lib as a plugin and so facilitate
our work with ASE objects.

Go to a clean directory and create repository using the ``asrlib`` plugin::

  $ tb init asrlib

You can use the ``tb info`` command to see global information about
the repository and verify that it uses ``asrlib``.

Set up structure optimization
-----------------------------

Write a workflow class called ``MaterialsWorkflow`` which:

 * takes ``atoms`` as an input variable
 * takes ``calculator`` as an input variable, which is a dictionary
   of keyword arguments that will be passed to GPAW
 * defines a ``relax`` task which performs a structure optimization
   that includes optimizing the unit cell.

The wise scientist first writes a relax task which only prints the
atoms.  That is enough to verify that the workflow works, and that the
atoms are passed and encoded correctly.  After that, unrun, edit,
rerun, and fix it until it works. When setting up a workflow and 
having to unrun tasks it can be useful to use the ``--rerun`` flag to 
immediatly rerun a failed task. The workflow should function
correctly when called on a bulk silicon system like this:

.. literalinclude:: workflow.py
   :pyobject: workflow

Here we have chosen some generic GPAW parameters that will be fine
for testing, but not for production.

The relaxation task can be implemented like this:

.. literalinclude:: tasks.py
   :pyobject: optimize_cell

Users unfamiliar with ASE may want to take a while to look up
ASE concepts like atoms and calculators.  What the function does is:

 * Attach a GPAW calculator to the atoms
 * Create a
   `Frechet cell filter
   <https://docs.ase-lib.org/ase/filters.html#the-frechetcellfilter-class>`_
   which exposes the cell degrees
   of freedom and stresses to an optimizer
 * Run a `BFGS <https://docs.ase-lib.org/ase/optimize.html#bfgs>`_
   optimization algorithm on the Frechet cell filter.

It also tells the optimizer to write a trajectory file, ``opt.traj``.

Once everything works and you run the relaxation task,
go to the task directory and use the ASE GUI (e.g. ``ase gui opt.traj``)
to visualize the trajectory.

GPAW also writes a log file to ``gpaw.txt``.
It is wise to have a brief look and observe that
the calculation used the parameters we expect.


Part 2: Add ground state and band structure tasks
=================================================

After the relaxation, we want to run a ground state
calculation to save a ``.gpw`` file, which we subsequently want
to pass to a non self-consistent calculation to get the band structure.

Add a ``groundstate()`` function to ``tasks.py``:

.. literalinclude:: tasks.py
   :pyobject: groundstate

In order to "return" the gpw file, we actually return a ``Path`` object
pointing to it.  When passing the path to another task, TaskBlaster
resolves it with respect to the task's own directory such
that the human will not need to remember or care about the actual directories
where the tasks run.

Next, add a ``groundstate()`` task to the workflow which calls the groundstate
function just added to ``tasks.py``.
By calling ``tb.node(..., atoms=self.relax)``, we can specify
that the atoms should be taken as the *output* of the ``relax`` task,
creating a dependency.

We can now run the workflow again.  The old task still exists and
will remain unchanged, whereas the new task should now appear
in the ``tree/groundstate`` directory.

Run the ground state task and check that the ``.gpw`` file was created as
expected.

In order to compute a band structure, we need to define a high-symmetry
band path in the Brillouin zone.
That can be done using the ``ase.cell.bandpath()`` method.  It is useful
to do this as a standalone task, so we can visualize it independently:

.. literalinclude:: tasks.py
   :pyobject: bandpath

Run the workflow and the resulting task.  Go to the directory.  The
band path object was saved in ASE's JSON format, so it can be visualized
using ASE's reciprocal cell tool::

  ase reciprocal tree/bandstructure/output.json


Finally, we write a band structure task in ``tasks.py`` which takes the
ground state (gpw file) and band path as an input:

.. literalinclude:: tasks.py
   :pyobject: bandstructure

Add the corresponding tasks to the workflow such that these computations
are composed and can run.

Now run the workflow and the resulting tasks.
The band structure object can again be visualized using one of the
ASE tools::

   ase band-structure tree/bandstructure/output.json


You can delete all the tasks with ``tb remove tree/`` and run them from
scratch by ``tb run tree/``, ``tb run tree/*``, or simply ``tb run
tree/bandstructure``.
The run command always executes tasks in
topological order, i.e., each task runs only when its dependencies
are done.

The ``tb ls`` command can also be used to list tasks in topological
order following the dependency graph::

  human@computer:~/myworkflow$ tb ls --parents tree/bandstructure/

  state    deps  tags        worker        time     folder
  ───────────────────────────────────────────────────────────────────────────────
  done     0/0               N/A-0/1       00:00:04 tree/relax
  done     1/1               N/A-0/1       00:00:01 tree/groundstate
  done     1/1               N/A-0/1       00:00:00 tree/bandpath
  done     2/2               N/A-0/1       00:00:05 tree/bandstructure

This way, we can comfortably work with larger numbers of tasks and we do
not have to care about running or changing them in the right order.

If we edit the workflow such that tasks receive different inputs,
TaskBlaster will mark the affected tasks with a conflict.
Such a conflict can be solved by removing the old calculations
or by telling TaskBlaster to consider it “resolved”, which we
have seen in a previous tutorial.


Part 3: Parametrize workflow over multiple materials
====================================================

Before continuing to work on our materials workflow it is instructive to go through the 
`“Workflow generator and for loops” <https://taskblaster.readthedocs.io/en/latest/tutorial/wf_generator/wf_generator.html>`_ 
tutorial in Taskblasters documentation. Once we have done that, we can then use dynamical 
workflows to parametrize our workflow over many materials for high-troughput calculations.

We next want to run the workflow to generate all the tasks for later
submission using myqueue or slurm.

Note that the syntax for some of these features is still being worked
on, and is a little bit convoluted.  This will be simplified in the
future, but for now we need some slightly verbose and specific
modifications.  First, create a workflow file which specifies
a parameterization:

.. literalinclude:: materials.py

Then include in ``tasks.py``:

.. literalinclude:: tasks.py
   :pyobject: ParametrizableMaterialsWorkflow


.. literalinclude:: tasks.py
   :pyobject: parametrize_materials_workflow


Run the workflow and observe how it creates an "initializer" task
and a "symbols" (finalizer) task.  Finalizers can be used to
run new things on some or all of the tasks generated by the workflow,
hence composing new workflows on top.

Running the task with ``tb run`` right now would start executing
all the computational tasks right away, which we probably do not want.
To run the workflow and generate its tasks *without* running them yet,
run e.g. the initializer explicitly::

    tb run tree/systems/init

This generates all the tasks, applying our previously developed
materials workflow to each material as a subworkflow.
Once the tasks are generated, we can run a few of the materials to
make sure everything works.  We can then look into how to submit
workers and use myqueue to execute the rest of them at larger scale.

Read the resources and tagging howto from the TaskBlaster documentation
and find a way to mark selections of tasks for execution as well as worker processes
to execute them.

Feel free to expand the selection of materials on which the workflow runs.


Part 4: Submitting workers
====================================================

So far we have been running our calculations with crude parameters so that the calculations finish 
quickly. If we want to calculate our material properties more accurately we'll have 
to increase the parameters. But quickly the computational cost becomes to much to do locally 
and when we run ``tb run .`` it only runs one task at a time. To really test this unrun all 
the relax tasks with ``tb unrun tree/*/relax``. This will also unrun all the tasks that depend on 
the relaxed structures. Then change the calculator parameter to something that gives more accurate 
results. For example:

.. code-block:: python
   
   calculator = {'mode': 'pw',
                 'ecut': 800,
                 'kpts': {'density': 8.0},
                 'txt': 'gpaw.txt'}

So fare we have only encountered three states that the tasks can be in; new, done and failed. Now we 
will need a fourth state, submitted. We need that because in taskblaster you don't submit a single 
task to a cluser. You submit a worker and this worker then picks up tasks to be computed. The workers 
only pick up tasks that have been submitted. To submit the relax tasks run ``tb submit tree/*/relax``. 
Running ``tb ls`` you should see that the atoms tasks are done, the relax tasks are queued and 
rest are in the new state.

Basic structure of submitting a worker is ``tb workers submit -R [cores]:[time]`` and one can then 
submit many of these workers and they will start picking up queued tasks. 

*Hint:* One most HPC clusters it is easier to get a job running if the time allocated is small. So 
it can be an advantage to try and guess how little time one needs. For example if you expect a task 
to take 30 min then submit it with time 1 hour of time instead of 1 day.


Part 5: Extract data from a workflow
====================================================

Sometimes we want to extract the data from a workflow into a single file. Maybe we 
want to plot it or send it to a colleague. For this example we'll want to extract the 
lattice constants of all the relaxed structures. To get the data out of a repository 
we use the ``Collect`` object. To use it we will create a new file called ``collect.py``.

.. literalinclude:: collect.py

This code collects all the relax tasks in a list of ``record`` objects. This object has 
several attributes that are of use to us

* *record.name*: task name relative to the repository tree.
* *record.directory*: path to the task directory.
* *record.inputs*: decoded task inputs, if all parents are available.
* *record.output*: decoded task output, or `None` if no output is available.
* *record.state*: task state.

So in the above example we access the output of the tasks which are the relaxed atoms objects. 
As an exercise modify the above script to also extract unrelaxed atoms which are passed into 
the relax task and compare them. 

*Hint:* Check what type ``r.input`` is
