=======================
Computational workflows
=======================

In this exercise we will write and run computational workflows
with `Taskblaster <https://taskblaster.readthedocs.io/en/latest/>`_.

The basic unit of computation in Taskblaster is a *task*.  A task is a Python
function along with a specification of the inputs to that function.
The inputs can be either concrete values like lists, strings, or numbers,
or references to the outputs of other tasks.
Tasks that depend on one another and form a graph.  A
Taskblaster workflow is a Python class which defines such a graph, along with
metadata about how the tasks should run and be stored.  Workflows can
then be parametrized so they run on a collection of materials,
for example.

When using Taskblaster, we define workflows and tasks using Python.
However, the tools used to *run* workflows and tasks are command-line
tools.  Therefore, for this exercise we will be using the terminal
rather than the usual notebooks.  Basic knowledge of shell commands
is an advantage.

This exercise consists of multiple parts:

 * Introductory tutorials to Taskblaster
 * Write a workflow which defines a structure optimization task
   followed by electronic ground-state and band structure computation
   using GPAW
 * Parametrize the workflow to comprise multiple materials
 * Submit multiple workers to execute the computational tasks at scale


Part 0: The Taskblaster tutorials and documentation
===================================================

Taskblaster comes with its own documentation and tutorials.
These are not related to physics or materials, but provide a generic
introduction to the framework:

 * To get started, go through the basic
   `Taskblaster tutorials <https://taskblaster.readthedocs.io/en/latest/tutorial/module.html>`_,
   specifically
   `“Hello world” <https://taskblaster.readthedocs.io/en/latest/tutorial/hello_world/hello_world.html>`_ and
   `“My first workflow” <https://taskblaster.readthedocs.io/en/latest/tutorial/my_first_workflow/my_first_workflow.html>`_.
   Make sure you can run the examples and feel free to explore
   the command-line interface.

 * It may be illustrating to read the explanation page on
   `Basic Concepts <https://taskblaster.readthedocs.io/en/latest/explanation/basic_concepts/basic_concepts.html>`_.

We have now acquired a basic familiarity with Taskblaster commands,
although we may still not see the full picture of how one runs
scientific high-throughput projects in practice.  We will deal with this
over the course of the next exercises.


Part 1: Simple materials workflow
=================================

In this exercise we will develop a first materials workflow using the
GPAW electronic structure code.
This exercise will be more open-ended.
Be sure to make good use of the different command-line tools'
:option:`--help` pages as well as Taskblaster's online documentation.

Taskblaster itself does not know anything about electronic structure,
materials, atoms, or ASE.
However we will want tasks to have ASE objects as input
and output, and that requires being able to save those objects.
Taskblaster can be extended with the ability to encode and decode arbitrary
objects.  Doing so requires a plugin.
To enable basic support for numpy and ASE objects, use the
`ase-taskblaster <https://gitlab.com/taskblaster/ase-taskblaster>`__ plugin.
We will create a project using using this plugin.
Make sure ``ase-taskblaster`` is installed by running, e.g.,
``pip install ase-taskblaster``.

Go to a clean directory and create a repository using
`asetaskblaster <https://pypi.org/project/ase-taskblaster/>`
as plugin::

  $ tb init asetaskblaster

This creates a ``tree/`` directory where tasks will be stored.
You can use the ``tb info`` command to see global information about
the repository and verify that it uses ``asetaskblaster``.

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
immediately rerun a failed task. The workflow should function
correctly when called on a bulk silicon system like this:

.. literalinclude:: src/myworkflow.py
   :pyobject: workflow

Here we have chosen some generic GPAW parameters that will be fine
for testing, but not for production.

It must be possible for Taskblaster to import the tasks and workflows,
i.e., the Python code must be installed or otherwise visible to Python.
Taskblaster will always look for code inside the ``src/`` subdirectory
of the repository (``tb info`` shows this as “Pythonpath”.), which
can be handy for writing quick examples.
So put the code in a Python file under ``src/`` and name the files
in a way that does not conflict with other Python packages.
We'll assume that the file is ``src/myworkflow.py``.


For a real project, it is a good idea to develop the code
using version control and ship it as a proper Python package.

The relaxation task can be implemented like this:

.. literalinclude:: src/myworkflow.py
   :pyobject: optimize_cell

Users unfamiliar with ASE may want to take a few moments to look up
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
It is a good idea to have a brief look and verify that
the calculation used the parameters we wanted.


Part 2: Add ground state and band structure tasks
=================================================

After the relaxation, we want to run a ground state
calculation to save a ``.gpw`` file, which we subsequently want
to pass to a non self-consistent calculation to get the band structure.

Add a ``groundstate()`` function to ``src/myworkflow.py``:

.. literalinclude:: src/myworkflow.py
   :pyobject: groundstate

In order to "return" the gpw file, we actually return a ``Path`` object
pointing to it.  When passing the path to another task, Taskblaster
resolves it with respect to the task's own directory such
that the human will not need to remember or care about the actual directories
where the tasks run.

Next, add a ``groundstate()`` task to the workflow which calls the groundstate
function just added to ``myworkflow.py``.
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

.. literalinclude:: src/myworkflow.py
   :pyobject: bandpath

Run the workflow and the resulting task.  Go to the directory.  The
band path object was saved in ASE's JSON format, so it can be visualized
using ASE's reciprocal cell tool::

  ase reciprocal tree/bandpath/output.json


Finally, we write a band structure task in ``src/myworkflow.py`` which takes the
ground state (gpw file) and band path as an input:

.. literalinclude:: src/myworkflow.py
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
Taskblaster will mark the affected tasks with a conflict.
Such a conflict can be solved by removing the old calculations
or by telling Taskblaster to consider it “resolved”, which we
have seen in a previous tutorial.


Part 3: Parametrize workflow over multiple materials
====================================================

Before continuing to work on our materials workflow it is instructive to go through the
`“Workflow generator and for loops”
<https://taskblaster.readthedocs.io/en/latest/tutorial/wf_generator/wf_generator.html>`_
tutorial in Taskblasters documentation. Once we have done that, we can then use dynamical
workflows to parametrize our workflow over many materials for high-troughput calculations.

We next want to run the workflow to generate all the tasks for later
submission using myqueue or slurm.

Note that the syntax for some of these features is still being worked
on, and is a little bit convoluted.  This will be simplified in the
future, but for now we need some slightly verbose and specific
modifications.  First, create a workflow file which specifies
a parameterization and save it in e.g. ``src/materials.py``:

.. literalinclude:: src/materials.py

Then include in ``src/myworkflow.py``:

.. literalinclude:: src/myworkflow.py
   :pyobject: ComputeMaterial


.. literalinclude:: src/myworkflow.py
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

Read the resources and tagging howto from the Taskblaster documentation
and find a way to mark selections of tasks for execution as well as worker processes
to execute them.

Feel free to expand the selection of materials on which the workflow runs.


Part 4: Extract data from a workflow
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
