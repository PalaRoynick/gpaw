.. _gbar using:

=====================================
Using the DTU computers interactively
=====================================

Logging using SSH (Windows)
===========================

If you are not already logged into the databar, do so by starting
MobaXterm.  There should be a session available from the welcome
screen of MobaXterm named ``login.gbar.dtu.dk`` or similar, created
when you logged in the first time.  Click on it to log in again.

Once you are logged in on the front-end, get a session on an interactive compute node by typing the command::

  linuxsh -X

You may want to open more than one connection, just remember to run ``linuxsh`` in them all.


Logging using SSH (Mac/Linux)
=============================

If you are not already logged into the databar, do so by starting
a Terminal window.  Log in to the databar front-end with the command::

  ssh -XY USERNAME@login.gbar.dtu.dk

Replace ``USERNAME`` with your DTU username.

Once you are logged in on the front-end, get a session on an interactive compute node by typing the command::

  linuxsh -X

You may want to open more than one Terminal window, just connect again in the new one (and don't forget to run ``linuxsh``).


Logging in using ThinLinc
=========================

Start the ThinLinc client (on the usual menus in Windows/MacOS, use the ``tlclient`` command on Linux).

Once logged in, connect to a compute note by clicking on the Terminal button at the bottom of the desktop.


Editing Python files
====================

Several editors can be used to edit Python files:

* **Spyder** is an Integrated Development Environment for Python.  You can edit your files, run them, and see plots within Spyder.


* **gedit**: A very simple editor for Python files.  You need to run the files manually in the terminal window.

* **nano**: A very simple editor that does not open a new window, but runs within the terminal window.

* **Emacs** and **vim**: Stay clear of these unless you already know them!


Using Spyder
------------

Start it by running the  command ``spyder -p .`` in the ``CAMD2026`` folder - the ``-p`` and the dot means open the current folder as a project::

    cd CAMD2026
    spyder -p .

ThinLinc users: Please open ``spyder`` with the command line above.  You may be able to find an ancient version in one of the application menus, but that will not work with the virtual environment.  The correct version is version 6, the old one is version 5.  The version number is displayed on the splash screen.

You can run the selected file by clicking on the green Run triangle.  Any plots will be shown in the Plot pane of the upper right panel.


Using gedit
-----------

You can start gedit by typing ``gedit filename.py &`` in the command window.  The ampersand after the file name is optional, it puts the command in the background so you can still use the command line windows for other things (although gedit may print a warning or two that then appears out of context).

ThinLinc users can also open the file browser and opening gedit by clicking on the Python file (the desktop environment calls the editor Text Edit instead of gedit).

To run a python file, type ``python filename.py`` in the command window.


Using nano
----------

Type ``nano filename.py`` to start the nano editor.

