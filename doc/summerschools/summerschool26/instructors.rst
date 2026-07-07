.. _instructors:

============================
Instructions for instructors
============================

The DTU databar
===============

The summer school exercises will be running in the DTU Databar, not on
Niflheim.  You can log in with your DTU username and password.  Like
Niflheim, it has a login node and a number of compute node, but unlike
Niflheim it also has *interactive compute nodes* where the students
and you can run e.g. your Jupyter notebooks.

Logging in
----------

::

   ssh -X login.gbar.dtu.dk

Then proceed from the login node to an interactive compute node with
the command::

  linuxsh -X


Creating an SSH key pair (while at DTU!)
----------------------------------------
.. _keypair:

When you are on the DTU network (or use a VPN) logging in requires
authentication with *either* a password *or* a cryptographic key.
But once we are off-campus, it requires *both*.  It is much easier to
set up while still on DTU.

Creating a key pair on Mac or Linux
...................................

First check if you already have a public key on your laptop.  In a
terminal, type::

  cd
  ls -lh .ssh

Look for files named ``id_rsa`` and ``id_rsa.pub``.  If they are not
there, create them with the command::

  ssh-keygen -b 4096 -t rsa

When prompted for a file name, just keep the default.  When asked for
a password, you can choose one if you feel paranoid / security minded,
or leave it blank to avoid having to type a password when you use the
key.

Once the file have been created, log in to the databar and type the
following commands::

  cd
  mkdir -p .ssh
  nano .ssh/authorized_keys

In that file, paste the content of the file ``id_rsa.pub`` from your
laptop.  Press Ctrl-X to exit the "nano" editor, and choose Yes to
save the file.

Not open a new window and try logging in to the databar.  If you are
at DTU, you should no longer be prompted for a password.


Creating a key pair on Windows
..............................

Open MobaXterm.  On the Tools menu, choose MobaKeyGen

Leave parameters as default (RSA, 2048 bits).  Click Generate, and
move the mouse to generate randomness.  When the key is ready, click
"Save private key".  Choose a name that makes sense.  You can set a
passphrase if you feel paranoid / security-minded.

Keep the MobaKeyGen window open (if you close it, you can load the
private key again).  Log in to the DTU databar.  On the databar, run
the commands::

  cd
  mkdir -p .ssh
  nano .ssh/authorized_keys

Now copy the private key from the top field of the MobaKeyGen window
(with Ctrl-C), and then paste it into the nano editor by
right-clicking on the window.  Once the key is pasted, press ENTER so
the line ends with a line break.  Then exit the "nano" editor by
pressing Ctrl-X, say Yes to save the file.

Now log out of the databar and close the MobaKeyGen window.

Click on the Settings button (one of the last ones on the MobaXTerm
toolbar). On the configuration window that opens you select the SSH
tab. The last box is labeled “SSH agents”, select “Use internal SSH
agent MobAgent”.

Click the Plus sign next to the box labeled “Load the following keys
at MobAgent startup” and then select the file you just saved with your
private SSH key.  Then press OK; MobaXterm will then need to restart.


Setting up the Virtual Environment etc
--------------------------------------

To prepare your account, run the script (you only need to do this
once)::

  bash ~jasc/setup2026

(This is the same script the students will be running to set up their account).

This will create a folder ``CAMD2026``, a virtual environment in
``CAMD2026/venv``, and a MyQueue configuration.

A line is added to your .bashrc to automatically activate the virtual
environment.  Remove it and activate manually if it causes trouble.
