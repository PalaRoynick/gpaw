.. _ipad:

==============
 Using an iPad
==============

Using an iPad for the computer exercises is **not recommended,** mainly
because you cannot run the ASE graphical user interface.  But it is a
possibility if there are no other options.


Terminal app
============

A terminal app that is known to work is Termius.

First you need to install your secret key (guest accounts only, DTU users
need to generate a new key).  Open a browser, go to
https://www.student.dtu.dk/~dccidmgr/CAMD24/ then add your user name to the
end of the URL and press enter.  You will now have to enter your user name
and password.  Click on the provided link (saving it will not help, you
need to use cut-and-paste).  Once the key is displayed (a lot of randomish
letters between a header and a footer line) select it all including header
and footer, and copy it.

Then start Termius, in the left menu click *Keychain*.  Choose the
plus sign at the top of the page, and choose *Paste key*.  Paste the
key you copied from the browser into the field called *Private Key*.
Give your new key a label (e.g. DTU), and if you feel paranoid you can
also set a password for it.  Save it.

Create a login by choosing ``Hosts`` in the sidebar, then select the Plus
sign at the top.  Select "New Host"

* Label: DTU gbar (or whatever you want)
* IP or Hostname: login.gbar.dtu.dk
* Use SSH should be turned on.
* Username: Your DTU user name
* Password: Your DTU password
* Key: Select the private key you just created.

Click on the new profile, you should be logged in.

Now follow the instructions for :ref:`logging in the first time with a Mac
<setuplinmac>`, with the important change that the command ``linuxsh -X``
should be replaced with ``linuxsh``.  The omitted ``-X`` makes it possible
for the graphical user interface to show atomic structures on your screen,
that does not work on an iPad and a workaround is described below.


Viewing atomic structures
=========================

The ``view(atoms)`` command in ASE will not work for you, as you do not
have an  X11 server on your ipad.  Instead use the ``plot_atoms`` command::

  from ase.visualize.plot import plot_atoms
  plot_atoms(atoms)

You rotate the atoms like this (45 degrees along the x-axis)::

  plot_atoms(atoms, rotation='45x')
