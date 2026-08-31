.. _ipad:

==============
 Using an iPad
==============

Using an iPad for the computer exercises is **not recommended,** mainly
because you cannot run the ASE graphical user interface, and because viewing plots is tedious.

But it is a possibility **if there are no other options.**


Terminal app
============

A terminal app that is known to work is Termius.

First you need to install your secret key (guest accounts only, DTU users
need to generate a new key).  Open a browser, go to
https://www.student.dtu.dk/~dccidmgr/CAMD26/ then add your user name to the
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

Click on the new profile, you should be logged in.  Run the ``linuxsh`` command to
connect to an interactive compute node.

Finding the control keys etc
----------------------------

On the left pane, you can find control keys etc needed to navigate the Linux command line.  See the screen shot:

.. image:: Termius_ipad.png
   :width: 100%

Get access to the software
==========================

After logging in, and running ``linuxsh`` to connect to a compute node, you can install the software by running the command::

    source ~jasc/setup2026

You only need to do this once!  Next time you log in, just run ``linuxsh`` to connect to an interactive compute node, and the software is available.

Viewing plots
=============

When running on the ipad, you need to save the plots as ``.png`` files with the ``fig.savefig()`` matplotlib command.  Then you can view them in an SFTP pane using Termius.

**Opening an SFTP connection:** On the tab labeled "Personal", choose the hosts section.  Press the three dots next to the name of the host, and select "Connect via SFTP".  Now you can see the file system on the DTU computer system, and you can navigate to the folder where your figures are.  Clicking on a ``.png`` file will display it.


Viewing atomic structures
=========================

The ``view(atoms)`` command in ASE will not work for you, as you do not
have an  X11 server on your ipad.  Instead use the ``plot_atoms`` command::

  from ase.visualize.plot import plot_atoms
  plot_atoms(atoms)

You rotate the atoms like this (45 degrees along the x-axis)::

  plot_atoms(atoms, rotation='45x')

You need to save the figure and view it through the SFTP connection.
