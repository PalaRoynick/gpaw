Using VScode to edit Python files
=================================

If you find the Spyder is too slow when 80 people are connecting from the same venue, then you may want to use VScode on your local computer as an editor.

If you have a Windows computer you still need a MobaXTerm to run the code in order to make the ASE Graphical User Interface work.

Getting VSCode to talk to the DTU computers
-------------------------------------------

* Install the VScode extension called *remote-SSH*
* Use Ctrl-P to get a promt, then choose *Remote-SSH: Connect to host*.  One option you get is to "Configure SSH Hosts".  Choose that.  The file that you open should be edited to look like this::

    Host login.gbar.dtu.dk
        HostName login.gbar.dtu.dk
        User camdXXXXX
        IdentityFile  C:\Users\XXXXX\Documents\camd26092

On Mac/Linux, it should look like this::

    Host login.gbar.dtu.dk
        HostName login.gbar.dtu.dk
        User camdXXXXX
        IdentityFile  ~/.ssh/camd26092
        ForwardX11 yes
        ForwardX11Timeout 36000    

Make sure that IdentityFile points to where you stored your SSH key, and that User is your user name, not the fake one shown here.

Opening a remote connection
---------------------------

You can now choose the remote connection named *login.gbar.dtu.dk*.  When prompted for the remote operating system, choose Linux.  Then you can open a remote folder.  It is normal that you are prompted for your DTU password twice during the process.

Running stuff
-------------

If you run stuff in the VSCode terminal panel, the ASE GUI will not work.  So use MobaXTerm to log in and run stuff.

