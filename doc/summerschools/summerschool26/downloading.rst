Download your files before leaving Friday!
==========================================

Remember to download your files before you leave your summer school, as they will otherwise be deleted.

Windows
-------

If you use MobaXTerm, you can drag-and-drop files and folders from the left-hand panel to your local computer.

**WARNING: Do not copy the entire CAMD2026 folder** as that will copy your venv sub-folder with all the installed software. It will be a multi-gigabyte download taking hours, and the files will be useless.

If you use VScode, select the files/folders you want do download, right-click and select download.  **The warning above still applies**

If you have both, then VScode seems to be faster and more reliable.

Linux
-----

You can mount the files on your local laptop by starting a local file browser and typing the address ``sftp://login.gbar.dtu.dk/camdXXXXX`` (use your own username).  Then log in with username and password.

If this fails, see the manual copying below.

Mac
---

If you have VSCode installed, login to the remote server and follow the instructions for Windows above.

You can also install firezilla and connect to the address mentioned for linux above (untested by us!).

Finally, you can copy manually as mentioned below.

Manual copy (Mac and Linux)
---------------------------

When logged in to the remote machine, make an archive file with the command::

    tar cvf $HOME/transfer.tgz CAMD2026

(the warning above about including the venv does not apply here, only a useless link to it will be included in the archive).

Now, in a terminal on your local machine you can transfer the archive with the command::

    scp camdXXXXX@login.gbar.dtu.dk:transfer.tgz .

(note the dot at the end of the command line, that instructs ``scp`` to dump the file in the current folder).

You can then unpack with::

    tar xvf transfer.tgz

IMPORTANT
---------

Check that you got all the files you expected, and do not hesistate to ask for help!
