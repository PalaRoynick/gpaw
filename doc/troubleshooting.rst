.. _troubleshooting:

Troubleshooting
===============

::

    $ gpaw info
    $ gpaw test

If tests pass, and the parallel version is built, test the parallel code::

    $ mpiexec -n 2 gpaw test
