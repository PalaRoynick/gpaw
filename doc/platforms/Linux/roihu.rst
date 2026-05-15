.. _roihu:

====================
Roihu (roihu.csc.fi)
====================

.. note::
   These instructions are up-to-date as of May 2026.

Stable GPAW releases
====================

Roihu has several versions of GPAW available as modules. You can load the
browse the available versions with ``module spider gpaw``.

Developer installation
======================

Developer installation on Roihu-CPU
-----------------------------------

These instructions are for installing GPAW main development branch newer
than April 18, 2026, which requires a C++ compiler.

The following steps build GPAW in a Python virtual environment:

.. code-block:: bash

  # Move to installation directory of your choice
  cd /projappl/project_.../$USER

  # Load the GPAW module (used for all dependencies)
  module load gpaw/...  # Use the desired version

  # Create virtual environment
  python3 -m venv --system-site-packages venv-gpaw-cpu

  # Activate venv
  source venv-gpaw-cpu/bin/activate

  # Update build tools
  python3 -m pip install --upgrade pip setuptools packaging pybind11

  # Create constraints for pip so that we don't accidentally override system provided packages
  python3 -m pip list --format=freeze | grep -v -E '^(gpaw|ase)' | tee $(dirname $(which pip))/../constraints.txt

  # Clone GPAW development repository
  git clone https://gitlab.com/gpaw/gpaw.git
  cd gpaw

  # Set build configuration for GPAW
  export GPAW_CONFIG=$(readlink -f doc/platforms/Linux/siteconfig-roihu-cpu.py)

  # Install GPAW. Leave the '-e' out if you don't want an editable install
  rm -rf _build build* _gpaw.*.so gpaw.egg-info
  GPAW_BUILD_JOBS=8 python3 -m pip install --no-build-isolation --constraint $(dirname $(which pip))/../constraints.txt -v --log build-cpu.log --config-settings editable_mode=compat -e .


Note that above the siteconfig file is obtained from the cloned git repository.
Alternatively, you can download the siteconfig file from here:
:download:`siteconfig-roihu-cpu.py`.

Usage on Roihu-CPU
------------------

.. code-block:: bash

  module load gpaw/...  # Use the same version
  source venv-gpaw-cpu/bin/activate
  gpaw info


Developer installation on Roihu-GPU
-----------------------------------

TODO
