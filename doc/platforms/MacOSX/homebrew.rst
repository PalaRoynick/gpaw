.. _homebrew:

========
Homebrew
========

.. highlight:: bash

Install the Xcode command line tools with the command::

    $ {
    >     xcode-select --print-path  # This checks if you already have it
    >     || xcode-select --install  # This installs it if not
    > }

Install `Homebrew from the source <https://brew.sh/>`_::

    $ /bin/bash -c "$(
    >    curl -fsSL
    >    https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh
    > )"
    $ echo 'export PATH=/usr/local/bin:$PATH' >> ~/.bash_profile

Install ASE and GPAW dependencies::

    $ python="python"
    $ blas="openblas"  # Or any equivalent
    $ libraries=( libxc "${blas}" open-mpi fftw )
    $ brew install "${python}" gcc "${libraries[@]}"

.. note::
    - Specific versions of Python can be installed with
      ``python@<version>``.  For Tkinter support (which is no longer
      shipped by default), use ``python-tk@<version>``.  Note that
      Tkinter is required for using ASE's graphical user interface
      (GUI).
    - While GPAW functions without a C-level BLAS library, it is
      *highly* recommended that one be installed.

Install :py:mod:`pip` (should not be necessary for most
installations)::

    $ python -m ensurepip

.. py:module:: pip

    ..
        FIXME: This is just a hack to get :py:mod:`pip` to link
        *somewhere* because PIP's :file:`objects.inv` doesn't contain
        itself.

    .. seealso::

        `The pip documentation`_

Create and activate a virtual environment with :py:mod:`venv` (or other
equivalent; optional but *highly* recommended)::

    $ # Name the environment anything you want -- it just has to be a
    $ # valid name for a new directory
    $ venv='gpaw'
    $ # Details depend on the shell you use
    $ activation_script="${venv}/bin/activate"
    $ python -m venv "${venv}" && source "${activation_script}"

.. note::
    - .. _externally-managed-env:

      Depending on the system and Python and :py:mod:`pip` versions, it
      the system-wide Python may be considered `externally managed`_,
      making a virtual environment *necessary* for installing Python
      packages via :py:mod:`pip` without resorting to "hacks" like ::

          alias pip="pip --user --break-system-packages"

    - It may be useful to patch the virtual-environment activation
      script to ensure that the installed libraries are available at the
      runtime via the ``${LD_LIBRARY_PATH}`` environment variable::

          $ _gpaw_venv_libraries=( "${libraries[@]}" )
          $ declare -p _gpaw_venv_libraries >>${activation_script}"
          $ cat >>"${activation_script}" <<-'!'
          > for lib in "${_gpaw_venv_libraries[@]}"; do
          >     lib_path="$(brew --prefix "${lib}")/lib"
          >     if [ -n "${LD_LIBRARY_PATH}" ]; then
          >         LD_LIBRARY_PATH+=:
          >     fi
          >     export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}${lib_path}"
          > done
          > !

Install the required Python packages::

    $ pip install numpy scipy matplotlib

.. note::
    - Outside of a virtual environment, some packages (e.g.
      :py:mod:`numpy` and :py:mod:`scipy`) should be installed via
      :command:`brew` instead of :command:`pip`.  However, as discussed
      :ref:`above <externally-managed-env>`, virtual environments may be
      necessary for using :py:mod:`pip` at all.
    - :py:mod:`matplotlib` is not *strictly* required, but is used for
      some visualization.

Install (and optionally test) ASE::

    $ test_ase=true  # `false` if you don't want to run tests
    $ if ${test_ase}; then
    >     pip install --upgrade ase[test]
    >     python -m ase test
    > else
    >     pip install --upgrade ase
    > fi

.. note::
    For installing the bleeding-edge version, one can
    `install from the repository`_ ::

        pip install ase @ git+https://gitlab.com/ase/ase.git

    The same goes for GPAW – just replace all ``ase`` in the above
    command with ``gpaw``.  See also :ref:`development workflow`.

Install GPAW::

    $ pip install --upgrade gpaw

.. note::
    If this simple installation fails, you may have to look into setting
    up the appropriate :file:`siteconfig.py` file.  Depending on your
    setup, it could be as simple as

    .. literalinclude:: siteconfig.py

    See also :ref:`siteconfig`.

Test GPAW::

    $ gpaw test  # Serial
    $ gpaw -P 4 test  # Parallel

.. _externally managed: https://packaging.python.org/en/latest/specifications/externally-managed-environments/
.. _install from the repository: https://pip.pypa.io/en/latest/topics/vcs-support/
.. _The pip documentation: https://pip.pypa.io/en/stable/index.html
