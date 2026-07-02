The scripts in this folder are used to set up the G Databar accounts
for the students at the CAMD summer school.  It is used in this way:

In the account of Jakob Schiøtz (jasc) a folder is made named
CAMD2026.  In that folder a clone of the GPAW git repository is made:
~jasc/CAMD2026/gpaw

In the home folder (~jasc) a softlink is placed to
~jasc/CAMD2026/gpaw/doc/summerschools/summerschool26/setup_scripts/setup2026

When the students source ~jasc/setup2026, that script first executes
setup2026_student and then activates the created venv ~/venv_CAMD2026

The script installs the Python packages necessary to run GPAW and the
summer school exercises.  In addition, it installs the Spyder IDE as
version installed in the databar does not work with a virtual
environment.
