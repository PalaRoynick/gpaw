"""This file checks that GPAW is not running on a compute node.

It is installed as ~/gpaw/rc.py by the installation script.
"""
import os

_hostname = os.getenv('HOSTNAME', '')
if 'login' in _hostname:
    raise RuntimeError(f'\n\nERROR:\nAttempting to run GPAW on login node {_hostname}\nDid you forget to run linuxsh?\n')
