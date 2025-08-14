#!/usr/bin/env python3
"""Install gpaw on LUMI in a virtual environment.

"""
import argparse
import os
import shutil
import subprocess
from pathlib import Path


def run(cmd: str, **kwargs) -> subprocess.CompletedProcess:
    print(cmd)
    return subprocess.run(cmd, shell=True, check=True, **kwargs)



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('project', help='CSC Project name. Determines easybuild path.')
    parser.add_argument('venv', help='Name of venv.')
    parser.add_argument('--dftd3', action='store_true',
                        help='Also build DFTD3.')
    parser.add_argument('--gpaw-branch', default='master',
                        help='Check out a particular gpaw branch')
    parser.add_argument('--run-benchmarks', action='store_true',
                        help='Also submit the GPAW benchmark suite'
                             ' with the checked out branch.')
    args = parser.parse_args()
    project_name = args.project


    install_root = f'/projappl/{project_name}'
    easybuild_root = install_root + '/EasyBuild'
    G_partition_root = easybuild_root + '/modules/LUMI/24.03/partition/G'
    step1 = f"""export EBU_USER_PREFIX={easybuild_root}
module load LUMI/24.03
module load partition/G
module load EasyBuild-user
"""

    modules = ['CuPy/13.4.1-cpeGNU-24.03-rocm-6.2.2',
               'magma/2.8.0-cpeGNU-24.03-rocm6.2.2',
               'libxc/7.0.0-cpeGNU-24.03']

    for module in modules:
        lua = G_partition_root + '/' + module + '.lua'
        if Path(lua).is_file():
            print(f'module {module} already found.')
        else:
            print(f'Installing {module}.')
            command = 'eb ' + module.replace('/', '-', 1) + '.eb -r'
            cmd = " && ".join((step1 + command).split("\n"))
            run(f'ssh localhost "{cmd}"')

if __name__ == '__main__':
    raise SystemExit(main())

