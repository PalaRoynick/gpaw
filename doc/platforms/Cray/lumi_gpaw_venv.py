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

def ssh_run_list(cmds: str):
    cmd = " && ".join(cmds.split("\n"))
    run(f'ssh localhost "{cmd}"')
   

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('project', help='CSC Project name. Determines the EasyBuild path.')
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
    venv_name = args.venv

    install_root = Path(f'/projappl/{project_name}')
    easybuild_root = install_root / 'EasyBuild'
    G_partition_root = easybuild_root / 'modules/LUMI/24.03/partition/G'
    venv_root = install_root / venv_name
    activate = venv_root / 'bin/activate'

    # STEP 1: Set up easy build environment
    export_ebu_path = f'export EBU_USER_PREFIX={easybuild_root}'

    step1_prefix = f"""{export_ebu_path}
module load LUMI/24.03
module load partition/G
module load EasyBuild-user
"""

    modules = ['CuPy/13.4.1-cpeGNU-24.03-rocm-6.2.2',
               'magma/2.8.0-cpeGNU-24.03-rocm6.2.2',
               'libxc/7.0.0-cpeGNU-24.03']

    for module in modules:
        lua = G_partition_root / (module + '.lua')
        if lua.is_file():
            print(f'module {module} already found.')
        else:
            print(f'Installing {module}.')
            command = 'eb ' + module.replace('/', '-', 1) + '.eb -r'
            ssh_run_list(step1_prefix + command)

    # Make sure the virtual environment doesn't exist yet
    assert not (Path(install_root) / venv_name).is_dir()

    # STEP 2: Create the virtual environment and update actiavte script.
    step2 = f"""{export_ebu_path}
cd {install_root} 
module load cray-python/3.11.7
python3 -m venv --system-site-packages {venv_name}"""

    ssh_run_list(step2)
    print('Virtual environment succesfully created.')
   
    extra = f"""{export_ebu_path}
module load LUMI/24.03
module load partition/G
module load EasyBuild-user
module load cpeGNU/24.03
module load SuiteSparse/5.13.0-cpeGNU-24.03-OpenMP    # Dependency of hipSolver for ROCm 6.2
module load rocm/6.2.2
module load cray-fftw/3.3.10.7
module load buildtools-python/24.03-cray-python3.11
module load CuPy/13.4.1-cpeGNU-24.03-rocm-6.2.2       # from EBU_USER_PREFIX
module load magma/2.8.0-cpeGNU-24.03-rocm6.2.2        # from EBU_USER_PREFIX
module load libxc/7.0.0-cpeGNU-24.03                  # from EBU_USER_PREFIX
export MPICH_GPU_SUPPORT_ENABLED=1
"""
    activate.write_text(extra + activate.read_text())


    # STEP 3: Clone and build GPAW

    # clone locally, ssh within ssh is problematic
    run(f'cd {venv_root} && git clone git@gitlab.com:gpaw/gpaw.git')
    step3 = f"""source {activate}
cd {venv_root}
export GPAW_CONFIG={venv_root / 'gpaw/doc/platforms/Cray/siteconfig-lumi-gpu.py'}
echo GPAW CONFIF=$GPAW_CONFIG
pip install -v --log build-gpu.log -e gpaw"""

    ssh_run_list(step3)

if __name__ == '__main__':
    raise SystemExit(main())

