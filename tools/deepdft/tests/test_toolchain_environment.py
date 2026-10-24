"""Regression checks for inherited HPC-X/MPI settings in a private build."""
import os
from pathlib import Path
import subprocess


HELPER = Path(__file__).resolve().parents[1] / 'isolate_toolchain.sh'


def test_host_overrides_removed_without_losing_gpu_allocation():
    environment = os.environ.copy()
    environment.update(
        OMPI_CXX='/site/nvidia/hpcx/bin/nvc++',
        PMIX_MCA_component_path='/site/hpcx/pmix',
        LDFLAGS='-L/site/hpcx/ompi/lib',
        LIBRARY_PATH='/site/hpcx/ompi/lib',
        LD_LIBRARY_PATH='/site/hpcx/ompi/lib:/site/nvidia/cuda/lib64:/site/driver/lib64',
        CUDA_VISIBLE_DEVICES='4',
        SLURM_JOB_NAME='allocated-density-training',
    )
    script = r'''
set -euo pipefail
source "$1"
deepdft_clean_host_toolchain
[[ ! -v OMPI_CXX && ! -v PMIX_MCA_component_path && ! -v LDFLAGS && ! -v LIBRARY_PATH ]]
[[ $LD_LIBRARY_PATH == /site/nvidia/cuda/lib64:/site/driver/lib64 ]]
[[ $CUDA_VISIBLE_DEVICES == 4 && $SLURM_JOB_NAME == allocated-density-training ]]
'''
    subprocess.run(['bash', '-c', script, 'environment-test', str(HELPER)],
                   env=environment, check=True)


def test_prefix_priority_survives_repeated_activation():
    script = r'''
set -euo pipefail
source "$1"
export PATH=/usr/bin:/bin
export LD_LIBRARY_PATH=/site/driver/lib64
unset LIBRARY_PATH
deepdft_prioritize_prefix /private/toolchain
deepdft_prioritize_prefix /private/toolchain
[[ $PATH == /private/toolchain/bin:/usr/bin:/bin ]]
[[ $LD_LIBRARY_PATH == /private/toolchain/lib:/site/driver/lib64 ]]
[[ $LIBRARY_PATH == /private/toolchain/lib ]]
'''
    subprocess.run(['bash', '-c', script, 'environment-test', str(HELPER)], check=True)
