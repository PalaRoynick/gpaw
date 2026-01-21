#!/bin/bash

# thanks copilot

if [[ "$1" != "lumi" && "$1" != "mahti" ]]; then
    echo "Usage: $0 [lumi|mahti]"
    exit 1
fi
TARGET_REMOTE="$1"

# NB: trailing / is important
LOCAL_DIR="/home/laurinie/gpaw_project/gpaw_fork/"

if [[ "$TARGET_REMOTE" == "lumi" ]]; then
    REMOTE_HOST="lumi"
    REMOTE_DIR="/scratch/project_462000007/laurinie/gpaw_rsync"
elif [[ "$TARGET_REMOTE" == "mahti" ]]; then
    REMOTE_HOST="mahti"
    REMOTE_DIR="/scratch/project_2002078/laurinie/gpaw_rsync"
fi

# Rsync options:
# -a : archive mode (preserves permissions, symlinks, etc.)
# -v : verbose
# -z : compress data during transfer
# --delete : remove remote files not present locally
# --progress : show progress
# --exclude : optional exclusions
rsync -avz --delete \
    --exclude=".git/" \
    --exclude="__pycache__/" \
    --exclude="build/" \
    --exclude="_build/" \
    --exclude="Makefile" \
    --exclude="*.egg-info/" \
    --exclude="doc/" \
    --exclude="siteconfig*.py" \
    --exclude="*.out" \
    --exclude="*.so" \
    --exclude="rsync_remote.sh" \
    --exclude="rsync_files.txt" \
    --exclude=".vscode/" \
    --exclude="*.code-workspace" \
    --exclude="*cache*" \
    --exclude="gpaw/test/benchmarks/data/*" \
    "$LOCAL_DIR" \
    "$REMOTE_HOST:$REMOTE_DIR"
