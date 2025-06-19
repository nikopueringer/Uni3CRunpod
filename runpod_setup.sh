#!/bin/bash

# This script automates the setup for a new Runpod pod instance.
# It handles navigation, system package installation, and conda environment activation.

# IMPORTANT: To make the conda environment activation apply to your current terminal session,
# you should run this script using the 'source' command:
#
#   source runpod_setup.sh
#
# Running it as a normal script (e.g., './runpod_setup.sh') will not make the
# 'uni3c' environment active in your current terminal after the script finishes.

echo "--- Starting Runpod Setup ---"

# This script is designed to be run from within the project directory.
# The following command ensures that we are in the correct directory
# by changing to the script's location, even if sourced from elsewhere.
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR"
echo "Running setup from project directory: $(pwd)"

# Update apt and install system dependencies from README.md
echo "Updating apt package list and installing system dependencies..."
sudo apt-get update && sudo apt-get install -y tmux ffmpeg libsm6 libxext6 libglm-dev

# Set Hugging Face cache directory to a persistent location
export HF_HOME=/workspace/huggingface_cache
mkdir -p $HF_HOME
echo "Hugging Face cache directory set to $HF_HOME"

# Activate the 'uni3c' Conda environment
# User specified the correct activation method and path.
CONDA_ACTIVATE_SCRIPT="/workspace/Miniconda3/bin/activate"
if [ -f "$CONDA_ACTIVATE_SCRIPT" ]; then
    echo "Activating 'uni3c' conda environment..."
    source "$CONDA_ACTIVATE_SCRIPT" uni3c
    echo "'uni3c' environment is now active."
else
    echo "Error: Conda activation script not found at $CONDA_ACTIVATE_SCRIPT."
    echo "Please verify your Miniconda installation path."
    return 1 2>/dev/null || exit 1
fi

echo "--- Setup Complete ---"
echo "You are in $(pwd) with the 'uni3c' environment active." 