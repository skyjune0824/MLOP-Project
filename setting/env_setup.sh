#!/bin/bash

# Set environment variables
ENV_NAME="traffic"
PYTHON_VER="3.10"

# Create a Conda environment
conda create --name "$ENV_NAME" python="$PYTHON_VER" --yes

# Activate the Conda environment
eval "$(conda shell.bash hook)"  # Initialize Conda for bash
conda activate "$ENV_NAME"

# Check if requirements.txt exists and install packages
if [ -f "requirements.txt" ]; then
    # Install packages from requirements.txt
    pip install -r requirements.txt
else
    echo "requirements.txt not found. No packages installed."
fi

echo "Conda environment '$ENV_NAME' created and dependencies installed."
echo "You can activate the environment using 'conda activate $ENV_NAME'."
