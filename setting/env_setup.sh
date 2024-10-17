#!/bin/bash

ENV_NAME="traffic"
PYTHON_VER="3.12"

# Create a Conda environment
conda create --name $ENV_NAME python=$PYTHON_VER --yes

# Activate the Conda environment
source activate $ENV_NAME

if [ -f "requirements.txt" ]; then
    # Install packages from requirements.txt
    pip install -r requirements.txt
else
    echo "requirements.txt not found. No packages installed."
fi

echo "Conda environment '$ENV_NAME' created and dependencies installed."
echo "Environment will activate using 'conda activate $ENV_NAME'."
