#!/bin/bash

ENV_NAME="traffic"

python3 -m venv $ENV_NAME
source $ENV_NAME/bin/activate

if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "requirements.txt not found. No packages installed."
fi

echo "Virtual environment '$ENV_NAME' created and dependencies installed."
