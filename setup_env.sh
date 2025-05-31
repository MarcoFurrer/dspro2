#!/bin/bash

python3 -m venv .venv

source .venv/bin/activate

# Add the kernel so it shows up in Jupyter
python -m ipykernel install --user --name=$ENV_NAME --display-name "Python (.venv)"

pip install -r requirements.txt

