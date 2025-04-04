#!/bin/bash

ENV_NAME=numerai-env

# Create the conda environment
conda env create -f environment.yml

# Activate it
conda activate $ENV_NAME

# Install ipykernel (if not already installed)
conda install ipykernel -y

# Add the kernel so it shows up in Jupyter
python -m ipykernel install --user --name=$ENV_NAME --display-name "Python ($ENV_NAME)"