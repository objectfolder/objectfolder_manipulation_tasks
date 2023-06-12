#!/bin/bash
##########################################
# Setting up virtualenv / conda / docker #
##########################################
# example here if using virtualenv
conda activate vir_env
echo "Virtual Env Activated"

cd dynamic_pushing
###################
# Run your script #
###################

# python tactile_refine.py
python train.py
