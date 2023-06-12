#!/bin/bash

##########################################
# Setting up virtualenv / conda / docker #
##########################################
# example here if using virtualenv
conda activate vir_env
echo "Virtual Env Activated"

cd grasp_stability task
###################
# Run your script #
###################
python train.py -N 10 \
                -obj_name 95 \
                -log_dir logs/95 \
                -model_dir models/95