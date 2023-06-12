#!/bin/bash

##########################################
# Setting up virtualenv / conda / docker #
##########################################
# example here if using virtualenv
conda activate vir_env
echo "Virtual Env Activated"

cd data_collection
###################
# Run your script #
###################

# python tactile_refine.py
# python surface_traversal.py
# python dynamic_pushing.py
python grasp_panda_demo.py -obj 22 -dxy -data_path data/22
