#!/bin/bash

##########################################
# Setting up virtualenv / conda / docker #
##########################################
# example here if using virtualenv
conda activate vir_env
echo "Virtual Env Activated"

cd /svg-prime/svg_prime
###################
# Run your script #
###################

python train_svg_prime.py \
    --obj_name 41\
    --name v \
    --a_dim 2 \
    --niter 1000 \
    --image_width 72 \
    --croped_size 64 \
    --modality v \
    --dataset surface_trav \
    --data_root train_data_path \
    --test_root test_data_path