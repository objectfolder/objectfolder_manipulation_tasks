#!/bin/bash
##########################################
# Setting up virtualenv / conda / docker #
##########################################
# example here if using virtualenv
conda activate vir_env
echo "Virtual Env Activated"
cd svg-prime/svg_prime

###################
# Run your script #
###################

python train_svg_prime.py \
    --obj_name 42\
    --name v \
    --a_dim 1 \
    --niter 1000 \
    --image_width 80 \
    --croped_size 64 \
    --modality v \
    --dataset tac_refine \
    --data_root train_data_path \
    --test_root test_data_path
