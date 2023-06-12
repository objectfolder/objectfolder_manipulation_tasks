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
# 202 for vt, 112 for v, 169 for t

python run_exp.py model_path=##.pth \
                  test_path=test_trail_path \
                  obj_name="str(77)" \
                  modality=v_t \
                  in_ch=202 \