#!/bin/bash
#
#SBATCH --job-name=sur_vt
#SBATCH --account=viscam
#SBATCH --partition=viscam --qos=normal
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time 192:00:00
#SBATCH --output=/viscam/projects/objectfolder_benchmark/Robot_related/Taxim_obj/server_logs/sonic_slurm_%A.out
#SBATCH --error=/viscam/projects/objectfolder_benchmark/Robot_related/Taxim_obj/server_logs/sonic_slurm_%A.err
#SBATCH --mail-user=li2053@stanford.edu
#SBATCH --mail-type=ALL
######################
# Begin work section #
######################
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR

##########################################
# Setting up virtualenv / conda / docker #
##########################################
# example here if using virtualenv
source /sailhome/li2053/.bashrc
conda activate data_collect
echo "Virtual Env Activated"

##############################################################
# Setting up LD_LIBRARY_PATH or other env variable if needed #
##############################################################
# export LD_LIBRARY_PATH=/usr/local/cuda-11.3/lib64:/usr/lib/x86_64-linux-gnu 
echo "Working with the LD_LIBRARY_PATH: "$LD_LIBRARY_PATH

cd /viscam/projects/objectfolder_benchmark/Robot_related/Taxim_obj/perceptual-metrics/perceptual_metrics/scripts
###################
# Run your script #
###################
#modality
#image_shape
#ckpoint
#obj_name
#initial_ori
#height
#modality
#img_shape

# python tactile_refine.py
python run_control.py \
    env=taxim_refine \
    env.a_dim=1 \
    env.config.action_dim=1 \
    task=refine \
    agent.optimizer.init_std=[0.05] \
    model.checkpoint_dir=["/viscam/projects/objectfolder_benchmark/Robot_related/Taxim_obj/svg-prime/svg_prime/logs/tac_refine/53/v/model_998.pth","/viscam/projects/objectfolder_benchmark/Robot_related/Taxim_obj/svg-prime/svg_prime/logs/tac_refine/53/t/model_998.pth"]\
    modality=v_t \
    image_shape=[64,128] \
    env.config.modality=v_t \
    env.config.image_shape=[64,128] \
    env.config.obj_name="str(53)" \
    env.config.obj_height=0.2 \
    env.config.initial_orn=[1.6,0,1.6] \
    env.config.contact_height=[-0.005,0.01] \

#   obj_name: "42"
#   obj_height: 0.2
#   contact_height: [-0.01, 0.01]
#   initial_orn: [1.6, 0, 1.6]

  # obj_name: "11"
  # obj_height: 0.2
  # contact_height: [-0.01, 0.01]
  # initial_orn: [1.6, 0, 1.6]

  # obj_name: "35"
  # initial_orn: [0, -0.09, 0]
  # contact_height: [-0.01, 0.01]
  # obj_height: 0.15

  # obj_name: "39"
  # obj_height: 0.2
  # contact_height: [-0.01, 0.01]
  # initial_orn: [1.6, 0, 1.6]

  # obj_name: "53"
  # obj_height: 0.2
  # contact_height: [-0.02, 0.01]
  # initial_orn: [1.6, 0, 1.6]