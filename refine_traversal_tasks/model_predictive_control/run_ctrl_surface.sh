#!/bin/bash
#
#SBATCH --job-name=sur_vt
#SBATCH --account=viscam
#SBATCH --partition=svl --qos=normal
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

# python tactile_refine.py
python run_control.py \
    env=taxim_surface \
    env.a_dim=2 \
    env.config.action_dim=2 \
    task=surface \
    agent.optimizer.init_std=[0.005,0.005] \
    model.checkpoint_dir=["/viscam/projects/objectfolder_benchmark/Robot_related/Taxim_obj/svg-prime/svg_prime/logs/surface_trav/51/v/model_614.pth","/viscam/projects/objectfolder_benchmark/Robot_related/Taxim_obj/svg-prime/svg_prime/logs/surface_trav/51/t/model_614.pth"]\
    modality=v_t \
    image_shape=[64,128] \
    env.config.modality=v_t \
    env.config.image_shape=[64,128] \
    env.config.obj_name="str(51)" \
    env.config.obj_height=0.2 \
    env.config.initial_orn=[0.1,1.5607963,0.01] \
    env.config.pos_y_limit=[-0.04,0.04] \
    env.config.pos_z_limit=[-0.03,0.03] \


# obj_name: "51"
# obj_height: 0.2
# initial_orn: [0.1, 1.5607963, 0.01]
# pos_y_limit: [-0.04, 0.04]
# pos_z_limit: [-0.03, 0.03]
    

  # obj_name: "85"
  # obj_height: 0.2
  # initial_orn: [1.58, 0, 1.55]
  # pos_y_limit: [-0.04, 0.04]
  # pos_z_limit: [-0.012, 0.012]

# only t
  # obj_name: "41"
  # obj_height: 0.2
  # initial_orn: [1.6, 0, 1.6]
  # pos_y_limit: [-0.04, 0.04]
  # pos_z_limit: [-0.015, 0.015]

  # obj_name: "53"
  # obj_height: 0.2
  # initial_orn: [1.6, 0, 1.6]
  # pos_y_limit: [-0.04, 0.04]
  # pos_z_limit: [-0.015, 0.015]

  # obj_name: "56"
  # obj_height: 0.2
  # initial_orn: [1.6, 0, 1.6]
  # pos_y_limit: [-0.05, 0.03]
  # pos_z_limit: [-0.015, 0.015]

  # obj_name: "17"
  # obj_height: 0.125
  # initial_orn: [0, 1.6307963, 0]
  # pos_y_limit: [-0.035, 0.035]
  # pos_z_limit: [-0.032, -0.018]

