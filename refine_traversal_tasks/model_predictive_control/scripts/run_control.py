# Run control using Visual Foresight with FitVid video prediction model and MPPI controller.
import numpy as np
import h5py
import datetime
import csv
import os
import copy
import hydra

from hydra.utils import instantiate
from omegaconf import OmegaConf, open_dict

from model_predictive_control.mpc.utils import *
# from perceptual_metrics.models.simulator_model import SimulatorModel
from model_predictive_control.mpc.agent import PlanningAgent
from tqdm import tqdm

def create_env(cfg):
    env = instantiate(cfg)
    ObservationList.IMAGE_SHAPE = env.observation_shape
    return env


def run_trajectory(cfg, folder_name, agent, env, initial_state, goal_state, goal_image):

    obs = env.reset_to(initial_state)

    # if isinstance(agent, PlanningAgent) and isinstance(
    #     agent.optimizer.model, SimulatorModel
    # ):
    #     agent.optimizer.model.reset_to(initial_state)

    # obs, _, _, _ = env.step(
    #     np.zeros(env.action_dimension)
    # )  # not taking this step delays iGibson observations, TODO debug this!!

    num_steps = 0
    observations = ObservationList.from_obs(obs, cfg)
    observations.save_image(f"{folder_name}/obs_after_reset", index=0)
    observations.append(ObservationList.from_obs(obs, cfg))
    state_observations = [env.get_state()]

    observations.save_image(f"{folder_name}/obs_after_step", index=-1)

    agent.reset()
    agent.set_log_dir(folder_name)

    rews = []

    while num_steps < cfg.max_traj_length:
        # TODO: goal_length should eventually just be cfg.planning_horizon, but the length of predictions from model
        # classes is currently cfg.planning_horizon + cfg.n_context - 1
        goal_length = cfg.n_context + cfg.planning_horizon - 1
        if num_steps < len(goal_image):
            goal_image = goal_image[num_steps : num_steps + goal_length]
        else:
            goal_image = goal_image[-1]
        if len(goal_image) < goal_length:
            goal_image = goal_image.append(
                goal_image[-1].repeat(goal_length - len(goal_image))
            )
        agent.set_goal(goal_image)

        action = agent.act(num_steps, observations, state_observations)
        obs, _, _, _ = env.step(action)

        observations.append(ObservationList.from_obs(obs, cfg))

        if (
            observations[-1][cfg.planning_modalities[0]].sum() == 0
            or observations[-1][cfg.planning_modalities[0]].min() >= 255
        ):
            # If rendering breaks (black screen), rerun the trajectory
            return None, None, False

        state_observations.append(env.get_state())
        # rews.append(env.get_reward())
        print(f"Step {num_steps}: Action = {action}")
        num_steps += 1

    for state_observation in state_observations:
        rews.append(env.compute_score(state_observation, goal_state))

    # Currently success is always returned True, even if the task is not solved, so each task is run once
    return observations, rews, True


def set_all_seeds(seed):
    import random

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True


def get_already_completed_runs(folder_name):
    rews_path = os.path.join(folder_name, "all_rews.txt")
    num_trajectories_completed = 0
    if os.path.exists(rews_path):
        with open(rews_path, "r") as f:
            rews = list(f.read().splitlines())
            num_trajectories_completed = len(rews)
    return num_trajectories_completed

def cal_success(task, succ_rate, final_error, error, sum_dist, spl):
    if task == 'refine':
        if final_error < 0.017453292:
            succ_rate[0] += 1
        if final_error < 0.008726646:
            succ_rate[1] += 1
        if final_error < 0.001745329:
            succ_rate[2] += 1
    elif task == 'surface':
        if final_error < 0.01:
            spl[0] += sum_dist
            succ_rate[0] += 1
        if final_error < 0.005:
            spl[1] += sum_dist
            succ_rate[1] += 1
        if final_error < 0.001:
            spl[2] += sum_dist
            succ_rate[2] += 1
    error += final_error
    return succ_rate, error, spl

@hydra.main(config_path="configs", config_name="config")
def run_control(cfg):
    set_all_seeds(cfg.seed)
    with open_dict(cfg):
        cfg.slurm_job_id = os.getenv("SLURM_JOB_ID", 0)

    with open("config.yaml", "w") as f:
        OmegaConf.save(config=cfg, f=f.name)

    # The model needs to be instantiated separately from the agent because it could have a complex __init__ function,
    # for example if it performs multiprocessing.
    model = instantiate(
        cfg.model, _recursive_=(not "SimulatorModel" in cfg.model._target_)
    )  # prevent recursive instantiation for simulator model
    agent = instantiate(cfg.agent, optimizer={"model": model})

    env = create_env(cfg.env)

    folder_name = os.getcwd()
    print(f"Log directory : {os.getcwd()}")

    all_traj_rews = list()

    # Resume previous run, if applicable
    if cfg.resume:
        num_trajectories_completed = get_already_completed_runs(folder_name)
    else:
        num_trajectories_completed = 0

    succ_rate = [0, 0, 0]
    spl = [0, 0 ,0]
    error = 0
    for t in tqdm(range(cfg.num_trajectories)):
        print(f"Running trajectory {t}...")
        init_state, goal_state, goal_image = env.goal_generator() #next(goal_itr)

        traj_folder = f"{folder_name}/traj_{t}/"
        if t < num_trajectories_completed:
            # The trajectory has already been completed in a previous run. Skip it, but make sure goal and initial state
            # are iterated.
            assert os.path.exists(
                traj_folder
            ), "Trajectory folder does not exist, but control is resuming from a further point"

            # Load the rewards from the previous run for bookkeeping
            all_traj_rews.append(np.load(f"{traj_folder}/traj_{t}_rews.npy"))
            # Skip this trajectory
            continue

        os.makedirs(traj_folder, exist_ok=True)
        if t < 10:
            goal_image.log_gif(f"{traj_folder}/goal_gif")
        success = False
        while not success:
            # Try to run control on a starting state and goal repeatedly
            # Not that here success does *not* mean that the task was solved,
            # but that the trajectory was run e.g. without any environment errors.
            obs_out, rews, success = run_trajectory(
                cfg, traj_folder, agent, env, init_state, goal_state, goal_image
            )
        if t < 10:
            obs_out.append(goal_image[-1].repeat(len(obs_out)), axis=1).log_gif(
                f"{traj_folder}/traj_{t}_vis"
            )
        if cfg.task == 'refine':            
            traj_rews = np.array(rews)
            all_traj_rews.append(traj_rews)
            np.save(f"{traj_folder}/traj_{t}_rews", traj_rews)
        else:
            er = []
            dis = []
            for r in rews:
                er.append(r[0])
                dis.append(r[1])
            traj_rews = np.array(er)
            traj_dis = np.array(dis)
            all_traj_rews.append(traj_rews)
            np.save(f"{traj_folder}/traj_{t}_rews", traj_rews)
        sum_dist = 0
        with open(f"{folder_name}/all_rews.txt", "a") as f:
            writer = csv.writer(f)
            if cfg.task == "refine":
                writer.writerow([t, traj_rews.min(), np.argmin(traj_rews)])
            elif cfg.task == "surface":
                idx = np.argmin(traj_rews)
                for i in range(idx+1):
                    sum_dist += traj_dis[i] 
                writer.writerow([t, traj_rews.min(), sum_dist])
        succ_rate, error, spl = cal_success(cfg.task, succ_rate, traj_rews.min(), error, sum_dist, spl)
        env.last_pos = None
        env.reset()
    error = error / cfg.num_trajectories
    spl = np.array(spl) / np.array(succ_rate)
    succ_rate = np.array(succ_rate) / cfg.num_trajectories
    with open(f"{folder_name}/all_rews.txt", "a") as f:
        writer = csv.writer(f)
        writer.writerow(["SR", succ_rate, error, spl])

def recal_sr(folder_name, task):
    with open(f"{folder_name}", "r") as f:
        data = f.read().splitlines()
    succ_rate = [0, 0, 0]
    num = 0
    error = 0
    spl = 0
    for d in data:
        num += 1
        if task == 'refine':
            final_error = float(d.split(',')[1])
            sum_dist = 0
        else:
            sum_dist = float(d.split(',')[2])
            final_error = 0
        succ_rate, error, spl = cal_success(task, succ_rate, final_error, error, sum_dist, spl)
    succ_rate = np.array(succ_rate) / num
    error /= num
    spl /= num
    print(succ_rate, error, spl)
        
if __name__ == "__main__":
    run_control()
    # recal_sr('/viscam/projects/objectfolder_benchmark/Robot_related/Taxim_obj/perceptual-metrics/perceptual_metrics/scripts/outputs/2022-11-08/00-36-16/all_rews.txt','refine')