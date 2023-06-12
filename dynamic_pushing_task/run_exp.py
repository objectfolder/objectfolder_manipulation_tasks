import os
import csv

import numpy as np
import pybullet as pb
import datetime
import taxim_robot
import experiments.utils
from experiments.panda_one_finger import Panda
from experiments.envs.PushEnv import PushEnv
import numpy as np
from tqdm import tqdm
import collections
import yaml
from dataset import DynamicPush
import collections
import yaml
import torch
from model import ForwardModel
import os
from datetime import datetime
import hydra


class Sampler:
    def __init__(self):
        pass

class GaussianSampler(Sampler):
    def __init__(self, horizon, a_dim):
        self.horizon = horizon
        self.a_dim = a_dim

    def sample_actions(self, num_samples, mu, std):
            m = np.expand_dims(mu, 0)
            s = np.random.normal(size=(num_samples, self.a_dim))
            s[:, 0] *= std[0]
            s[:, 1] *= std[1]
            return m + s

def parse_config(config):

    """
    Parse iGibson config file / object
    """
    try:
        collectionsAbc = collections.abc
    except AttributeError:
        collectionsAbc = collections

    if isinstance(config, collectionsAbc.Mapping):
        return config
    else:
        assert isinstance(config, str)

    if not os.path.exists(config):
        raise IOError(
            "config path {} does not exist. Please either pass in a dict or a string that represents the file path to the config yaml.".format(
                config
            )
        )
    with open(config, "r") as f:
        config_data = yaml.load(f, Loader=yaml.FullLoader)
    return config_data

def random_push():
    theta = np.random.uniform(-np.pi/6, np.pi/6)
    d = np.random.uniform(0.2, 0.25)
    return theta, d

@hydra.main(config_path="config", config_name="launch_experiments")
def main(cfg):
    print(cfg)
    config = cfg
    testset = DynamicPush(config.test_path, train=False, modality=config.modality)
    testLoader = torch.utils.data.DataLoader(
        testset, batch_size=1, shuffle=True, num_workers=12, pin_memory=True
    )
    model = ForwardModel(config.in_ch)

    model.load_state_dict(torch.load(config.model_path))
    model.eval()
    model.cuda()
    with torch.no_grad():
        for i, data in enumerate(testLoader):
            for d in data:
                d.to(torch.device('cuda'))
            x_1, x_2, x_3, act_in, pose_4_init, pose_4_final = data
            x_1 = x_1.cuda()
            x_2 = x_2.cuda()
            x_3 = x_3.cuda()
            act_in = act_in.cuda()
            pose_4_init = pose_4_init.cuda()
            pose_4_final = pose_4_final.cuda()
            fea_1 = model.encoder(x_1)
            fea_2 = model.encoder(x_2)
            fea_3 = model.encoder(x_3)
            feat = (fea_1 + fea_2 + fea_3) / 3
            break
    feat = feat.cuda()
    dynamic_model = model.dyn
    
    mu = np.array([0., 0.225])
    std = np.array([0.2, 0.0125])
    sampler = GaussianSampler(horizon=1, a_dim=2)
    
    goal = []
    for _ in range(config.test_ep):
        goal.append(np.array([np.random.uniform(1.5, 2.4), np.random.uniform(-0.4, 0.4)]))
    action = []
    init_pos = pose_4_init.repeat(config.num_sample, 1).type(torch.FloatTensor).cuda()
    feat = feat.repeat(config.num_sample, 1)
    pred_goal = []
    for g in tqdm(goal):
        for i in range(150):
            new_action_samples = sampler.sample_actions(config.num_sample, mu, std)
            new_action_samples[:, 0] = np.clip(new_action_samples[:, 0], -np.pi/6 + 1e-6, np.pi/6 - 1e-6)
            new_action_samples[:, 1] = np.clip(new_action_samples[:, 1], 0.2 + 1e-6, 0.25-1e-6)
            new_action_samples = torch.from_numpy(new_action_samples).type(torch.FloatTensor).cuda()
            with torch.no_grad():
                final_pos = dynamic_model(feat, init_pos, new_action_samples)
                cost = np.linalg.norm(final_pos.cpu().numpy() - g, axis=-1)
                best_idxs = np.argsort(cost)[:5]
                best_actions = [new_action_samples[i] for i in best_idxs]
                next_act = np.array([0.,0.])
                for a in best_actions:
                    next_act += a.cpu().numpy()
                next_act /= 5
                mu = next_act
        action.append(mu)
        with torch.no_grad():
            pred_goal.append(dynamic_model(feat[0].unsqueeze(0), init_pos[0], torch.tensor([mu[0], mu[1]]).unsqueeze(0).type(torch.FloatTensor).cuda()).cpu().numpy())
    test(config, action, goal, pred_goal)
    
def test(config, actions, goals, pred_goal):
    env = PushEnv(config, action_timestep=1/480, physics_timestep=1/480)
    tot_error = 0
    succ = [0, 0, 0, 0]
    date = datetime.now().strftime("%m_%d_%H_%M_%S")
    for i, a in tqdm(enumerate(actions)):
        print("executing action", a)
        print("aiming for goal", goals[i])
        theta, d = a
        robot_start, robot_end = env.prepare_push(theta, d)
        tmp_state = env.get_state()
        obj_pose_data = np.expand_dims(tmp_state['obj_pose'], axis=0)
        speed = int(4.5 * int(d * 35))
        action_lsit = np.linspace(robot_start, robot_end, speed)
        action_iter = iter(action_lsit)
        for e in tqdm(range(800)): #
            try:
                pos = next(action_iter) 
            except:
                pos = env.robot.pos
            state, done, info = env.step(pos)
            obj_pose_data = state['obj_pose']
        error = np.linalg.norm(obj_pose_data[:2] - goals[i])
        tot_error += error
        if error < 0.1:
            succ[0] += 1
        if error < 0.2:
            succ[1] += 1
        if error < 0.3:
            succ[2] += 1
        if error < 0.5:
            succ[3] += 1
        obj_path = os.path.join("results/", env.config["obj_name"], env.config["modality"], date + "_rews.txt")
        os.makedirs(os.path.join("results", env.config["obj_name"], env.config["modality"]), exist_ok=True)   
        with open(obj_path, "a") as f:
            writer = csv.writer(f)
            writer.writerow(["error", error, a, goals[i], pred_goal[i], obj_pose_data[:2]])   
        env.reset(True)
    
    tot_error /= len(actions)
    succ = np.array(succ) / len(actions)
    with open(obj_path, "a") as f:
        writer = csv.writer(f)
        writer.writerow(["SR", succ, tot_error])
        
if __name__ == "__main__":
    main()