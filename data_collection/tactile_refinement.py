import argparse
import datetime
import logging
import os
import time

import numpy as np
import pybullet as pb
import pybullet_data

import taxim_robot
import utils
from setup import getObjInfo
from taxim_robot.sensor import get_r15_one_config_path
from collections import defaultdict
from OBJ_Robot.Robotic_Tasks.data_collection.envs.TactileEnv import TactileEnv
import numpy as np
import h5py
from datetime import datetime
import cv2
import time
import warnings

parser = argparse.ArgumentParser()
parser.add_argument('-config_file', default='config/tactile_refinement.yaml', help='Config Path.')
args = parser.parse_args()

# set ee gap
utils.ee_gap = 0 #0.20045875

def create_dset(hf, observation, name, seq_length, episode_len=3000):
    print(f"creating {name}")
    if name == 'action':
        d0 = np.array([0.])
    else:
        d0 = observation[name]
    shape = (episode_len, seq_length,) + d0.shape
    chunk = (1, seq_length,) + d0.shape
    print(f"d0 shape: {d0.shape}")
    print(f"d0 type: {d0.dtype}")
    new_tuple = (episode_len, seq_length,) + d0.shape
    print(f"new shape: {new_tuple}")
    hf.create_dataset(name, shape=shape, dtype=d0.dtype, chunks=chunk, maxshape=new_tuple, compression="gzip")

def random_policy(d_pos=0, d_ori=0):
    random_action = np.random.randint(0, 3)
    mapping = {#0: {'dpos': [0.001, 0, 0], 'dori': [0, 0, 0]}, 
               #1: {'dpos': [-0.001, 0, 0], 'dori': [0, 0, 0]}, 
               0: {'dpos': [0, 0, 0], 'dori': [0, 0, 0]}, 
               1: {'dpos': [0, 0, 0], 'dori': [d_ori, 0, 0]}, 
               2: {'dpos': [0, 0, 0], 'dori': [-d_ori, 0, 0]}}
    dpos = mapping[random_action]['dpos']
    dori = mapping[random_action]['dori']
    return dpos, dori, np.array([random_action])

def gaussian_policy(mu=0, sigma=0.05):
    random_action = np.random.normal(mu, sigma)
    if random_action > 0.05:
        random_action = 0.05
    elif random_action < -0.05:
        random_action = -0.05
    return random_action
    
def main(args):
    env = TactileEnv(args.config_file, action_timestep=1.0)
    state = env.get_state()
    t = datetime.now().strftime("%m_%d_%H_%M_%S") + '_' + str(env.config['num_episodes']) + "_" + env.config['obj_name']
    if not os.path.isdir(os.path.join(env.config["dir_path"], t)):
        os.mkdir(os.path.join(env.config["dir_path"], t))
    data_path = os.path.join(env.config["dir_path"], t, 'data.hdf5')
    hf = h5py.File(data_path, 'w')
    for name in ['tactile', 'visionRGB', 'visionDepth', "robot_pose"]:
        create_dset(hf, state, name, env.config["seq_len"] + 1, env.config['num_episodes'])
    create_dset(hf, state, 'action', env.config["seq_len"], env.config['num_episodes'])
    for k in range(env.config['num_episodes']):
        tmp_state = env.get_state()
        tactile_data = np.expand_dims(tmp_state['tactile'], axis=0)
        visionRGB_data = np.expand_dims(tmp_state['visionRGB'], axis=0)
        visionDepth_data = np.expand_dims(tmp_state['visionDepth'], axis=0)
        pose_data = np.expand_dims(tmp_state['robot_pose'], axis=0)
        action_data = None
        for i in range(20):
            if i % 4 == 0:
                action = gaussian_policy(0, 0.05)
            state, done, info = env.step(action)
            tactile_data = np.vstack((tactile_data, np.expand_dims(state['tactile'], axis=0)))
            visionRGB_data = np.vstack((visionRGB_data, np.expand_dims(state['visionRGB'], axis=0)))
            visionDepth_data = np.vstack((visionDepth_data, np.expand_dims(state['visionDepth'], axis=0)))
            pose_data = np.vstack((pose_data, np.expand_dims(state['robot_pose'], axis=0)))
            action_data = np.array([[action]]) if action_data is None else np.vstack((action_data, np.expand_dims(action, axis=0)))
            
        dataset = {'tactile': tactile_data, 
                   'visionRGB': visionRGB_data, 
                   'visionDepth': visionDepth_data, 
                   "robot_pose": pose_data, 
                   'action': action_data}
        s = time.time()
        for m, v in dataset.items():
            if m == 'tactile' or m == 'visionRGB':
                hf[m][k:k+1, :, :, :, :] = np.expand_dims(v, axis=0)
            elif m == 'visionDepth':
                hf[m][k:k+1, :, :, :] = np.expand_dims(v, axis=0)
            else:
                hf[m][k:k+1, :, :] = np.expand_dims(v, axis=0)
        print(time.time() - s)
        env.reset()
    hf.close()
        
        
if __name__ == "__main__":
    main(args)