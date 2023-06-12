import numpy as np
from PIL import Image
from transforms3d.quaternions import *
from model_predictive_control.mpc.utils import ObservationList
import torch
from torchvision import transforms

import gym.spaces as spaces
from model_predictive_control.envs.base import BaseEnv
from experiments.envs.RobotEnv import RobotEnv
from experiments.envs.SurfaceEnv import SurfaceEnv
import pybullet as pb
import cv2

class ObjTaximEnv(BaseEnv):
    def __init__(self, config, a_dim):
        self.env = RobotEnv(
            config_file=config,
            action_timestep = 1.0
        )
        self.modality = self.env.config['modality'].split('_')
        self.croped_size = self.env.config["croped_size"]
        self.image_size = self.env.config["resize_size"]
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.trainTransform = transforms.Compose(
            [
                transforms.Resize(self.image_size),
                transforms.CenterCrop(self.croped_size),
            ]
        )
        self.a_dim = a_dim
        
    def reset(self):
        # it will reset both robot and object state
        state = self.env.reset()
        new_state = {}
        im_v = torch.from_numpy(state["visionRGB"]).permute(2,0,1)
        im_t = torch.from_numpy(state["tactile"]).permute(2,0,1)
        im_v = self.trainTransform(im_v).permute(1,2,0).numpy()
        im_t = self.trainTransform(im_t).permute(1,2,0).numpy()
        if 'v' in self.modality and 't' in self.modality:
            new_state["rgb"] = np.concatenate((im_v, im_t), axis=-2)
        elif 'v' in self.modality:
            new_state["rgb"] = im_v
        elif 't' in self.modality:
            new_state["rgb"] = im_t
        new_state["robot_pose"] = state["robot_pose"]
        return new_state

    def reset_to(self, s):
        # reset to will not reset the object state, it will only reset the robot state
        state = self.env.reset_to(s)
        new_state = {}
        im_v = torch.from_numpy(state["visionRGB"]).permute(2,0,1)
        im_t = torch.from_numpy(state["tactile"]).permute(2,0,1)
        im_v = self.trainTransform(im_v).permute(1,2,0).numpy()
        im_t = self.trainTransform(im_t).permute(1,2,0).numpy()
        if 'v' in self.modality and 't' in self.modality:
            new_state["rgb"] = np.concatenate((im_v, im_t), axis=-2)
        elif 'v' in self.modality:
            new_state["rgb"] = im_v
        elif 't' in self.modality:
            new_state["rgb"] = im_t
        new_state["robot_pose"] = state["robot_pose"]
        return new_state

    def reset_state(self, s):
        state = self.env.reset_to(s)
        new_state = {}
        im_v = torch.from_numpy(state["visionRGB"]).permute(2,0,1)
        im_t = torch.from_numpy(state["tactile"]).permute(2,0,1)
        im_v = self.trainTransform(im_v).permute(1,2,0).numpy()
        im_t = self.trainTransform(im_t).permute(1,2,0).numpy()
        if 'v' in self.modality and 't' in self.modality:
            new_state["rgb"] = np.concatenate((im_v, im_t), axis=-2)
        elif 'v' in self.modality:
            new_state["rgb"] = im_v
        elif 't' in self.modality:
            new_state["rgb"] = im_t        
        new_state["robot_pose"] = state["robot_pose"]
        return new_state


    def step(self, action):
        state, done, info = self.env.step(action)
        new_state = {}
        im_v = torch.from_numpy(state["visionRGB"]).permute(2,0,1)
        im_t = torch.from_numpy(state["tactile"]).permute(2,0,1)
        im_v = self.trainTransform(im_v).permute(1,2,0).numpy()
        im_t = self.trainTransform(im_t).permute(1,2,0).numpy()
        if 'v' in self.modality and 't' in self.modality:
            new_state["rgb"] = np.concatenate((im_v, im_t), axis=-2)
        elif 'v' in self.modality:
            new_state["rgb"] = im_v
        elif 't' in self.modality:
            new_state["rgb"] = im_t
        new_state["robot_pose"] = state["robot_pose"]
        return new_state, 0, done, info

    def get_state(self):
        quat = self.env.get_state()["robot_pose"][3:]
        angle = pb.getEulerFromQuaternion(quat)[0]
        if angle < 0:
            angle += 2 * np.pi
        return angle

    def compute_score(self, state, goal_state):
        differences = np.abs(state - goal_state)
        return differences

    @property
    def action_dimension(self):
        return self.env.action_dimension

    @property
    def observation_shape(self):
        return self.env.config["image_shape"]

    def goal_generator(self):
        """
        Create a goal generator from a robosuite-formatted hdf5 file.
        :param file_path: path to hdf5 goal file
        :param camera_name: name of the camera feed to load goal RGB, depth, etc. from.
        :return: generator object which yields a tuple of (start state, goal state, goal obs) for a different task
        at each iteration
        """
        
        for _ in range(10):
            initial_state = np.random.uniform(-1, 1) * self.env.config["ori_x_limit"] + np.pi
            goal_state = np.random.uniform(-1, 1) * self.env.config["ori_x_limit"] + np.pi
            ang = abs(goal_state - initial_state)
            if ang > 0.174444444:
                break
        goals = dict()
        state = self.reset_to(goal_state)
        # must divided by 255
        goals['rgb'] = np.expand_dims(state["rgb"], axis=0) / 255.0
        goals = ObservationList(goals, image_shape=self.env.config["image_shape"])

        return initial_state, goal_state, goals


class RobotSurfaceEnv(BaseEnv):
    def __init__(self, config, a_dim):
        self.env = SurfaceEnv(
            config_file=config,
            action_timestep = 1.0
        )
        self.modality = self.env.config['modality'].split('_')
        self.croped_size = self.env.config["croped_size"]
        self.image_size = self.env.config["resize_size"]
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.trainTransform = transforms.Compose(
            [
                transforms.Resize(self.image_size),
                transforms.CenterCrop(self.croped_size),
            ]
        )
        self.a_dim = a_dim
        self.last_pos = None
        
    def reset(self):
        # it will reset both robot and object state
        state = self.env.reset()
        new_state = {}
        im_v = torch.from_numpy(state["visionRGB"]).permute(2,0,1)
        im_t = torch.from_numpy(state["tactile"]).permute(2,0,1)
        im_v = self.trainTransform(im_v).permute(1,2,0).numpy()
        im_t = self.trainTransform(im_t).permute(1,2,0).numpy()
        if 'v' in self.modality and 't' in self.modality:
            new_state["rgb"] = np.concatenate((im_v, im_t), axis=-2)
        elif 'v' in self.modality:
            new_state["rgb"] = im_v
        elif 't' in self.modality:
            new_state["rgb"] = im_t
        new_state["robot_pose"] = state["robot_pose"]
        return new_state

    def reset_to(self, s):
        # reset to will not reset the object state, it will only reset the robot state
        state = self.env.reset_to(s)
        new_state = {}
        im_v = torch.from_numpy(state["visionRGB"]).permute(2,0,1)
        im_t = torch.from_numpy(state["tactile"]).permute(2,0,1)
        im_v = self.trainTransform(im_v).permute(1,2,0).numpy()
        im_t = self.trainTransform(im_t).permute(1,2,0).numpy()
        if 'v' in self.modality and 't' in self.modality:
            new_state["rgb"] = np.concatenate((im_v, im_t), axis=-2)
        elif 'v' in self.modality:
            new_state["rgb"] = im_v
        elif 't' in self.modality:
            new_state["rgb"] = im_t
        new_state["robot_pose"] = state["robot_pose"]
        return new_state

    def reset_state(self, s):
        state = self.env.reset_to(s)
        new_state = {}
        im_v = torch.from_numpy(state["visionRGB"]).permute(2,0,1)
        im_t = torch.from_numpy(state["tactile"]).permute(2,0,1)
        im_v = self.trainTransform(im_v).permute(1,2,0).numpy()
        im_t = self.trainTransform(im_t).permute(1,2,0).numpy()
        if 'v' in self.modality and 't' in self.modality:
            new_state["rgb"] = np.concatenate((im_v, im_t), axis=-2)
        elif 'v' in self.modality:
            new_state["rgb"] = im_v
        elif 't' in self.modality:
            new_state["rgb"] = im_t        
        new_state["robot_pose"] = state["robot_pose"]
        return new_state


    def step(self, action):
        state, done, info = self.env.step(action)
        new_state = {}
        im_v = torch.from_numpy(state["visionRGB"]).permute(2,0,1)
        im_t = torch.from_numpy(state["tactile"]).permute(2,0,1)
        im_v = self.trainTransform(im_v).permute(1,2,0).numpy()
        im_t = self.trainTransform(im_t).permute(1,2,0).numpy()
        if 'v' in self.modality and 't' in self.modality:
            new_state["rgb"] = np.concatenate((im_v, im_t), axis=-2)
        elif 'v' in self.modality:
            new_state["rgb"] = im_v
        elif 't' in self.modality:
            new_state["rgb"] = im_t        
        new_state["robot_pose"] = state["robot_pose"]
        return new_state, 0, done, info

    def get_state(self):
        pos = np.array([self.env.robot.pos[1], self.env.robot.pos[2]])
        return pos

    def compute_score(self, state, goal_state):
        # l2 distance is the score
        if self.last_pos is not None:
            dis = np.linalg.norm(state - self.last_pos)
        else:
            dis = 0
        self.last_pos = state
        differences = np.array([np.linalg.norm(state - goal_state), dis])
        return differences

    @property
    def action_dimension(self):
        return self.env.action_dimension

    @property
    def observation_shape(self):
        return self.env.observation_space["tactile"].shape[:2]

    def goal_generator(self):
        """
        Create a goal generator from a robosuite-formatted hdf5 file.
        :param file_path: path to hdf5 goal file
        :param camera_name: name of the camera feed to load goal RGB, depth, etc. from.
        :return: generator object which yields a tuple of (start state, goal state, goal obs) for a different task
        at each iteration
        """
        # because after reset the robot initial position is already random, we won't need randomize again
        initial_state = np.array([self.env.robot.pos[1], self.env.robot.pos[2]])
        for _ in range(20):
            goal_state = np.array([np.random.uniform(self.env.pos_y_limit[0], self.env.pos_y_limit[1]), \
                                np.random.uniform(self.env.pos_z_limit[0], self.env.pos_z_limit[1])])
            dist = np.linalg.norm(goal_state - initial_state)
            if dist >= 0.005 and dist <= 0.02:
                break
        goals = dict()
        state = self.reset_to(goal_state)
        # must divided by 255
        goals['rgb'] = np.expand_dims(state["rgb"], axis=0) / 255.0
        goals = ObservationList(goals, image_shape=self.env.config["image_shape"])
        return initial_state, goal_state, goals


if __name__ == "__main__":
    # import robodesk

    # env = robodesk.RoboDesk()
    config = "/home/haoli/Taxim/experiments/config/data_collection.yaml"
    env = ObjTaximEnv(config, a_dim=1)
    obs = env.reset()
    # import ipdb
    # ipdb.set_trace()
    done = False
    while not done:
        action = env.action_space.sample()
        obs, _, done, info = env.step(action)
        env.get_state()
