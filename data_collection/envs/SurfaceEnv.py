import logging

import gym
import cv2
import numpy as np
import copy
from .BaseEnv import BaseEnv
from .TactileEnv import TactileEnv
import os
from .simulator import Simulator
from collections import OrderedDict
import data_collection.utils as utils
import pybullet as p
import yaml
import collections
from data_collection.tasks.BaseTask import BaseTask
from data_collection.tasks.DataTask import DataTask

log = logging.getLogger(__name__)

def parse_config(config):

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

class SurfaceEnv(TactileEnv):
    """
    Base Env class that handles loading scene and robot, following OpenAI Gym interface.
    Functions like reset and step are not implemented.
    """

    def __init__(
        self,
        config_file,
        action_timestep=1 / 10.0,
        physics_timestep=1 / 240.0,
    ):
        """
        :param config_file: config_file path
        :param action_timestep: environment executes action per action_timestep second
        :param physics_timestep: physics timestep for pybullet
        :param use_pb_gui: concurrently display the interactive pybullet gui (for debugging)
        """
        if type(config_file) == str:
            self.config = parse_config(config_file)
        else:
            self.config = config_file
        self.action_dimension = self.config["action_dim"]
        self.action_timestep = action_timestep
        self.physics_timestep = physics_timestep
        self.contact_location = self.config["contact_location"]
        self.use_pb_gui = self.config["use_pb_gui"]
        self.simulator = Simulator(
            env_config = self.config,
            physics_timestep=physics_timestep,
            render_timestep=action_timestep,
            image_width=self.config.get("image_width", 480),
            image_height=self.config.get("image_height", 640),
            use_pb_gui=self.use_pb_gui,
            gelsight_config=self.config["gelsight_config"]
        )
        self.contact_height = 0
        self.contact_y = 0
        self.load()
        self.reset()
        self.num_episode = 0
    
    def load_task_setup(self):
        if "task" not in self.config:
            self.task = BaseTask(self)
        elif self.config["task"] == "DataTask":
            self.task = DataTask(self)
        
    def build_obs_space(self, shape, low, high):
        """
        Helper function that builds individual observation spaces.

        :param shape: shape of the space
        :param low: lower bounds of the space
        :param high: higher bounds of the space
        """
        return gym.spaces.Box(low=low, high=high, shape=shape, dtype=np.float32)
    
    def load_observation_space(self):
        """
        Load observation space.
        """
        self.image_width = self.config.get("image_width", 480)
        self.image_height = self.config.get("image_height", 640)
        self.tactile_width = self.config.get("tactile_width", 480)
        self.tactile_height = self.config.get("tactile_height", 640)
        
        observation_space = OrderedDict()
        self.cam = utils.Camera(p, [self.image_width, self.image_height])
        self.gelsight = self.simulator.gelsight
        
        observation_space["visionRGB"] = self.build_obs_space(
            shape=(self.image_height, self.image_width, 3), low=0.0, high=1.0
        )
            
        observation_space["visionDepth"] = self.build_obs_space(
            shape=(self.image_height, self.image_width, 1), low=0.0, high=1.0
        )
        
        observation_space["tactile"] = self.build_obs_space(
            shape=(self.tactile_height, self.tactile_width, 3), low=0.0, high=1.0
        )
        
        observation_space["robot_pose"] = self.build_obs_space(
            shape=(7,), low=0.0, high=1.0
        )
            
        self.observation_space = gym.spaces.Dict(observation_space)
        
    def load_action_space(self):
        """
        Load action space.
        """
        # Return this action space
        d_ori = self.config["d_ori"]
        self.action_space = gym.spaces.Box(-d_ori, d_ori)
    
    def load_miscellaneous_variables(self):
        """
        Load miscellaneous variables for book keeping.
        """
        self.current_step = 0
        self.current_episode = 0

    def reload(self, config_file):
        """
        Reload another config file.
        This allows one to change the configuration on the fly.

        :param config_file: new config file path
        """
        self.config = parse_config(config_file)
        self.simulator.reload()
        self.load()

    def reload_model(self):
        """
        Reload another scene model.
        This allows one to change the scene on the fly.

        :param scene_id: new scene_id
        """
        self.simulator.reload()
        self.load()
    
    def find_best_contact(self):
        self.robot.init_robot()
        objPosLast, objOriLast, _ = utils.get_object_pose(p, self.simulator.objID)
        self.robot.go(_pos=[objPosLast[0]-0.1, objPosLast[1] + self.contact_y, self.robot.pos[2]])
        for _ in range(100):
            p.stepSimulation()
        self.robot.go(_pos=[objPosLast[0]-0.1, objPosLast[1] + self.contact_y, objPosLast[2] + self.contact_height])
        for _ in range(100):
            p.stepSimulation()
        initial_position = objPosLast[0]-0.1
        delta_x = 0.0003
        while True:
            initial_position += delta_x
            self.robot.go(_pos=[initial_position, objPosLast[1] + self.contact_y, objPosLast[2] + self.contact_height])
            p.stepSimulation()
            _, depth = self.gelsight.render()
            if depth[0].min() > 0:
                self.contact_location = objPosLast[0] - initial_position - 0.002
                break
    
    def reset_robot(self):
        
        self.robot.init_robot()
        objPosLast, objOriLast, _ = utils.get_object_pose(p, self.simulator.objID)
        self.robot.go(_pos=[objPosLast[0]-0.1, objPosLast[1] + self.contact_y, self.robot.pos[2]])
        for _ in range(100):
            p.stepSimulation()
        
        self.robot.go(_pos=[objPosLast[0]-0.1, objPosLast[1] + self.contact_y, objPosLast[2] + self.contact_height])
        for _ in range(100):
            p.stepSimulation()
        initial_position = objPosLast[0]-self.contact_location
        self.robot.go(_pos=[initial_position, objPosLast[1] + self.contact_y, objPosLast[2] + self.contact_height])
        for _ in range(100):
            p.stepSimulation()
                    
        self.pos_y_limit = np.array([objPosLast[1]+self.config.get('pos_y_limit')[0], objPosLast[1] + self.config.get('pos_y_limit')[1]])
        self.pos_z_limit = np.array([objPosLast[2]+self.config.get('pos_z_limit')[0], objPosLast[2] + self.config.get('pos_z_limit')[1]])

    def reset_object(self):
        ori = self.config.get("initial_orn")
        d_ori_x = 0 #np.random.uniform(-0.1, 0.1)
        d_ori_y = 0 #np.random.uniform(-0.01, 0.01)
        d_ori_z = 0 #np.random.uniform(-0.01, 0.01)
        ori = [ori[0] + d_ori_x, ori[1] + d_ori_y, ori[2] + d_ori_z]
        self.simulator.objStartOrientation = p.getQuaternionFromEuler(ori)
        self.contact_height = np.random.uniform(self.config["pos_z_limit"][0], self.config["pos_z_limit"][1])
        self.contact_y = np.random.uniform(self.config["pos_y_limit"][0], self.config["pos_y_limit"][1])
        p.resetBasePositionAndOrientation(self.simulator.objID, self.simulator.objStartPos, self.simulator.objStartOrientation)
    
    def reset_robot_to(self, state):
        
        self.robot.init_robot()
        objPosLast, objOriLast, _ = utils.get_object_pose(p, self.simulator.objID)
        self.robot.go(_pos=[objPosLast[0]-0.1, state[0], self.robot.pos[2]])
        for _ in range(100):
            p.stepSimulation()
        
        self.robot.go(_pos=[objPosLast[0]-0.1, state[0], state[1]])
        for _ in range(100):
            p.stepSimulation()
        
        initial_position = objPosLast[0]-self.contact_location
        self.robot.go(_pos=[initial_position, state[0], state[1]])
        for _ in range(100):
            p.stepSimulation()
        
        self.pos_y_limit = np.array([objPosLast[1]+self.config.get('pos_y_limit')[0], objPosLast[1] + self.config.get('pos_y_limit')[1]])
        self.pos_z_limit = np.array([objPosLast[2]+self.config.get('pos_z_limit')[0], objPosLast[2] + self.config.get('pos_z_limit')[1]])
        self.ori_x_limit = np.array([self.robot.ori[0] - self.config.get('ori_x_limit'), self.robot.ori[0] + self.config.get('ori_x_limit')])

        # This reset to function is epecific to refinement task
        pos = [self.robot.pos[0], state[0], state[1]]
        ori = self.robot.ori
        if not self.check_limits(pos, ori):
            raise Exception("Initial state not available!!!")
    
    def load(self):
        """
        Load the scene and robot specified in the config file.
        """
        self.simulator.import_scene()

        # Get robot config
        robot_config = self.config["robot"]

        self.simulator.import_robot(robot_config['robotURDF_path'], robot_config['basePosition'] , robot_config['useFixedBase'])
        self.robot = self.simulator.robot
        print("About to reset")
        self.load_task_setup()
        self.load_observation_space()
        self.load_action_space()
        self.load_miscellaneous_variables()
        self.reset_robot()
    
    def get_state(self):
        """
        Get the current observation.

        :return: observation as a dictionary
        """
        state = OrderedDict()
        visionColor, visionDepth = self.cam.get_image()
        tactileColor, depth = self.gelsight.render()
        if self.use_pb_gui:
            self.gelsight.updateGUI(tactileColor, depth)
        pos, ori = self.robot.get_ee_pose()
        robot_pose = np.array([pos[0], pos[1], pos[2], ori[0], ori[1], ori[2], ori[3]])
        state["visionRGB"] = np.array(visionColor[:,:,:3])
        state["visionDepth"] = np.array(visionDepth)
        state["tactile"] = np.array(tactileColor)[0]
        state["robot_pose"] = np.array(robot_pose)

        return state

    def run_simulation(self):
        """
        Run simulation for one action timestep (same as one render timestep in Simulator class).

        :return: a list of collisions from the last physics timestep
        """
        self.simulator_step()
    
    def populate_info(self, info):
        """
        Populate info dictionary with any useful information.

        :param info: the info dictionary to populate
        """
        info["episode_length"] = self.current_step

    def clean(self):
        """
        Clean up the environment.
        """
        if self.simulator is not None:
            self.simulator.disconnect()
    
    def check_limits(self, pos, ori):
        y_good = False
        z_good = False
        if pos[1] >= self.pos_y_limit[0] and pos[1] <= self.pos_y_limit[1]:
            y_good = True
        if pos[2] >= self.pos_z_limit[0] and pos[2] <= self.pos_z_limit[1]:
            z_good = True
        
        return y_good and z_good

    def step(self, action=np.array([0.,0.])):
        """
        Apply robot's action and return the next state, reward, done and info,
        following OpenAI Gym's convention

        :param action: robot actions
        :return: state: next observation
        :return: reward: reward of this time step
        :return: done: whether the episode is terminated
        :return: info: info dictionary with any useful information
        """
        # apply action to the robot and step forward to update states
        dpos_y= action[0] #actions["dpos"]
        dpos_z= action[1] #actions["dpos"]
        dori= 0 #s['dori']
        self.current_step += 1
        pos = self.robot.pos + np.array([0, dpos_y, dpos_z])
        ori = np.array([self.robot.ori[0] + dori, self.robot.ori[1],self.robot.ori[2]])  #np.array(dori)
        if self.check_limits(pos, ori):
            self.robot.go(_pos=pos, _ori=ori)
        self.run_simulation()

        state = self.get_state()
        info = {}
        done, info = self.task.get_termination(self, info)
        self.task.step(self)
        self.populate_info(info)
        
        if done:
            info["last_observation"] = state
            self.num_episode += 1            
            self.reset()
            
        return state, done, info

    def reset(self):
        """
        Reset episode.
        """
        print("Reset Evironment!")
        # reset obj to a random pose
        self.reset_object()
        # let robot find a contact position
        self.find_best_contact()
        self.task.reset(self)
        state = self.get_state()
        self.reset_variables()
        return state
    
    def reset_to(self, state):
        # self.find_best_contact()
        self.task.reset_to(self, state)
        state = self.get_state()
        self.reset_variables()
        return state
    
    def reset_variables(self):
        """
        Reset bookkeeping variables for the next new episode.
        """
        self.current_episode += 1
        self.current_step = 0