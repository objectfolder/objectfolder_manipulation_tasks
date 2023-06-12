import logging

import gym
import cv2
import numpy as np
import copy
import collections
import yaml
import os
from .simulator import Simulator

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

class BaseEnv(gym.Env):
    """
    Base Env class that handles loading scene and robot, following OpenAI Gym interface.
    Functions like reset and step are not implemented.
    """

    def __init__(
        self,
        config_file,
        action_timestep=1 / 10.0,
        physics_timestep=1 / 240.0,
        use_pb_gui=False,
    ):
        """
        :param config_file: config_file path
        :param action_timestep: environment executes action per action_timestep second
        :param physics_timestep: physics timestep for pybullet
        :param use_pb_gui: concurrently display the interactive pybullet gui (for debugging)
        """
        self.config = parse_config(config_file)

        self.action_timestep = action_timestep
        self.physics_timestep = physics_timestep

        self.simulator = Simulator(
            physics_timestep=physics_timestep,
            render_timestep=action_timestep,
            image_width=self.config.get("image_width", 128),
            image_height=self.config.get("image_height", 128),
            use_pb_gui=use_pb_gui,
        )
        self.load()

    def reload(self, config_file):
        """
        Reload another config file.
        This allows one to change the configuration on the fly.

        :param config_file: new config file path
        """
        self.config = parse_config(config_file)
        self.simulator.reload()
        self.load()

    def reload_model(self, scene_id):
        """
        Reload another scene model.
        This allows one to change the scene on the fly.

        :param scene_id: new scene_id
        """
        self.simulator.reload()
        self.load()

    def load(self):
        """
        Load the scene and robot specified in the config file.
        """
        # TODO: import scene, robot gelsight here
        self.simulator.import_scene()

        # Get robot config
        robot_config = self.config["robot"]

        self.simulator.import_robot(robot_config['robotURDF_path'], robot_config['basePosition'] , robot_config['useFixedBase'])
        self.robot = self.simulator.robot

    def clean(self):
        """
        Clean up the environment.
        """
        if self.simulator is not None:
            self.simulator.disconnect()
        # TODO: close gelsight here or in the simulator??
        
    def close(self):
        """
        Synonymous function with clean.
        """
        self.clean()

    def simulator_step(self):
        """
        Step the simulation.
        This is different from environment step that returns the next
        observation, reward, done, info.
        """
        self.simulator.step()

    def step(self, action):
        """
        Overwritten by subclasses.
        """
        return NotImplementedError()

    def reset(self):
        """
        Overwritten by subclasses.
        """
        return NotImplementedError()
    
