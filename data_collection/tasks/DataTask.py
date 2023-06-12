import numpy as np
import pybullet as p
from experiments.tasks.BaseTask import BaseTask
from experiments.termination_conditions.timeout import Timeout


class DataTask(BaseTask):
    """
    Point Nav Fixed Task
    The goal is to navigate to a fixed goal position
    """

    def __init__(self, env):
        super(DataTask, self).__init__(env)
        # self.reward_type = self.config.get("reward_type", "l2")
        self.termination_conditions = [
            Timeout(self.config),
        ]
        self.num_actions = 0
        self.obj = self.config.get("obj_name")
        self.initial_pos = np.array(self.config.get("initial_pos", [0, 0, 0]))
        self.initial_orn = np.array(self.config.get("initial_orn", [0, 0, 0]))

        self.load_obj(env)

    def load_obj(self, env):
        """
        Load visualization, such as initial and target position, shortest path, etc

        :param env: environment instance
        """
        env.simulator.import_object(self.obj, self.initial_pos, self.initial_orn)

    def get_l2_potential(self, env):
        """
        Get potential based on L2 distance

        :param env: environment instance
        :return: L2 distance to the target position
        """
        return l2_distance(env.robots[0].get_position()[:2], self.target_pos[:2])

    def reset_agent(self, env):
        """
        Task-specific agent reset: land the robot to initial pose, compute initial potential

        :param env: environment instance
        """
        # TODO: reset robot pos and orn
        env.reset_robot()

    def reset_agent_to(self, env, state):
        env.reset_robot_to(state)
    
    def reset_to(self, env, state):
        self.reset_agent_to(env, state)
        self.reset_variables(env)
        for reward_function in self.reward_functions:
            reward_function.reset(self, env)
        for termination_condition in self.termination_conditions:
            termination_condition.reset(self, env)
    
    def reset_variables(self, env):
        self.num_actions = 0

    def get_termination(self, env, info={}):
        """
        Aggreate termination conditions and fill info
        """
        done, info = super(DataTask, self).get_termination(env, info)

        return done, info

    def step(self, env):
        """
        Perform task-specific step: step visualization and aggregate path length

        :param env: environment instance
        """
        self.num_actions += 1
