from abc import ABCMeta, abstractmethod

import numpy as np
import h5py

from model_predictive_control.mpc.utils import ObservationList


class BaseEnv(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def reset_to(self, *args, **kwargs):
        pass

    @abstractmethod
    def reset_state(self, state):
        pass

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def compute_score(self, state, goal_state):
        pass

    @abstractmethod
    def get_state(self):
        pass

    @property
    def action_dimension(self):
        pass

    @property
    def observation_shape(self):
        pass

    @property
    def metadata(self):
        return None

    def goal_generator(self):
        """
        :return: An iterator that returns a tuple of three objects for a different task at each iteration:
        1) The starting state of the environment for the task
        2) The goal state of the environment for the task
        3) The observation image corresponding to the goal state of the environment for the task, as an ObservationList
        """
        pass

    def goal_generator_from_robosuite_hdf5(self, file_path, camera_name):
        """
        Create a goal generator from a robosuite-formatted hdf5 file.
        :param file_path: path to hdf5 goal file
        :param camera_name: name of the camera feed to load goal RGB, depth, etc. from.
        :return: generator object which yields a tuple of (start state, goal state, goal obs) for a different task
        at each iteration
        """
        f = h5py.File(
            file_path, "r", driver="core"
        )  # core prevents MP collision, but should just load in at once?
        demos = list(f["data"].keys())
        inds = np.argsort([int(elem[5:]) for elem in demos])
        demos = [demos[i] for i in inds]
        for ind in range(len(demos)):
            ep = f"demo_{ind + 1}"
            # load states
            states = f["data/{}/states".format(ep)][()]
            # load all goal images
            goals = dict()
            if self.env_hparams["goal_ims_from_data"]:
                goal_im_source = f[f"data/{ep}/goal_obs"]
            else:
                goal_im_source = f[f"data/{ep}/obs"]

            for modality in self.env_hparams["planning_modalities"]:
                if modality == "rgb":
                    goals[modality] = goal_im_source[f"{camera_name}_image"][:] / 255.0
                elif modality == "depth":
                    goals[modality] = goal_im_source[f"{camera_name}_depth"][:]
                    if goals[modality].shape[-1] != 1:
                        # Happens only for the iGibson renderer, TODO make cleaner
                        goals[modality] = goals[modality][..., None]
                elif modality == "normal":
                    normal_goals = goal_im_source[f"{camera_name}_normal"][:] / 255.0
                    goals[modality] = normal_goals
            goals = ObservationList(goals)

            # Determine which state from the trajectory or initial state to use as the start state
            # First, if the data contains start indices, use those
            if "start_index" in f[f"data/{ep}"]:
                start_idx = f[f"data/{ep}/start_index"][()]
                print(f"Using start index {start_idx} loaded from task benchmark!")
            # Otherwise, use the index specified in the hyperparameters
            else:
                if self.env_hparams["traj_start_idx"] == 1:
                    raise ValueError(
                        "Trajectory start index must be specified in hyperparameters if not in the goal dataset"
                    )
                else:
                    print(
                        f"Using default start index {self.env_hparams['traj_start_idx']} from config"
                    )
                    start_idx = self.env_hparams["traj_start_idx"]

            initial_state = dict(states=states[start_idx])
            initial_state["model"] = f[f"data/{ep}"].attrs.get("model_file", None)

            # Use either the final goal image or entire image sequence as the goal
            if self.env_hparams["use_final_goal_img"]:
                goals = goals[-1]
            else:
                goals = goals[start_idx:]

            goal_state = states[-1]

            yield initial_state, goal_state, goals
