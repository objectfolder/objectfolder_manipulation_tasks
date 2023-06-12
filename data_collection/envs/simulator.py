import logging
import os
import platform

import numpy as np
import pybullet as p
import taxim_robot
from panda_one_finger import Panda
from data_collection.setup import getObjInfo
import pybullet_data

log = logging.getLogger(__name__)

class Simulator:
    """
    Simulator class is a wrapper of physics simulator (pybullet) and MeshRenderer, it loads objects into
    both pybullet and also MeshRenderer and syncs the pose of objects and robot parts.
    """

    def __init__(
        self,
        env_config,
        gravity=9.81,
        physics_timestep=1 / 120.0,
        render_timestep=1 / 10.0,
        solver_iterations=100,
        image_width=480,
        image_height=640,
        use_pb_gui=False,
        gelsight_config="taxim_robot/config_r15_one_finger.yml",
    ):
        # physics simulator
        self.config = env_config
        self.gravity = gravity
        self.physics_timestep = physics_timestep
        self.render_timestep = render_timestep
        self.solver_iterations = solver_iterations
        self.physics_timestep_num = self.render_timestep / self.physics_timestep
        assert self.physics_timestep_num.is_integer(), "render_timestep must be a multiple of physics_timestep"

        self.physics_timestep_num = int(self.physics_timestep_num)

        self.image_width = image_width
        self.image_height = image_height
        self.use_pb_gui = use_pb_gui
        self.gelsight_config = gelsight_config

        self.initialize_physics_engine()
        self.import_scene()
        self.initialize_gelsight(width=image_width, height=image_height, visualize_gui=use_pb_gui, config_path=gelsight_config)

    def set_timestep(self, physics_timestep, render_timestep):
        """
        Set physics timestep and render (action) timestep.

        :param physics_timestep: physics timestep for pybullet
        :param render_timestep: rendering timestep for renderer
        """
        self.physics_timestep = physics_timestep
        self.render_timestep = render_timestep
        p.setTimeStep(self.physics_timestep)

    def disconnect(self, release_renderer=True):
        """
        Clean up the simulator.

        :param release_renderer: whether to release the MeshRenderer
        """
        if p.getConnectionInfo(self.cid)["isConnected"]:
            p.resetSimulation(physicsClientId=self.cid)
            p.disconnect(self.cid)

    def reload(self):
        """
        Destroy the MeshRenderer and physics simulator and start again.
        """
        self.disconnect()
        self.initialize_physics_engine()

    def initialize_physics_engine(self):
        """
        Initialize the physics engine (pybullet).
        """
        if self.use_pb_gui:
            self.cid = p.connect(p.GUI)
        else:
            self.cid = p.connect(p.DIRECT)

        # Needed for deterministic action replay
        p.resetSimulation()
        p.setPhysicsEngineParameter(deterministicOverlappingPairs=1)
        p.setPhysicsEngineParameter(numSolverIterations=self.solver_iterations)
        p.setTimeStep(self.physics_timestep)
        p.setGravity(0, 0, -self.gravity)
        p.setPhysicsEngineParameter(enableFileCaching=0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
        p.resetDebugVisualizerCamera(cameraDistance=0.6, cameraYaw=15, cameraPitch=-15,
                                cameraTargetPosition=[0.5, 0, 0.08])

    def import_scene(self):
        """
        Import a scene into the simulator. A scene could be a synthetic one or a realistic Gibson Environment.

        :param scene: a scene object to load
        """
        # TODO: import a large plane and robot in side
        planeId = p.loadURDF("plane.urdf")  # Create plane
        self.scene = planeId
        
    def initialize_gelsight(self, width=480, height=640, visualize_gui=False, config_path=None):
        # TODO: initialize a gelsight, togther with pyrender here
        self.gelsight = taxim_robot.Sensor(width=width, height=height, visualize_gui=visualize_gui, config_path=config_path)

        
    def import_object(self, obj, pos=[0,0,0], ori=[0,0,0]):
        """
        Import a non-robot object into the simulator.

        :param obj: a non-robot object to load
        """
        # Add object to pybullet and tacto simulator
        urdfObj, obj_mass, obj_height, force_range, deformation, _ = getObjInfo(obj)
        self.obj_height = self.config.get("obj_height", obj_height / 2 + 0.02)
        self.obj_x = self.config.get("obj_x", 0.5)
        self.objStartPos = [self.obj_x, 0., self.obj_height]
        self.objStartOrientation = p.getQuaternionFromEuler(ori)

        # add object in pybullet
        self.objID = p.loadURDF(urdfObj, self.objStartPos, self.objStartOrientation, useFixedBase=self.config.get("fix_obj", True))
        p.changeVisualShape(self.objID, -1, rgbaColor=[1, 0.5, 0.5, 1])
        # add object to pyrender, pos same as pybullet
        try:
            visual_file = urdfObj.replace("model.urdf", "visual.urdf")
            self.gelsight.add_object(visual_file, self.objID, force_range=force_range, deformation=deformation)
        except:
            self.gelsight.add_object(urdfObj, self.objID, force_range=force_range, deformation=deformation)
        # add object into pybullet and pyrender in the scene

    def import_robot(self, robotURDF_path, basePosition=[0, 0, -0.1] , useFixedBase=True):
        # import the robot into the pybullet
        self.robotID = p.loadURDF(robotURDF_path, basePosition=basePosition, useFixedBase=useFixedBase)
        self.robot = Panda(self.robotID)
        self.sensorID = self.robot.get_id_by_name(["panda_finger_joint1"])  # [21, 24]
        # add the link id to gelsight camera list (no camera is created in this step)
        self.gelsight.add_camera(self.robotID, self.sensorID)
        
    def step(self):
        """
        Step the simulation at self.render_timestep and update positions in renderer.
        """
        for _ in range(self.physics_timestep_num):
            p.stepSimulation()
        self.gelsight.update()
