import collections
import logging
import os
from dataclasses import dataclass

import cv2
import numpy as np
import pybullet as p
import trimesh
from urdfpy import URDF

from .renderer import Renderer

logger = logging.getLogger(__name__)


def _get_default_config(filename):
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), filename)

def get_digit_config_path():
    return _get_default_config("config_digit.yml")

def get_gelsight_config_path():
    return _get_default_config("config_gelsight.yml")

def get_r15_config_path():
    return _get_default_config("config_r15.yml")

def get_r15_one_config_path():
    return _get_default_config("config_r15_one_finger.yml")

@dataclass
class Link:
    obj_id: int  # pybullet obj ID
    link_id: int  # pybullet link ID (-1 means base)
    cid: int  # physicsClientId

    def get_pose(self):
        if self.link_id < 0:
            # get the base pose if link ID < 0
            position, orientation = p.getBasePositionAndOrientation(
                self.obj_id, physicsClientId=self.cid
            )
        else:
            # get the link pose if link ID >= 0
            position, orientation = p.getLinkState(
                self.obj_id, self.link_id, physicsClientId=self.cid
            )[:2]

        orientation = p.getEulerFromQuaternion(orientation, physicsClientId=self.cid)
        return position, orientation


class Sensor:
    def __init__(
            self,
            width=120,
            height=160,
            background=None,
            config_path=get_r15_one_config_path(),
            visualize_gui=True,
            show_depth=True,
            zrange=0.004,
            cid=0,
    ):
        """

        :param width: scalar
        :param height: scalar
        :param background: image
        :param visualize_gui: Bool
        :param show_depth: Bool
        :param config_path:
        :param cid: Int
        """
        self.cid = cid
        self.renderer = Renderer(width, height, background, config_path)

        self.visualize_gui = visualize_gui
        self.show_depth = show_depth
        self.zrange = zrange

        self.cameras = {}
        self.nb_cam = 0
        self.objects = {}
        self.object_poses = {}
        self.normal_forces = {}
        self.shear_forces = {}
        self._static = None

    @property
    def height(self):
        return self.renderer.height

    @property
    def width(self):
        return self.renderer.width

    @property
    def background(self):
        return self.renderer.background

    def add_camera(self, obj_id, link_ids):
        """
        Add camera into tacto

        self.cameras format: {
            "cam0": Link,
            "cam1": Link,
            ...
        }
        """
        if not isinstance(link_ids, collections.abc.Sequence):
            link_ids = [link_ids]

        for link_id in link_ids:
            cam_name = "cam" + str(self.nb_cam)
            self.cameras[cam_name] = Link(obj_id, link_id, self.cid)
            self.nb_cam += 1

    def add_object(self, urdf_fn, obj_id, globalScaling=1.0, force_range=None, deformation=None):
        # Load urdf file by urdfpy
        robot = URDF.load(urdf_fn)

        for link_id, link in enumerate(robot.links):
            if len(link.visuals) == 0:
                continue
            link_id = link_id - 1
            # Get each links
            visual = link.visuals[0]
            obj_trimesh = visual.geometry.meshes[0]

            # Set mesh color to default (remove texture)
            obj_trimesh.visual = trimesh.visual.ColorVisuals()

            # Set initial origin (pybullet pose already considered initial origin position, not orientation)
            pose = visual.origin
            # print("object's pose: {}".format(pose))

            # Scale if it is mesh object (e.g. STL, OBJ file)
            mesh = visual.geometry.mesh
            if mesh is not None and mesh.scale is not None:
                S = np.eye(4, dtype=np.float64)
                S[:3, :3] = np.diag(mesh.scale)
                pose = pose.dot(S)

            # Apply interial origin if applicable
            inertial = link.inertial
            if inertial is not None and inertial.origin is not None:
                pose = np.linalg.inv(inertial.origin).dot(pose)

            # Set global scaling
            pose = np.diag([globalScaling] * 3 + [1]).dot(pose)

            obj_trimesh = obj_trimesh.apply_transform(pose)
            obj_name = "{}_{}".format(obj_id, link_id)
            # print(obj_name)
            self.objects[obj_name] = Link(obj_id, link_id, self.cid)
            position, orientation = self.objects[obj_name].get_pose()
            # TODO: test to see if this really works?????
            # position = [position[1], -position[0], position[2]]
            # Add object in pyrender
            self.renderer.add_object(
                obj_trimesh,
                obj_name,
                position=position,  # [-0.015, 0, 0.0235],
                orientation=orientation,  # [0, 0, 0],
                force_range=force_range,
                deformation=deformation
            )

    def add_body(self, body):
        self.add_object(
            body.urdf_path, body.id, globalScaling=body.global_scaling or 1.0
        )

    def loadURDF(self, *args, **kwargs):
        """
        Load the object urdf to pybullet and tacto simulator.
        The tacto simulator will create the same scene in OpenGL for faster rendering
        """
        urdf_fn = args[0]
        globalScaling = kwargs.get("globalScaling", 1.0)
        force_range = kwargs.pop("force_range", None)
        deformation = kwargs.pop("deformation", None)
        # print(force_range[-4:], deformation[-4:])
        # Add to pybullet
        obj_id = p.loadURDF(physicsClientId=self.cid, *args, **kwargs)

        # Add to tacto simulator scene
        self.add_object(urdf_fn, obj_id, globalScaling=globalScaling, force_range=force_range, deformation=deformation)

        return obj_id

    def update(self):
        self._update_object_poses()

    def _update_object_poses(self):
        """
        Update the pose of each objects registered in tacto simulator
        """
        for obj_name in self.objects.keys():
            self.object_poses[obj_name] = self.objects[obj_name].get_pose()

    def get_force(self, cam_name):
        # Load contact force

        # get robot id and camera link id
        obj_id = self.cameras[cam_name].obj_id
        link_id = self.cameras[cam_name].link_id

        # get contact points that only involve this camera
        # see pybullet doc for use of this function
        pts = p.getContactPoints(
            bodyA=obj_id, linkIndexA=link_id, physicsClientId=self.cid
        )

        # accumulate forces from 0. using defaultdict of float
        self.normal_forces[cam_name] = collections.defaultdict(float)
        self.shear_forces[cam_name] = collections.defaultdict(lambda: [0, 0, 0])

        # print("number of contact pts", len(pts))
        for pt in pts:
            body_id_b = pt[2]
            link_id_b = pt[4]

            obj_name = "{}_{}".format(body_id_b, link_id_b)

            # ignore contacts we don't care (those not in self.objects)
            if obj_name not in self.objects:
                continue

            # Accumulate normal forces
            self.normal_forces[cam_name][obj_name] += pt[9]
            # calculate the equivalent force here
            self.shear_forces[cam_name][obj_name][0] += pt[11][0] * pt[10] + pt[13][0] * pt[12]
            self.shear_forces[cam_name][obj_name][1] += pt[11][1] * pt[10] + pt[13][1] * pt[12]
            self.shear_forces[cam_name][obj_name][2] += pt[11][2] * pt[10] + pt[13][2] * pt[12]
        # print("shear force", self.shear_forces)
        return self.normal_forces[cam_name], self.shear_forces[cam_name]

    @property
    def static(self):
        #TODO: just for debugging purpose
        if self._static is None:
            colors, _ = self.renderer.render(noise=False)
            depths = [np.zeros_like(d0) for d0 in self.renderer.depth0]
            self._static = (colors, depths)

        return self._static

    def _render_static(self):
        colors, depths = self.static
        colors = [self.renderer._add_noise(color) for color in colors]
        return colors, depths

    def render(self):
        """
        Render tacto images from each camera's view.
        """

        # first update object poses from pybullet to pyrender
        self._update_object_poses()
        colors = []
        depths = []

        for i in range(self.nb_cam):
            cam_name = "cam" + str(i)

            # get the contact normal forces of the link that camera is attached to
            # we use this normal forces to calculate depth
            normal_forces, shear_forces = self.get_force(cam_name)

            if normal_forces:
                # update cam pos/ori from camera link in pybullet
                position, orientation = self.cameras[cam_name].get_pose()
                # self.renderer.update_camera_pose(position, orientation)
                color, depth = self.renderer.render(self.object_poses, normal_forces, shear_forces=shear_forces,
                                                    camera_pos_old=position, camera_ori_old=orientation)

                # Remove the depth from curved gel
                for j in range(len(depth)):
                    depth[j] = self.renderer.depth0[j] - depth[j]
            else:
                # position, orientation = self.cameras[cam_name].get_pose()
                # self.renderer.update_camera_pose(position, orientation)
                color, depth = self._render_static()

            colors += color
            depths += depth

        return colors, depths

    def _depth_to_color(self, depth):
        gray = (np.clip(depth / self.zrange, 0, 1) * 255).astype(np.uint8)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def updateGUI(self, colors, depths):
        """
        Update images for visualization
        """
        if not self.visualize_gui:
            return

        # concatenate colors horizontally (axis=1)
        color = np.concatenate(colors, axis=1)
        # cv2.destroyAllWindows()
        if self.show_depth:
            # concatenate depths horizontally (axis=1)
            depth = np.concatenate(list(map(self._depth_to_color, depths)), axis=1)

            # concatenate the resulting two images vertically (axis=0)
            color_n_depth = np.concatenate([cv2.cvtColor(color, cv2.COLOR_BGR2RGB), depth], axis=0)

            cv2.imshow("color and depth", color_n_depth)
        else:
            cv2.imshow("color", cv2.cvtColor(color, cv2.COLOR_BGR2RGB))

        cv2.waitKey(1)
