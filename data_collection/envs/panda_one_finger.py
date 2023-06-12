import time
import numpy as np
import math
import pybullet as pb
import copy
from transforms3d.quaternions import *

useNullSpace = 1
ikSolver = 0
pandaEndEffectorIndex = 8 #8
pandaNumDofs = 7

#lower limits for null space (TODO: set them to proper range)
ll = [-7]*pandaNumDofs
#upper limits for null space (TODO: set them to proper range)
ul = [7]*pandaNumDofs
#joint ranges for null space (TODO: set them to proper range)
jr = [7]*pandaNumDofs
# restposes for null space
jointPositions=(0.8045609285966308, 0.525471701354679, -0.02519566900946519, -1.3925086098003587, 0.013443782914225877, 1.9178323512245277, -0.007207024243406651, 0)
            # [0.98, 0.458, 0.31, -2.24, -0.30, 2.66, 2.32, 0.02, 0.02]
rp = jointPositions

# robot params
ee_fingertip_T_mat = np.array([[0.707, 0.707, 0, 0], [-0.707, 0.707, 0, 0], [0, 0, 1, 0.1034488], [0, 0, 0, 1]])
gel_ft_trans = np.array([0.0147249, 0.0, 0.0970000]) #x y z direction may wrong, need tuning
gel_ft_rot = np.array([[0.7071068,  0.7071068, 0], [-0.7071068,  0.7071068, 0], [0, 0, 1]])

class Panda(object):
    def __init__(self, robotID):
        self.robotID = robotID
        index = 0

        # reset panda joints
        for j in range(pb.getNumJoints(self.robotID)):
            pb.changeDynamics(self.robotID, j, linearDamping=0, angularDamping=0)
            info = pb.getJointInfo(self.robotID, j)
            #print("info=",info)
            jointType = info[2]
            ## Joint that moves linearly 
            if (jointType == pb.JOINT_PRISMATIC):
                pb.resetJointState(self.robotID, j, jointPositions[index]) 
                index=index+1
            ## Joint that revolves
            if (jointType == pb.JOINT_REVOLUTE):
                pb.resetJointState(self.robotID, j, jointPositions[index]) 
                index=index+1
        # self.t = 0.
        self.armNames = [
            "panda_joint1",
            "panda_joint2",
            "panda_joint3",
            "panda_joint4",
            "panda_joint5",
            "panda_joint6",
            "panda_joint7",
        ]

        self.armJoints = self.get_id_by_name(self.armNames)
        self.armControlID = self.get_control_id_by_name(self.armNames)
        print(self.armControlID)

        self.gripperNames = [
            "panda_finger_joint1", #9
        ]
        self.gripperJoints = self.get_id_by_name(self.gripperNames)
        pb.enableJointForceTorqueSensor(self.robotID, self.gripperJoints[0])
        
        # Get ID for end effector
        self.eeName = ["panda_hand_joint"]
        self.eefID = self.get_id_by_name(self.eeName)[0]

        self.armHome = [0.0, -0.524, 0.0, -2.617, 0.0, 2.094, 0.0]
        self.jointHome = [0.08, 0.08]

        self.pos = np.array([0.581, 0.002, 0.3])
        self.ori = np.array([np.pi, 0, 0]) #[0, np.pi, -np.pi/2]
        self.width = 2 * self.jointHome[0] #- self.jointHome[1]
        self.rot = 0

        self.tol = 1e-9
        self.delta_pos = 0.05
        self.delta_rot = np.pi / 10
        self.delta_width = 0.01
        self.init_robot()
    
    def get_joint_limit(self, joint_ind):
        """Get the limit of the joint.

        These limits are specified by the URDF.

        Parameters
        ----------
        joint_uid :
            A tuple of the body Unique ID and the joint index.

        Returns
        -------
        limit
            A dictionary of lower, upper, effort and velocity.

        """
        (_, _, _, _, _, _, _, _, lower, upper, max_force, max_vel, _,
         _, _, _, _) = pb.getJointInfo(
            bodyUniqueId=self._arm_id,
            jointIndex=joint_ind,
            physicsClientId=self._physics_id)

        limit = {
            'lower': lower,
            'upper': upper,
            'effort': max_force,
            'velocity': max_vel}

        return limit
        
    def get_id_by_name(self, names):
        """
        get joint/link ID by name
        """
        nbJoint = pb.getNumJoints(self.robotID)
        jointNames = {}
        for i in range(nbJoint):
            name = pb.getJointInfo(self.robotID, i)[1].decode()
            jointNames[name] = i

        return [jointNames[name] for name in names]

    def get_control_id_by_name(self, names):
        """
        get joint/link ID by name
        """
        nbJoint = pb.getNumJoints(self.robotID)
        jointNames = {}
        ctlID = 0
        for i in range(nbJoint):
            jointInfo = pb.getJointInfo(self.robotID, i)
            name = jointInfo[1].decode("utf-8")
            # skip fixed joint
            if jointInfo[2] == 4:
                continue

            # skip base joint
            # if jointInfo[-1] == -1:
            #     continue
            jointNames[name] = ctlID
            ctlID += 1
        return [jointNames[name] for name in names]

    def reset_robot(self):
        for j in range(len(self.armJoints)):
            pb.resetJointState(self.robotID, self.armJoints[j], self.armHome[j])
        for j in range(len(self.gripperJoints)):
            pb.resetJointState(self.robotID, self.gripperJoints[j], self.jointHome[j])
    
    def init_robot(self):
        self.reset_robot()
        self.pos = np.array([0.2, 0.002, 0.4])
        self.ori = np.array([np.pi, 0, 0])
        ori = self.ori.copy()
        ori[-1] = self.rot
        self.go(self.pos, ori)
        for _ in range(10):
            pb.stepSimulation()
    
    # Get the position and orientation of the UR5 end-effector
    def get_ee_pose(self):
        res = pb.getLinkState(self.robotID, self.eefID)
        world_positions = res[0]
        world_orientations = res[1]
        return world_positions, world_orientations

    def get_all_state(self):
        all_states = [_[0] for _ in pb.getJointStates(self.robotID, self.armJoints)]
        return all_states

    # Get the joint angles (6 ur5 joints)
    def get_arm_angles(self):
        joint_angles = [_[0] for _ in pb.getJointStates(self.robotID, self.armJoints)]
        return joint_angles

    # Get the gripper width gripper width
    def get_gripper_width(self):
        width = 2 * np.abs(pb.getJointState(self.robotID, self.gripperJoints[-1])[0])
        return width

    def get_rotation(self):
        _, ori_q = self.get_ee_pose()
        ori = pb.getEulerFromQuaternion(ori_q)
        rot = ori[-1]
        return rot

    def operate(self, pos, rot=None, width=None, gripForce=20, wait=False):
        ori = self.ori.copy()
        ori[-1] = rot
        self.go(pos, ori, width=width, gripForce=gripForce, wait=wait)
    
    def go(self, _pos, _ori=None, width=None, wait=False, gripForce=20, max_vel=None):
        if _ori is None:
            _ori = self.ori
        _quat = pb.getQuaternionFromEuler(_ori)
        quat = np.array(copy.deepcopy(_quat))
        pos = np.array(copy.deepcopy(_pos))
        tmp = quat[3]
        quat[3] = quat[2]
        quat[2] = quat[1]
        quat[1] = quat[0]
        quat[0] = tmp
        fingertip_pos = pos - (quat2mat(quat) @ gel_ft_trans.T).T
        # transform finger mid point to ee pose
        ee_ori = quat
        ee_pos = (fingertip_pos - (quat2mat(ee_ori) @ ee_fingertip_T_mat[:3, 3].T).T)
        tmp = ee_ori[0]
        ee_ori[0] = ee_ori[1]
        ee_ori[1] = ee_ori[2]
        ee_ori[2] = ee_ori[3]
        ee_ori[3] = tmp
        
        ori_q = ee_ori
        pos = ee_pos
        width=0.0
        ## Calculate target position of each joint
        jointPose = pb.calculateInverseKinematics(self.robotID, self.eefID, pos, ori_q, ll, ul, jr, rp, maxNumIterations=300, residualThreshold=0.000001)
        jointPose = np.array(jointPose)
        maxForces = np.ones(len(jointPose)) * 2000
        # Select the relavant joints for arm and gripper
        jointPose = jointPose[self.armControlID]
        maxForces = maxForces[self.armControlID]
        ## Set position for both grippers\
        if max_vel is not None:
            maxVelocities = np.ones(len(jointPose)) * max_vel
            maxVelocities = maxForces[self.armControlID]
            pb.setJointMotorControlArray(
                self.robotID,
                self.armJoints,
                pb.POSITION_CONTROL,
                targetVelocities= maxVelocities,
                targetPositions=jointPose,
                forces=maxForces,
            )
        else:
            pb.setJointMotorControlArray(
                self.robotID,
                self.armJoints,
                pb.POSITION_CONTROL,
                # targetVelocities= max_vel,
                targetPositions=jointPose,
                forces=maxForces,
                # positionGains=[1.2] * 7,
                # velocityGains=[1.8] * 7,
            )

        ## Can use pb.getJointState for joint position

        self.pos = np.array(_pos)
        if _ori is not None:
            self.ori = _ori
        if width is not None:
            self.width = width

        if wait:
            last_err = 1e6
            while True:
                pb.stepSimulation()
                ee_pose = self.get_ee_pose()
                w = self.get_gripper_width()
                err = (
                        np.sum(np.abs(np.array(ee_pose[0]) - pos))
                        + np.sum(np.abs(np.array(ee_pose[1]) - ori_q))
                        + np.abs(w - width)
                )
                diff_err = last_err - err
                last_err = err

                if np.abs(diff_err) < self.tol:
                    break

        # print("FINISHED")