import numpy as np
import warnings
from collections import namedtuple
import math
import os

class UR5Robotiq140:
    def __init__(self, pb, robot_params, use_gui):
        self.vis = use_gui
        self._pb = pb
        self.arm_num_dofs = 6
        self.action_scale = 0.2
        self.gripper_scale = 0.02
        if "tcp_link_name" in robot_params:
            self.tcp_link_name = robot_params["tcp_link_name"]
        else:
            self.tcp_link_name = "ee_link"
        self.arm_rest_poses = robot_params["reset_arm_poses"]  # default joint pose for ur5
        self.gripper_range = robot_params["reset_gripper_range"]
        self.load_urdf()
        self.left_finger_pad_id = self.link_name_to_index['left_inner_finger_pad']
        self.right_finger_pad_id = self.link_name_to_index['right_inner_finger_pad']
        
        # set info specific to arm
        self.setup_ur5_info()
        self.setup_gripper_info()

        # reset the arm to rest poses
        # self.reset()

    def close(self):
        if self._pb.isConnected():
            self._pb.disconnect()

    def load_urdf(self):
        """
        Load the robot arm model into pybullet
        """
        self.base_pos = [0, 0, 0]
        self.base_rpy = [0, 0, 0]
        self.base_orn = self._pb.getQuaternionFromEuler(self.base_rpy)
        asset_name = "./assets/urdfs/ur5_robotiq_140.urdf"
        self.embodiment_id = self._pb.loadURDF(asset_name, self.base_pos, self.base_orn, useFixedBase=True, flags=self._pb.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)

        # create dicts for mapping link/joint names to corresponding indices
        self.num_joints, self.link_name_to_index, self.joint_name_to_index = self.create_link_joint_mappings(self.embodiment_id)

        # get the link and tcp IDs
        self.tcp_link_id = self.link_name_to_index[self.tcp_link_name]

    def create_link_joint_mappings(self, urdf_id):

        num_joints = self._pb.getNumJoints(urdf_id)

        # pull relevent info for controlling the robot
        joint_name_to_index = {}
        link_name_to_index = {}
        for i in range(num_joints):
            info = self._pb.getJointInfo(urdf_id, i)
            joint_name = info[1].decode("utf-8")
            link_name = info[12].decode("utf-8")
            joint_name_to_index[joint_name] = i
            link_name_to_index[link_name] = i

        return num_joints, link_name_to_index, joint_name_to_index
    def reset(self,gripper_enable):
        for rest_pose, joint_id in zip(self.arm_rest_poses, self.arm_controllable_joints):
            self._pb.resetJointState(self.embodiment_id, joint_id, rest_pose, 0)
            self._pb.changeDynamics(self.embodiment_id, joint_id, linearDamping=0.04, angularDamping=0.04)
            self._pb.changeDynamics(self.embodiment_id, joint_id, jointDamping=0.01)
        open_angle = 0.715 - math.asin((self.gripper_range[1] - 0.010) / 0.1143)  # angle calculation
        if gripper_enable:
            self._pb.resetJointState(self.embodiment_id, self.mimic_parent_id, open_angle, 0)
        else:
            self._pb.setJointMotorControl2(self.embodiment_id, self.mimic_parent_id, self._pb.POSITION_CONTROL, targetPosition=open_angle,
                                force=self.joints[self.mimic_parent_id].maxForce, maxVelocity=self.joints[self.mimic_parent_id].maxVelocity)
        
        
        
    def open_gripper(self):
        current_gripper_open_length = math.sin(0.715-self._pb.getJointState(self.embodiment_id, self.mimic_parent_id)[0])*0.1143 + 0.010
        action = (self.gripper_range[1] - current_gripper_open_length) / self.gripper_scale
        self.move_gripper(action)

    def close_gripper(self):
        current_gripper_open_length = math.sin(0.715-self._pb.getJointState(self.embodiment_id, self.mimic_parent_id)[0])*0.1143 + 0.010
        action = (self.gripper_range[0] - current_gripper_open_length) / self.gripper_scale
        self.move_gripper(action)

    def move_gripper(self, action):
        current_gripper_open_length = math.sin(0.715-self._pb.getJointState(self.embodiment_id, self.mimic_parent_id)[0])*0.1143 + 0.010
        target_gripper_open_length = np.clip(current_gripper_open_length + action * self.gripper_scale, *self.gripper_range)
        open_angle = 0.715 - math.asin((target_gripper_open_length - 0.010) / 0.1143)  # angle calculation
        # Control the mimic gripper joint(s)
        self._pb.setJointMotorControl2(self.embodiment_id, self.mimic_parent_id, self._pb.POSITION_CONTROL, targetPosition=open_angle,
                                force=self.joints[self.mimic_parent_id].maxForce, maxVelocity=self.joints[self.mimic_parent_id].maxVelocity)
    def move_ee(self, action, control_method):
        '''
        Move the end effector of the robot
        action: (np.ndarray)
        '''
        assert control_method in ('joint', 'end')
        if control_method == 'end':
            ee_displacement = action * self.action_scale  # limit maximum change in position
            ee_position =np.array(self._pb.getLinkState(self.embodiment_id, self.tcp_link_id)[4])
            # self._pb.addUserDebugPoints(pointPositions = [ee_position], pointColorsRGB = [[0, 0, 255]], pointSize= 40, lifeTime= 0)
            target_ee_position = ee_position + ee_displacement
            # Clip the height target. For some reason, it has a great impact on learning
            target_ee_position[2] = np.max((0, target_ee_position[2]))
            joint_poses =np.array(self._pb.calculateInverseKinematics(self.embodiment_id, self.tcp_link_id, 
                                                                      target_ee_position,
                                                                    #   targetOrientation = self._pb.getQuaternionFromEuler([0, math.pi/2, 0]),
                                                                      lowerLimits = self.arm_lower_limits, 
                                                                      upperLimits = self.arm_upper_limits, 
                                                                      jointRanges = self.arm_joint_ranges,
                                                                    #   solver = 1,
                                                                      maxNumIterations=200))
            joint_poses = joint_poses[:self.arm_num_dofs]
            # self._pb.addUserDebugPoints(pointPositions = [target_ee_position], pointColorsRGB = [[0, 0, 255]], pointSize= 40, lifeTime= 0)
        elif control_method == 'joint':
            assert len(action) == self.arm_num_dofs
            arm_joint_ctrl = action * self.action_scale  # limit maximum change in position
            current_arm_joint_angles = np.array([self._pb.getJointState(self.embodiment_id, i)[0] for i in self.arm_controllable_joints])
            joint_poses = current_arm_joint_angles + arm_joint_ctrl
        # arm
        for i, joint_id in enumerate(self.arm_controllable_joints):
            self._pb.setJointMotorControl2(self.embodiment_id, joint_id, self._pb.POSITION_CONTROL, joint_poses[i],
                                    force=self.joints[joint_id].maxForce, maxVelocity=self.joints[joint_id].maxVelocity)

    def setup_ur5_info(self):
        """
        Set some of the parameters used when controlling the UR5
        """
        self.name = 'sim_ur5'
        self.max_force = 1000.0
        self.pos_gain = 1.0
        self.vel_gain = 1.0

        jointInfo = namedtuple('jointInfo', 
            ['id','name','type','damping','friction','lowerLimit','upperLimit','maxForce','maxVelocity','controllable'])
        self.joints = []
        self.control_joint_ids = []
        for i in range(self._pb.getNumJoints(self.embodiment_id)):
            info = self._pb.getJointInfo(self.embodiment_id, i)
            jointID = info[0]
            jointName = info[1].decode("utf-8")
            jointType = info[2]  # JOINT_REVOLUTE, JOINT_PRISMATIC, JOINT_SPHERICAL, JOINT_PLANAR, JOINT_FIXED
            jointDamping = info[6]
            jointFriction = info[7]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            controllable = (jointType != self._pb.JOINT_FIXED)
            if controllable:
                self.control_joint_ids.append(jointID)
                self._pb.setJointMotorControl2(self.embodiment_id, jointID, self._pb.VELOCITY_CONTROL, targetVelocity=0, force=0)
            info = jointInfo(jointID,jointName,jointType,jointDamping,jointFriction,jointLowerLimit,
                            jointUpperLimit,jointMaxForce,jointMaxVelocity,controllable)
            self.joints.append(info)
        self.control_joint_names = [self.joints[i].name for i in self.control_joint_ids]

        # get the control and calculate joint ids in list form, useful for pb array methods
        assert self.control_joint_ids == [self.joint_name_to_index[name] for name in self.control_joint_names]
        self.num_control_dofs = len(self.control_joint_ids)
        assert self.num_control_dofs >= self.arm_num_dofs
        self.arm_controllable_joints = self.control_joint_ids[:self.arm_num_dofs]
        self.arm_lower_limits = [info.lowerLimit for info in self.joints if info.controllable][:self.arm_num_dofs]
        self.arm_upper_limits = [info.upperLimit for info in self.joints if info.controllable][:self.arm_num_dofs]
        self.arm_joint_ranges = [info.upperLimit - info.lowerLimit for info in self.joints if info.controllable][:self.arm_num_dofs]
    
    def setup_gripper_info(self):
        mimic_parent_name = 'left_outer_knuckle_joint'
        mimic_children_names = {'right_outer_knuckle_joint': -1,
                                'left_inner_knuckle_joint': -1,
                                'right_inner_knuckle_joint': -1,
                                'left_inner_finger_joint': 1,
                                'right_inner_finger_joint': 1}
        self.mimic_parent_id = [joint.id for joint in self.joints if joint.name == mimic_parent_name][0]
        self.mimic_child_multiplier = {joint.id: mimic_children_names[joint.name] for joint in self.joints if joint.name in mimic_children_names}

        for joint_id, multiplier in self.mimic_child_multiplier.items():
            c = self._pb.createConstraint(self.embodiment_id, self.mimic_parent_id,
                                   self.embodiment_id, joint_id,
                                   jointType=self._pb.JOINT_GEAR,
                                   jointAxis=[0, 1, 0],
                                   parentFramePosition=[0, 0, 0],
                                   childFramePosition=[0, 0, 0])
            self._pb.changeConstraint(c, gearRatio=-multiplier, maxForce=100, erp=1)  # Note: the mysterious `erp` is of EXTREME importance

    def step_simulation(self):
        raise RuntimeError('`step_simulation` method of RobotBase Class should be hooked by the environment.')
    
    def get_joint_obs(self,control_type,gripper_enable) -> np.ndarray:
        positions = []
        if control_type == 'joint':
            control_joint_ls = self.control_joint_ids if gripper_enable else self.arm_controllable_joints
            for joint_id in control_joint_ls:
                pos, _, _, _ = self._pb.getJointState(self.embodiment_id, joint_id)
                positions.append(pos)
        else:
            # positions_arm = self._pb.getLinkState(self.embodiment_id, self.tcp_link_id)[4]
            left_finger_info = self._pb.getLinkState(self.embodiment_id, self.left_finger_pad_id,computeLinkVelocity=1)
            right_finger_info = self._pb.getLinkState(self.embodiment_id, self.right_finger_pad_id,computeLinkVelocity=1)
            finger_pos = list((np.array(left_finger_info[4])+np.array(right_finger_info[4]))/2)
            ee_orn = list(self._pb.getEulerFromQuaternion(np.array(self._pb.getLinkState(self.embodiment_id, self.tcp_link_id)[5])))
            finger_linear_veocity = list((np.array(left_finger_info[6])+np.array(right_finger_info[6]))/2)
            finger_angular_veocity = list((np.array(left_finger_info[7])+np.array(right_finger_info[7]))/2)
            arm_obs = finger_pos+ee_orn+finger_linear_veocity+finger_angular_veocity
            # self._pb.addUserDebugPoints(pointPositions = [positions_arm], pointColorsRGB = [[0, 0, 255]], pointSize= 40, lifeTime= 0)
            if gripper_enable:
                positions_gripper = []
                for joint_id in self.control_joint_ids[self.arm_num_dofs:]:
                    pos, _, _, _ = self._pb.getJointState(self.embodiment_id, joint_id)
                    positions_gripper.append(pos)
                positions = arm_obs+positions_gripper
            else:
                positions = arm_obs
        robot_obs = np.array(positions)
        return robot_obs