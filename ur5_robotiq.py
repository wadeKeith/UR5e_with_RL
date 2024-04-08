import numpy as np
import warnings
from collections import namedtuple
import math
import os

class UR5Robotiq140:
    def __init__(self, pb, robot_params):

        self._pb = pb
        self.arm_num_dofs = 6
        if "tcp_link_name" in robot_params:
            self.tcp_link_name = robot_params["tcp_link_name"]
        else:
            self.tcp_link_name = "ee_link"
        self.arm_rest_poses = robot_params["reset_arm_poses"]  # default joint pose for ur5
        self.gripper_range = robot_params["reset_gripper_range"]
        self.load_urdf()

        

        self._min_constant_vel = 0.0001
        self._max_constant_vel = 0.001
        self.set_constant_vel_percentage(percentage=25)
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
        asset_name = os.path.join('/Users/yin/Documents/GitHub/robotics_pybullet_learn/UR5', "ur5/urdfs/ur5_robotiq_140.urdf")
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
    def reset(self):
        self.reset_arm()
        self.reset_gripper()

    def reset_arm(self):
        """
        reset to rest poses
        """
        for rest_pose, joint_id in zip(self.arm_rest_poses, self.arm_controllable_joints):
            self._pb.resetJointState(self.embodiment_id, joint_id, rest_pose)
            self._pb.changeDynamics(self.embodiment_id, joint_id, linearDamping=0.04, angularDamping=0.04)
            self._pb.changeDynamics(self.embodiment_id, joint_id, jointDamping=0.01)
        self._pb.setJointMotorControlArray(
            bodyIndex=self.embodiment_id,
            jointIndices=self.arm_controllable_joints,
            controlMode=self._pb.POSITION_CONTROL,
            targetPositions=self.arm_rest_poses,
            targetVelocities=[0] * self.arm_num_dofs,
            positionGains=[self.pos_gain] * self.arm_num_dofs,
            velocityGains=[self.vel_gain] * self.arm_num_dofs,
            forces=np.zeros(self.arm_num_dofs) + self.max_force,
        )
        # Wait for a few steps
        # for _ in range(10):
        #     self.step_simulation()

    def reset_gripper(self):
        self.open_gripper()

    def open_gripper(self):
        self.move_gripper(self.gripper_range[1])

    def close_gripper(self):
        self.move_gripper(self.gripper_range[0])

    def move_gripper(self, open_length):
        # open_length = np.clip(open_length, *self.gripper_range)
        open_angle = 0.715 - math.asin((open_length - 0.010) / 0.1143)  # angle calculation
        # Control the mimic gripper joint(s)
        self._pb.setJointMotorControl2(self.embodiment_id, self.mimic_parent_id, self._pb.POSITION_CONTROL, targetPosition=open_angle,
                                force=self.joints[self.mimic_parent_id].maxForce, maxVelocity=self.joints[self.mimic_parent_id].maxVelocity)

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
        # joints which can be controlled (not fixed)
        # self.control_joint_ids = [i for i in range(self._pb.getNumJoints(self.embodiment_id)) if self._pb.getJointInfo(self.embodiment_id, i)[2] != self._pb.JOINT_FIXED]
        # self.control_joint_names = [
        #     "base_joint",
        #     "shoulder_joint",
        #     "elbow_joint",
        #     "wrist_1_joint",
        #     "wrist_2_joint",
        #     "wrist_3_joint",
        # ]
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

        

    def get_current_joint_pos_vel(self):
        """
        Get the current joint states of the ur5
        """
        cur_joint_states = self._pb.getJointStates(self.embodiment_id, self.control_joint_ids)
        cur_joint_pos = [cur_joint_states[i][0] for i in range(self.num_control_dofs)]
        cur_joint_vel = [cur_joint_states[i][1] for i in range(self.num_control_dofs)]
        return cur_joint_pos, cur_joint_vel

    def get_current_tcp_pose_vel(self):
        """
        Get the current velocity of the TCP
        """
        tcp_state = self._pb.getLinkState(
            self.embodiment_id,
            self.tcp_link_id,
            computeLinkVelocity=True,
            computeForwardKinematics=False,
        )
        tcp_pos = np.array(tcp_state[0])  # worldLinkPos
        tcp_orn = np.array(tcp_state[1])  # worldLinkOrn
        tcp_rpy = self._pb.getEulerFromQuaternion(tcp_orn)
        tcp_pose = np.array([*tcp_pos, *tcp_rpy])

        tcp_lin_vel = np.array(tcp_state[6])  # worldLinkLinearVelocity
        tcp_ang_vel = np.array(tcp_state[7])  # worldLinkAngularVelocity
        tcp_vel = np.array([*tcp_lin_vel, *tcp_ang_vel])

        return tcp_pose, tcp_vel

    def get_tcp_pose(self):
        """
        Returns pose of the Tool Center Point in world frame.
        """
        cur_TCP_pose, _ = self.get_current_tcp_pose_vel()
        return np.array(cur_TCP_pose)

    def get_tcp_vel(self):
        """
        Returns velocity of the Tool Center Point in world frame.
        """
        _, cur_TCP_vel = self.get_current_tcp_pose_vel()
        return np.array(cur_TCP_vel)

    def get_joint_angles(self):
        """
        Returns joint positions of the robot arm.
        """
        joint_pos, _ = self.get_current_joint_pos_vel()
        return np.array(joint_pos)

    def get_joint_vel(self):
        """
        Returns joint velocities of the robot arm.
        """
        _, joint_vel = self.get_current_joint_pos_vel()
        return np.array(joint_vel)

    def compute_gravity_compensation(self):
        """
        Calculates torques to apply that compensate for effect of gravity.
        """
        cur_joint_pos, cur_joint_vel = self.get_current_joint_pos_vel()
        grav_comp_torque = self._pb.calculateInverseDynamics(
            self.embodiment_id, cur_joint_pos, cur_joint_vel, [0] * self.num_control_dofs
        )
        return np.array(grav_comp_torque)

    def apply_gravity_compensation(self):
        """
        Applys motor torques that compensate for gravity.
        """
        grav_comp_torque = self.compute_gravity_compensation()

        self._pb.setJointMotorControlArray(
            bodyIndex=self.embodiment_id,
            jointIndices=self.control_joint_ids,
            controlMode=self._pb.TORQUE_CONTROL,
            forces=grav_comp_torque,
        )

    def set_target_tcp_pose(self, target_pose):
        """
        Go directly to a tcp position specified relative to the worldframe.
        """

        # transform from work_frame to world_frame
        target_pos, target_rpy = target_pose[:3], target_pose[3:]
        target_orn = np.array(self._pb.getQuaternionFromEuler(target_rpy))

        # get target joint poses through IK
        joint_positions = self._pb.calculateInverseKinematics(
            self.embodiment_id,
            self.tcp_link_id,
            target_pos,
            target_orn,
            restPoses=self.arm_rest_poses,
            maxNumIterations=100,
            residualThreshold=1e-8,
        )
        joint_velocities = np.array([0] * self.num_control_dofs)

        # set joint control
        self._pb.setJointMotorControlArray(
            self.embodiment_id,
            self.control_joint_ids,
            self._pb.POSITION_CONTROL,
            targetPositions=joint_positions,
            targetVelocities=joint_velocities,
            positionGains=[self.pos_gain] * self.num_control_dofs,
            velocityGains=[self.vel_gain] * self.num_control_dofs,
            forces=[self.max_force] * self.num_control_dofs,
        )

        # set target positions for blocking move
        self._target_joints_positions = np.array(joint_positions)

    def set_target_tcp_velocities(self, target_vels):
        """
        Set desired tcp velocity.
        """

        # get current joint positions and velocities
        q, qd = self.get_current_joint_pos_vel()

        # calculate the jacobian for tcp link
        # used to map joing velocities to TCP velocities
        jac_t, jac_r = self._pb.calculateJacobian(
            self.embodiment_id,
            self.tcp_link_id,
            [0, 0, 0],
            q,
            qd,
            [0] * self.num_control_dofs,
        )

        # merge into one jacobian matrix
        jac = np.concatenate([np.array(jac_t), np.array(jac_r)])

        # invert the jacobian to map from tcp velocities to joint velocities
        # be careful of singnularities and non square matrices
        # use pseudo-inverse when this is the case
        # this is all the time for 7 dof arms like panda
        if jac.shape[1] > np.linalg.matrix_rank(jac.T):
            inv_jac = np.linalg.pinv(jac)
        else:
            inv_jac = np.linalg.inv(jac)

        # convert desired velocities from cart space to joint space
        joint_vels = np.matmul(inv_jac, target_vels)

        # apply joint space velocities
        self._pb.setJointMotorControlArray(
            self.embodiment_id,
            self.control_joint_ids,
            self._pb.VELOCITY_CONTROL,
            targetVelocities=joint_vels,
            velocityGains=[self.vel_gain] * self.num_control_dofs,
            forces=[self.max_force] * self.num_control_dofs,
        )

        # set target positions for blocking move
        self._target_joints_velocities = np.array(joint_vels)

    def set_target_joint_positions(self, joint_positions):
        """
        Go directly to a specified joint configuration.
        """
        joint_velocities = np.array([0] * self.num_control_dofs)

        # set joint control
        self._pb.setJointMotorControlArray(
            self.embodiment_id,
            self.control_joint_ids,
            self._pb.POSITION_CONTROL,
            targetPositions=joint_positions,
            targetVelocities=joint_velocities,
            positionGains=[self.pos_gain] * self.num_control_dofs,
            velocityGains=[self.vel_gain] * self.num_control_dofs,
            forces=[self.max_force] * self.num_control_dofs,
        )

        # set target positions for blocking move
        self._target_joints_positions = np.array(joint_positions)

    def set_target_joint_velocities(self, joint_velocities):
        """
        Set the desired joint velicities.
        """
        self._pb.setJointMotorControlArray(
            self.embodiment_id,
            self.control_joint_ids,
            self._pb.VELOCITY_CONTROL,
            targetVelocities=joint_velocities,
            positionGains=[self.pos_gain] * self.num_control_dofs,
            velocityGains=[self.vel_gain] * self.num_control_dofs,
            forces=[self.max_force] * self.num_control_dofs,
        )

    def step_sim(self):
        """
        Take a step of the simulation whilst applying neccessary forces
        """

        # compensate for the effect of gravity
        self.apply_gravity_compensation()

        # step the simulation
        self._pb.stepSimulation()

    def set_constant_vel_percentage(self, percentage):
        """
        Sets constant velocity for position moves as a percentage of maximum.
        """
        if percentage == float("inf"):
            self._constant_vel = None
            self._max_position_move_steps = 1000
        else:
            if percentage < 1 or percentage > 100:
                raise Exception("Speed value outside range of 1-100%")

            constant_vel_range = self._max_constant_vel - self._min_constant_vel
            self._constant_vel = self._min_constant_vel + constant_vel_range * (percentage / 100.0)
            self._max_position_move_steps = 10000

    def get_constant_vel_percentage(self):
        """
        Gets constant velocity used for position moves as a percentage of maximum..
        """
        if self._constant_vel is None:
            return float("inf")
        else:
            constant_vel_range = self._max_constant_vel - self._min_constant_vel
            return (self._constant_vel - self._min_constant_vel) / constant_vel_range

    def blocking_position_move(
        self,
        max_steps=1000,
        constant_vel=None,
        j_pos_tol=0.1,
        j_vel_tol=0.1,
    ):
        """
        step the simulation until a target position has been reached or the max
        number of steps has been reached
        """
        # get target position
        targ_j_pos = self._target_joints_positions

        for i in range(max_steps):

            # get the current joint positions and velocities
            cur_j_pos, cur_j_vel = self.get_current_joint_pos_vel()

            # Move with constant velocity (from google-ravens)
            # break large position move to series of small position moves.
            if constant_vel is not None:
                diff_j = np.array(targ_j_pos) - np.array(cur_j_pos)
                norm = np.linalg.norm(diff_j)
                v = diff_j / norm if norm > 0 else np.zeros_like(cur_j_pos)
                step_j = cur_j_pos + v * constant_vel

                # reduce vel if joints are close enough,
                # this helps to acheive final pose
                if all(np.abs(diff_j) < constant_vel):
                    constant_vel /= 2

                # set joint control
                self._pb.setJointMotorControlArray(
                    self.embodiment_id,
                    self.control_joint_ids,
                    self._pb.POSITION_CONTROL,
                    targetPositions=step_j,
                    targetVelocities=[0.0] * self.num_control_dofs,
                    positionGains=[self.pos_gain] * self.num_control_dofs,
                    velocityGains=[self.vel_gain] * self.num_control_dofs
                )

            # step the simulation
            self.step_sim()

            # calc totoal velocity
            j_pos_err = np.sum(np.abs(targ_j_pos - cur_j_pos))
            j_vel_err = np.sum(np.abs(cur_j_vel))

            # break if the pose error is small enough
            # and the velocity is low enough
            if (j_pos_err < j_pos_tol) and (j_vel_err < j_vel_tol):
                break

        # Warn user is correct pose was not reached within max steps
        if i == max_steps-1:
            warnings.warn("Blocking position move failed to reach tolerance within max_steps.")

    def blocking_velocity_move(
        self,
        blocking_steps=100
    ):
        """
        step the simulation until a target position has been reached or the max
        number of steps has been reached
        """
        for i in range(blocking_steps):
            # step the simulation
            self.step_sim()

    def apply_blocking_velocity_move(self, blocking_steps):
        self.blocking_velocity_move(blocking_steps=blocking_steps)

    def apply_blocking_position_move(self, quick_mode=False):
        if not quick_mode:
            # slow but more realistic moves
            self.blocking_position_move(
                max_steps=self._max_position_move_steps,
                constant_vel=self._constant_vel,
                j_pos_tol=1e-6,
                j_vel_tol=1e-3,
            )

        else:
            # fast but unrealistic moves (bigger_moves = worse performance)
            self.blocking_position_move(
                max_steps=1000,
                constant_vel=None,
                j_pos_tol=1e-6,
                j_vel_tol=1e-3,
            )

    def move_linear(self, targ_pose, quick_mode=False):
        self.set_target_tcp_pose(targ_pose)
        self.apply_blocking_position_move(quick_mode=quick_mode)

    def move_linear_vel(self, targ_vels, blocking_steps=100):
        self.set_target_tcp_velocities(targ_vels)
        self.apply_blocking_velocity_move(blocking_steps=blocking_steps)

    def move_joints(self, targ_joint_angles, quick_mode=False):
        self.set_target_joint_positions(targ_joint_angles)
        self.apply_blocking_position_move(quick_mode=quick_mode)

    def move_joints_vel(self, targ_vels, blocking_steps=100):
        self.set_target_joint_velocities(targ_vels)
        self.apply_blocking_velocity_move(blocking_steps=blocking_steps)
