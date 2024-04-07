import os
from UR5_arm import UR5Arm


class ArmEmbodiment:
    def __init__(self, pb, robot_arm_params={}):
        self._pb = pb

        if "tcp_link_name" in robot_arm_params:
            self.tcp_link_name = robot_arm_params["tcp_link_name"]
        else:
            self.tcp_link_name = "ee_link"

        # load the urdf file
        self.load_urdf()

        # instantiate a robot arm
        self.arm = UR5Arm(
            pb,
            embodiment_id=self.embodiment_id,
            tcp_link_id=self.tcp_link_id,
            link_name_to_index=self.link_name_to_index,
            joint_name_to_index=self.joint_name_to_index,
            rest_poses=robot_arm_params['rest_poses'],
        )

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

    def reset(self, reset_tcp_pose):
        """
        Reset the pose of the arm and sensor
        """
        self.arm.reset()

        # move to the initial position
        self.arm.move_linear(reset_tcp_pose, quick_mode=True)

    def full_reset(self):
        self.load_urdf()
        self.sensor.turn_off_collisions()
