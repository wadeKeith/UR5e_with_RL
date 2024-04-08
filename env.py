import os
from ur5_robotiq import UR5Robotiq140


class Env:
    def __init__(self, pb, robot_params={}, root_path=""):
        self._pb = pb
        self.load_standard_environment(root_path)

        # instantiate a robot arm
        self.arm_gripper = UR5Robotiq140(
            self._pb,
            robot_params=robot_params,
        )
        self.boxID = self._pb.loadURDF("./assets/urdfs/skew-box-button.urdf",
                                [0.8, 0.0, 0.0],
                                # p.getQuaternionFromEuler([0, 1.5706453, 0]),
                                self._pb.getQuaternionFromEuler([0, 0, 0]),
                                useFixedBase=True,
                                flags=self._pb.URDF_MERGE_FIXED_LINKS | self._pb.URDF_USE_SELF_COLLISION)
        # For calculating the reward
        self.box_opened = False
        self.btn_pressed = False
        self.box_closed = False


    def close(self):
        if self._pb.isConnected():
            self._pb.disconnect()
    

    def reset(self):
        """
        Reset the pose of the arm and sensor
        """
        self.arm_gripper.reset()

        # move to the initial position
        # self.arm.move_linear(reset_tcp_pose, quick_mode=True)

    def load_standard_environment(self,root_path):
        """
        Load a standard environment with a plane and a table.
        """
        self._pb.loadURDF(
            "./assets/environment_objects/plane/plane.urdf",
            [0, 0, -0.625],
        )
        self._pb.loadURDF(
            "./assets/environment_objects/table/table.urdf",
            [0.50, 0.00, -0.625],
            [0.0, 0.0, 0.0, 1.0],
    )
