import os
import time
from ur5_robotiq import UR5Robotiq140
from utilize import connect_pybullet, set_debug_camera, Camera


class Env:
    def __init__(self, show_gui,timestep, robot_params, visual_sensor_params):
        self.vis = show_gui
        self._pb = connect_pybullet(timestep, show_gui=self.vis)
        self.SIMULATION_STEP_DELAY = timestep
        self.load_standard_environment()

        # instantiate a robot arm
        self.arm_gripper = UR5Robotiq140(
            self._pb,
            robot_params=robot_params,
        )
        self.arm_gripper.step_simulation = self.step_simulation
        # self.arm_gripper.reset()
        self.boxID = self._pb.loadURDF("./assets/urdfs/skew-box-button.urdf",
                                [0.7, 0.0, 0.0],
                                # p.getQuaternionFromEuler([0, 1.5706453, 0]),
                                self._pb.getQuaternionFromEuler([0, 0, 0]),
                                useFixedBase=True,
                                flags=self._pb.URDF_MERGE_FIXED_LINKS | self._pb.URDF_USE_SELF_COLLISION)
        # For calculating the reward
        self.box_opened = False
        self.btn_pressed = False
        self.box_closed = False
        self.camera = Camera(self._pb, visual_sensor_params)
        if self.vis:
            set_debug_camera(self._pb, visual_sensor_params)
    

    def reset(self):
        """
        Reset the pose of the arm and sensor
        """
        self.arm_gripper.reset()
        self.reset_box()
        return self.get_observation()

        # move to the initial position
        # self.arm.move_linear(reset_tcp_pose, quick_mode=True)

    def step(self, action, control_method='joint'):
        """
        action: (x, y, z, roll, pitch, yaw, gripper_opening_length) for End Effector Position Control
                (a1, a2, a3, a4, a5, a6, a7, gripper_opening_length) for Joint Position Control
        control_method:  'end' for end effector position control
                         'joint' for joint position control
        """
        assert control_method in ('joint', 'end')
        self.arm_gripper.move_ee(action[:-1], control_method)
        self.arm_gripper.move_gripper(action[-1])
        for _ in range(120):  # Wait for a few steps
            self.step_simulation()

        reward = self.update_reward()
        done = True if reward == 1 else False
        info = dict(box_opened=self.box_opened, btn_pressed=self.btn_pressed, box_closed=self.box_closed)
        return self.get_observation(), reward, done, info

    def update_reward(self):
        reward = 0
        if not self.box_opened:
            if self._pb.getJointState(self.boxID, 1)[0] > 1.9:
                self.box_opened = True
                print('Box opened!')
        elif not self.btn_pressed:
            if self._pb.getJointState(self.boxID, 0)[0] < - 0.02:
                self.btn_pressed = True
                print('Btn pressed!')
        else:
            if self._pb.getJointState(self.boxID, 1)[0] < 0.1:
                print('Box closed!')
                self.box_closed = True
                reward = 1
        return reward
    
    def step_simulation(self):
        """
        Hook p.stepSimulation()
        """
        self._pb.stepSimulation()
        if self.vis:
            time.sleep(self.SIMULATION_STEP_DELAY)

    def load_standard_environment(self):
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
    def reset_box(self):
        self._pb.setJointMotorControl2(self.boxID, 0, self._pb.POSITION_CONTROL, force=1)
        self._pb.setJointMotorControl2(self.boxID, 1, self._pb.POSITION_CONTROL, force=1)

    def get_observation(self):
        obs = dict()
        if isinstance(self.camera, Camera):
            rgb, depth, seg = self.camera.shot()
            obs.update(dict(rgb=rgb, depth=depth, seg=seg))
        else:
            assert self.camera is None
        obs.update(self.arm_gripper.get_joint_obs())

        return obs
    def close(self):
        if self._pb.isConnected():
            self._pb.disconnect()
