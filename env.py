import os
import time
from ur5_robotiq import UR5Robotiq140
from utilize import connect_pybullet, set_debug_camera, Camera
import gymnasium
from gymnasium import spaces
import numpy as np
import math


class Env(gymnasium.Env):
    def __init__(self, show_gui,timestep, robot_params, visual_sensor_params):
        self.vis = show_gui
        self._pb = connect_pybullet(timestep, show_gui=self.vis)
        self.SIMULATION_STEP_DELAY = timestep
        self.load_standard_environment()

        # instantiate a robot arm
        self.arm_gripper = UR5Robotiq140(
            self._pb,
            robot_params=robot_params,
            use_gui = self.vis,
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
        rgb_obs_space = spaces.Box(low=0, high=255, shape=(visual_sensor_params['image_size'][0], visual_sensor_params['image_size'][1], 4), dtype=np.uint8)
        depth_obs_space = spaces.Box(low=0, high=1, shape=(visual_sensor_params['image_size'][0], visual_sensor_params['image_size'][1]), dtype=np.float32)
        seg_obs_space = spaces.Box(low=-1, high=255, shape=(visual_sensor_params['image_size'][0], visual_sensor_params['image_size'][1]), dtype=np.int32)
        positions_obs_space = spaces.Box(low=-3.14159265359, high=3.14159265359, shape=(self.arm_gripper.num_control_dofs,), dtype=np.float32)
        velocity_bound = np.array([3.16, 3.16, 3.16, 3.3, 3.3, 3.3, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
        assert velocity_bound.shape[0] == self.arm_gripper.num_control_dofs
        velocities_obs_space = spaces.Box(-velocity_bound, velocity_bound, dtype=np.float32)
        ee_pos_bound = np.array([8.5, 8.5, 8.5])
        ee_pos_obs_space = spaces.Box(-ee_pos_bound, ee_pos_bound, dtype=np.float32)
        self.observation_space = spaces.Dict({
            'rgb': rgb_obs_space,
            'depth': depth_obs_space,
            'seg': seg_obs_space,
            'positions': positions_obs_space,
            'velocities': velocities_obs_space,
            'ee_pos': ee_pos_obs_space
        })
        self.action_space = spaces.Box(low=np.array([-3.14159265359, -3.14159265359, -3.14159265359, -3.14159265359, -3.14159265359, -3.14159265359, 0]), 
                                       high=np.array([3.14159265359, 3.14159265359, 3.14159265359, 3.14159265359, 3.14159265359, 3.14159265359, 0.085]), 
                                       dtype=np.float32)

    

    def reset(self,seed=None):
        """
        Reset the pose of the arm and sensor
        """
        # np.random.seed(seed)
        self.arm_gripper.reset()
        self.reset_box()
        info = dict(box_opened=self.box_opened, btn_pressed=self.btn_pressed, box_closed=self.box_closed)
        return (self.get_observation(), info)

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
        time_reward = 0
        while True:
            obs = self.arm_gripper.get_joint_obs()
            self.step_simulation()
            time_reward+=1
            obs_next = self.arm_gripper.get_joint_obs()
            if np.linalg.norm(obs['positions'] - obs_next['positions'],ord=2)<1e-4:
                break
                

        reward, done = self.update_reward(time_reward)
        # done = True if reward == 1 else False
        info = dict(box_opened=self.box_opened, btn_pressed=self.btn_pressed, box_closed=self.box_closed)
        truncated = False
        # q_key = ord("q")
        # keys = self._pb.getKeyboardEvents()
        # if q_key in keys and keys[q_key] & self._pb.KEY_WAS_TRIGGERED:
        #     truncated = True
        # else:
        #     truncated = False
        return self.get_observation(), reward, done, truncated, info

    def update_reward(self,time_reward):
        done = False
        if len(self._pb.getClosestPoints(self.boxID, self.arm_gripper.embodiment_id, 0.01,1, self.arm_gripper.tcp_link_id))==0:
            box_ee_distance =  np.linalg.norm(np.array(self._pb.getLinkState(self.boxID,1,1,1)[0])-self.arm_gripper.get_joint_obs()['ee_pos'],ord=2)
            reward = math.exp(-box_ee_distance)
        else:
            box_ee_distance =  np.linalg.norm(np.array(self._pb.getLinkState(self.boxID,1,1,1)[0])-self.arm_gripper.get_joint_obs()['ee_pos'],ord=2)
            if self._pb.getJointState(self.boxID, 1)[0] <= 1.9:
                reward = math.exp(-box_ee_distance) + self._pb.getJointState(self.boxID, 1)[0]/3.14159265359
            else:
                done = True
                self.box_opened = True
                reward = math.exp(-box_ee_distance) + self._pb.getJointState(self.boxID, 1)[0]/3.14159265359
        # reward = math.exp(-) 1, self.arm_gripper.tcp_link_id
        # if not self.box_opened:
        #     if self._pb.getJointState(self.boxID, 1)[0] > 1.9:
        #         self.box_opened = True
        #         reward +=10
        #         print('Box opened!')
        # elif not self.btn_pressed:
        #     if self._pb.getJointState(self.boxID, 0)[0] < - 0.02:
        #         self.btn_pressed = True
        #         reward +=10
        #         print('Btn pressed!')
        # else:
        #     if self._pb.getJointState(self.boxID, 1)[0] < 0.1:
        #         print('Box closed!')
        #         self.box_closed = True
        #         reward +=10
        reward = reward+math.exp(-time_reward/300)
        return reward, done
    
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
        self._pb.resetJointState(self.boxID, 0, -1.1275702593849252e-18,0)
        self._pb.resetJointState(self.boxID, 1, 0,0 )
        # self._pb.setJointMotorControl2(self.boxID, 0, self._pb.POSITION_CONTROL, force=1)
        # self._pb.setJointMotorControl2(self.boxID, 1, self._pb.POSITION_CONTROL, force=1)

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
    def render(self, mode='human'):
        pass
    def seed(self, seed=None):
        pass
