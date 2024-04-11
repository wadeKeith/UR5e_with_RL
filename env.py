import os
import time
from ur5_robotiq import UR5Robotiq140
from utilize import connect_pybullet, set_debug_camera, Camera, distance
import gymnasium
from gymnasium import spaces
import numpy as np
import math


class UR5Env(gymnasium.Env):
    def __init__(self, show_gui,timestep, robot_params, visual_sensor_params,control_type='joint'):
        super().__init__()
        self.vis = show_gui
        self._pb = connect_pybullet(timestep, show_gui=self.vis)
        self.SIMULATION_STEP_DELAY = timestep
        self.control_type = control_type
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
        velocity_bound = np.array([3.16, 3.16, 3.16, 3.3, 3.3, 3.3, 2.1, 2.1, 2.1, 2.1, 2.1, 2.1])
        assert velocity_bound.shape[0] == self.arm_gripper.num_control_dofs
        velocities_obs_space = spaces.Box(-velocity_bound, velocity_bound, dtype=np.float32)
        ee_pos_bound = np.array([8.5, 8.5, 8.5])
        ee_pos_obs_space = spaces.Box(-ee_pos_bound, ee_pos_bound, dtype=np.float32)
        # self.observation_space = spaces.Dict({
        #     'rgb': rgb_obs_space,
        #     'depth': depth_obs_space,
        #     'seg': seg_obs_space,
        #     'positions': positions_obs_space,
        #     'velocities': velocities_obs_space,
        #     'finger_pos': ee_pos_obs_space
        # })
        self.observation_space = spaces.Dict({
            'positions_old': positions_obs_space,
            'velocities_old': velocities_obs_space,
            'finger_pos_old': ee_pos_obs_space,
            'positions': positions_obs_space,
            'velocities': velocities_obs_space,
            'finger_pos': ee_pos_obs_space
        })
        n_action = 4 if self.control_type == "end" else 7  # control (x, y z) if "ee", else, control the 7 joints
        self.action_space = spaces.Box(low=-1, high=1, shape=(n_action,),dtype=np.float32)

        self.time = None
        self.step_limit = 30000
        self.handle_pos = np.array([0.645, 1.4456028966473391e-18, 0.165])

    

    def reset(self,seed=None):
        """
        Reset the pose of the arm and sensor
        """
        # np.random.seed(seed)
        self.arm_gripper.reset()
        self.reset_box()
        self.time = 0
        info = dict(box_state='initial')
        # self._pb.addUserDebugPoints(pointPositions = [[0.48, -0.17256, 0.186809]], pointColorsRGB = [[255, 0, 0]], pointSize= 30, lifeTime= 0)
        # self._pb.addUserDebugPoints(pointPositions = [[0.645, 1.4456028966473391e-18, 0.165]], pointColorsRGB = [[255, 0, 0]], pointSize= 30, lifeTime= 0)
        obs = self.get_observation('old')
        obs.update(self.get_observation('now'))
        return (obs, info)

    def step(self, action):
        """
        action: (x, y, z, roll, pitch, yaw, gripper_opening_length) for End Effector Position Control
                (a1, a2, a3, a4, a5, a6, a7, gripper_opening_length) for Joint Position Control
        control_method:  'end' for end effector position control
                         'joint' for joint position control
        """
        obs = self.get_observation('old')
        truncated = False
        assert self.control_type in ('joint', 'end')
        self.arm_gripper.move_ee(action[:-1], self.control_type)
        self.arm_gripper.move_gripper(action[-1])
        self.step_simulation()
        reward, terminated, info_r = self.update_reward()
        # done = True if reward == 1 else False
        info = dict(box_state=info_r)
        obs.update(self.get_observation('now'))
        # self.time =+ 1
        # if self.time > self.step_limit:
        #     truncated = True
        return obs, reward, terminated, truncated, info

    def update_reward(self):
        terminated = False
        handle_finger_distance =  distance(np.array(self._pb.getLinkState(self.arm_gripper.embodiment_id, self.arm_gripper.left_finger_pad_id)[0]),self.handle_pos)
        if handle_finger_distance>0.05:
            reward = math.exp(-handle_finger_distance*100)
            info = 'far from box'
        else:
            rot_box =  self._pb.getJointState(self.boxID, 1)[0]
            terminated = True
            if rot_box <= 1.9:
                reward = math.exp(-handle_finger_distance*100) + rot_box/3.14159265359
                info = 'close to box'
            else:
                self.box_opened = True
                reward = math.exp(-handle_finger_distance*100) + rot_box/3.14159265359
                info = 'open box'
        return reward, terminated, info
    
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


    def get_observation(self,flag):
        obs = dict()
        # if isinstance(self.camera, Camera):
        #     rgb, depth, seg = self.camera.shot()
        #     obs.update(dict(rgb=rgb, depth=depth, seg=seg))
        # else:
        #     assert self.camera is None
        obs.update(self.arm_gripper.get_joint_obs(flag))

        return obs

    def close(self):
        if self._pb.isConnected():
            self._pb.disconnect()
    def render(self, mode='human'):
        pass
    def seed(self, seed=None):
        pass
