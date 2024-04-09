import numpy as np
from env import Env
import math
from stable_baselines3.common.env_checker import check_env
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

root_path = '/Users/yin/Documents/GitHub/robotics_pybullet_learn/UR5'
timestep = 1/240
seed = 1234
reset_arm_poses = [math.pi, -math.pi/2, -math.pi*5/9, -math.pi*4/9,
                               math.pi/2, 0]
reset_gripper_range = [0, 0.085]
visual_sensor_params = {
        'image_size': [128, 128],
        'dist': 1.0,
        'yaw': 90.0,
        'pitch': -25.0,
        'pos': [0.6, 0.0, 0.0525],
        'fov': 75.0,
        'near_val': 0.1,
        'far_val': 5.0,
        'show_vision': False
    }
robot_params = {
    "reset_arm_poses": reset_arm_poses,
    "reset_gripper_range": reset_gripper_range,
}
use_gui = True
env_kwargs_dict = {"show_gui": use_gui, "timestep": timestep, "robot_params": robot_params, "visual_sensor_params": visual_sensor_params}


vec_env = Env(use_gui, timestep, robot_params,visual_sensor_params)
obs,_ = vec_env.reset()
box_pos = (0.69, 0.2, 0.8)+ (math.pi/2,math.pi/2,0)+(0.085,)
obs_next, reward, done, truncated, info = vec_env.step(box_pos,control_method='end')
box_pos = (0.69,0,0.8)+ (math.pi/2,math.pi/2,0)+(0.085,)
obs_next, reward, done, truncated, info = vec_env.step(box_pos,control_method='end')


while True:
    vec_env.step_simulation()
    # time.sleep(timestep)
    q_key = ord("q")
    keys = vec_env._pb.getKeyboardEvents()
    if q_key in keys and keys[q_key] & vec_env._pb.KEY_WAS_TRIGGERED:
        exit()
# obs_next, reward, done, truncated, info = vec_env.step([math.pi, -math.pi/2, -math.pi*5/9, -math.pi*4/9, math.pi/2, math.pi/4, 0.085])
# obs_next1, reward1, done, truncated, info = vec_env.step([math.pi, -math.pi/2, -math.pi*5/9, -math.pi*4/9, math.pi/2, 0, 0.085])
