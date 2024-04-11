import numpy as np
from env import UR5Env
import math
from stable_baselines3.common.env_checker import check_env
import gymnasium as gym
import os

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

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
control_type = 'joint'

stats_path = os.path.join('./normalize_file/', "vec_normalize_ppo.pkl")
use_gui = True
# env_kwargs_dict = {"show_gui": use_gui, "timestep": timestep, "robot_params": robot_params, "visual_sensor_params": visual_sensor_params}
vec_env = UR5Env(use_gui, timestep, robot_params,visual_sensor_params,control_type)
vec_env = make_vec_env(lambda:vec_env, seed=seed)
vec_env = VecNormalize.load(stats_path, vec_env)
#  do not update them at test time
vec_env.training = False
# reward normalization is not needed at test time
vec_env.norm_reward = False

# Load the agent
model = PPO.load("./model/ur5_robotiq140_ppo",env=vec_env)
# model = PPO.load(log_dir + "ppo_halfcheetah", env=vec_env)
# vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)
obs = vec_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")