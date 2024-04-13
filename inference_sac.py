import numpy as np
from env import UR5Env
import math
from stable_baselines3.common.env_checker import check_env
import gymnasium as gym
import os

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env

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
sim_params = {"use_gui":False,
              'timestep':1/240,
              'control_type':'joint',
              'gripper_enable':False}

stats_path = os.path.join('./normalize_file/', "vec_normalize_sac.pkl")
sim_params['use_gui'] = True
sim_params['is_train'] = False
env_kwargs_dict = {"sim_params":sim_params, "robot_params": robot_params, "visual_sensor_params": visual_sensor_params}
vec_env = make_vec_env(UR5Env, n_envs=1, env_kwargs = env_kwargs_dict, seed=seed)
vec_env = VecNormalize.load(stats_path, vec_env)
#  do not update them at test time
vec_env.training = False
# reward normalization is not needed at test time
vec_env.norm_reward = False

# Load the agent
model = SAC.load("./model/ur5_robotiq140_sac",env=vec_env)
obs = vec_env.reset()
dones=False
while not dones:
    action, _states = model.predict(obs,deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")
vec_env.close()

