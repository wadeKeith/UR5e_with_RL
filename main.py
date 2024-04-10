import numpy as np
from env import UR5Env
import math
from stable_baselines3.common.env_checker import check_env
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import time

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
use_gui = False
control_type = 'joint'
env_kwargs_dict = {"show_gui": use_gui, "timestep": timestep, "robot_params": robot_params, "visual_sensor_params": visual_sensor_params, "control_type": control_type}



# vec_env = UR5Env(use_gui, timestep, robot_params,visual_sensor_params,control_type)
# check_env(vec_env)
# obs, info = env.reset(seed=seed)
# vec_env = UR5Env(use_gui, timestep, robot_params,visual_sensor_params,control_type)
# obs,_ = vec_env.reset()
# while True:
#     # vec_env.step_simulation()
#     time.sleep(timestep)
#     q_key = ord("q")
#     keys = vec_env._pb.getKeyboardEvents()
#     if q_key in keys and keys[q_key] & vec_env._pb.KEY_WAS_TRIGGERED:
#         exit()

# obs_next, reward, done, truncated, info = vec_env.step([math.pi, -math.pi/2, -math.pi*5/9, -math.pi*4/9, math.pi/2, math.pi/4, 0.085])
# obs_next1, reward1, done, truncated, info = vec_env.step(np.array([1,1,1,1,1,1,-1]))

# vec_env = make_vec_env(lambda:vec_env, n_envs=16, seed=seed)
vec_env = make_vec_env(UR5Env, n_envs=16, env_kwargs = env_kwargs_dict, seed=seed)
# vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

model = PPO("MultiInputPolicy",vec_env, 
            learning_rate = 1e-7,
            n_steps=4,
            batch_size = 64,
            # n_epochs = 100,
            gamma = 0.99,
            normalize_advantage=True,
            ent_coef = 0.01,
            vf_coef = 0.5,
            max_grad_norm = 0.5,
            stats_window_size = 10,
            tensorboard_log = './logs',
            seed = seed,
            verbose=1,
            device='cuda')
model.learn(total_timesteps=25000, 
            log_interval=100, 
            # tb_log_name="ur5_robotiq140_ppo",
            progress_bar=True)
model.save("./model/ur5_robotiq140_ppo")

del model # remove to demonstrate saving and loading
vec_env.close()

model = PPO.load("./model/ur5_robotiq140_ppo")

use_gui = True
# env_kwargs_dict = {"show_gui": use_gui, "timestep": timestep, "robot_params": robot_params, "visual_sensor_params": visual_sensor_params}
vec_env = UR5Env(use_gui, timestep, robot_params,visual_sensor_params,control_type)
vec_env = make_vec_env(lambda:vec_env, seed=seed)
# vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)
obs = vec_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")



# while True:
#     # vec_env.step_simulation()
#     time.sleep(timestep)
#     q_key = ord("q")
#     keys = vec_env._pb.getKeyboardEvents()
#     if q_key in keys and keys[q_key] & vec_env._pb.KEY_WAS_TRIGGERED:
#         exit()

# # 断开连接
# env.close()