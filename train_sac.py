import numpy as np
from env import UR5Env
import math
from stable_baselines3.common.env_checker import check_env
import gymnasium as gym
from sb3_contrib import TQC
from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import time
import os

from utilize import linear_schedule

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
              'gripper_enable':False,
              'is_train':True}
env_kwargs_dict = {"sim_params":sim_params, "robot_params": robot_params, "visual_sensor_params": visual_sensor_params}


# vec_env = UR5Env(sim_params, robot_params,visual_sensor_params)
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
vec_env = make_vec_env(UR5Env, n_envs=1, env_kwargs = env_kwargs_dict, seed=seed)
# vec_env.close()
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)
# exit()
model = SAC("MultiInputPolicy",vec_env, 
            learning_rate = linear_schedule(1e-4),
            replay_buffer_class= HerReplayBuffer,
            replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy="future",copy_info_dict=True),
            buffer_size = 1000000,
            learning_starts = 10,
            batch_size = 256,
            tau = 0.005,
            gamma = 0.99,
            train_freq = (1, "episode"), #(2, "episode"), (5, "step")
            tensorboard_log = './logs',
            seed = seed,
            # optimize_memory_usage = True,
            gradient_steps=1,
            verbose=1,
            device='cuda')
model.learn(total_timesteps=500000, 
            log_interval=10,
            tb_log_name="ur5_robotiq140_sac",
            progress_bar=False)
model.save("./model/ur5_robotiq140_sac")
stats_path = os.path.join('./normalize_file/', "vec_normalize_sac.pkl")
vec_env.save(stats_path)

vec_env.close()
del model ,vec_env# remove to demonstrate saving and loading



sim_params['use_gui'] = True
sim_params['is_train'] = False
# env_kwargs_dict = {"sim_params":sim_params, "robot_params": robot_params, "visual_sensor_params": visual_sensor_params}
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
exit()

# while True:
#     # vec_env.step_simulation()
#     time.sleep(timestep)
#     q_key = ord("q")
#     keys = vec_env._pb.getKeyboardEvents()
#     if q_key in keys and keys[q_key] & vec_env._pb.KEY_WAS_TRIGGERED:
#         exit()

# # 断开连接
# env.close()