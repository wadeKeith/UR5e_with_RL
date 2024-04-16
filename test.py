import numpy as np
from env import UR5Env
import random
import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from DDPG_her import DDPG, ReplayBuffer_Trajectory, Trajectory,PolicyNet
import math
import pickle

def evluation_policy(env, state_dim, action_dim,hidden_dim, device, model_num):
    model = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
    model.load_state_dict(torch.load("./model/ddpg_her_ur5_%d.pkl" % model_num))
    model.eval()
    episode_return = 0
    state,_ = env.reset()
    done = False
    while not done:
        state = torch.tensor(state, dtype=torch.float).to(device)
        action = model(state).detach().cpu().numpy()
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        episode_return += reward
    print("Test rawrd of the model %d is %.3f and info: is_success: %r, goal is %r" % (model_num, episode_return, info['is_success'],env.goal))

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
# control type: joint, end
sim_params = {"use_gui":True,
              'timestep':1/240,
              'control_type':'end',
              'gripper_enable':False,
              'is_train':True,
              'distance_threshold':0.05,}
env = UR5Env(sim_params, robot_params,visual_sensor_params)

obs,_ = env.reset()
state, reward, terminated, truncated, info = env.step(np.array([1,1,1]))
t = 1
while True:
    env.step_simulation()
    t +=1
    
