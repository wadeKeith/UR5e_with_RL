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
    # env.goal = env.handle_pos+np([0.1,0.1,0.05])
    done = False
    while not done:
        state = torch.tensor(state, dtype=torch.float).to(device)
        action = model(state).detach().cpu().numpy()
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        episode_return += reward
    print("Test rawrd of the model %d is %.3f and info: is_success: %r, goal is %r" % (model_num, episode_return, info['is_success'],env.goal))



seed = 3407
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
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
sim_params = {"use_gui":False,
              'timestep':1/240,
              'control_type':'end',
              'gripper_enable':False,
              'is_train':True,
              'distance_threshold':0.05,}
# env_kwargs_dict = {"sim_params":sim_params, "robot_params": robot_params, "visual_sensor_params": visual_sensor_params}



env = UR5Env(sim_params, robot_params,visual_sensor_params)

state_dim = env.observation_space['observation'].shape[0]+env.observation_space['desired_goal'].shape[0]+env.observation_space['achieved_goal'].shape[0]
action_dim = env.action_space.shape[0]




state_len = env.observation_space['observation'].shape[0]
achieved_goal_len = env.observation_space['achieved_goal'].shape[0]
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "mps")




hidden_dim = 128
    


sim_params['is_train'] = False
sim_params['use_gui'] = True
test_env  = UR5Env(sim_params, robot_params,visual_sensor_params)
# test_env.goal = test_env.handle_pos+np([0.1,0.1,0.05])
evluation_policy(env=test_env, state_dim=12,
                    action_dim = 3,
                    hidden_dim=hidden_dim, 
                    device=device,
                    model_num=57)
test_env.close()
del test_env

