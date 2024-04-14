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

sim_params = {"use_gui":False,
              'timestep':1/240,
              'control_type':'joint',
              'gripper_enable':False,
              'is_train':True,
              'distance_threshold':0.05,}
# env_kwargs_dict = {"sim_params":sim_params, "robot_params": robot_params, "visual_sensor_params": visual_sensor_params}
env = UR5Env(sim_params, robot_params,visual_sensor_params)

state_dim = env.observation_space['observation'].shape[0]+env.observation_space['desired_goal'].shape[0]+env.observation_space['achieved_goal'].shape[0]
action_dim = env.action_space.shape[0]



actor_lr = 3e-4
critic_lr = 3e-3
alpha_lr = 3e-4
num_episodes = 1000
hidden_dim = 256
gamma = 0.99999
sigma = 0.1
tau = 0.005  # 软更新参数
buffer_size = 100000
minimal_episodes = 10
n_train = 20
batch_size = 512
state_len = env.observation_space['observation'].shape[0]
achieved_goal_len = env.observation_space['achieved_goal'].shape[0]
target_entropy = -env.action_space.shape[0]
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "mps")


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

her_buffer = ReplayBuffer_Trajectory(capacity= buffer_size, 
                                     dis_threshold=sim_params['distance_threshold'], 
                                     use_her=True,
                                     batch_size=batch_size,
                                     state_len=state_len,
                                     achieved_goal_len=achieved_goal_len,)
agent = DDPG(state_dim, hidden_dim, action_dim,
                 actor_lr, critic_lr, sigma, tau, gamma, device)

return_list = []
for i in range(100):
    agent.lr_decay(i)
    with tqdm(total=int(num_episodes), desc='Iteration %d' % i) as pbar:
        for i_episode in range(num_episodes):
            episode_return = 0
            state,_ = env.reset()
            traj = Trajectory(state)
            done = False
            while not done:
                action = agent.take_action(state)
                state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_return += reward
                traj.store_step(action, state, reward, done)
            her_buffer.add_trajectory(traj)
            return_list.append(episode_return)
            # her_ratio = 1
            if her_buffer.size() >= minimal_episodes:
                her_buffer_len_ls = her_buffer.buffer[-1].length
                her_ratio = (her_buffer_len_ls-1)/env.time_limitation
                for _ in range(n_train):
                    transition_dict = her_buffer.sample(her_ratio)
                    agent.update(transition_dict)
                pbar.set_postfix({
                    # 'goal':
                    # '%r' % (env.goal),
                    'her_bf_min_len': her_buffer_len_ls,
                    'episode':
                        '%d' % (num_episodes* i + i_episode + 1),
                    "her dones":np.count_nonzero(transition_dict['dones']),
                    'return':
                    '%.3f' % np.array(return_list[-1]),
                    "lr": agent.actor_optimizer.param_groups[0][
                                "lr"],
                    "info:is success": info['is_success'],
                    "HER ratio":her_ratio
                })
            else:
                pbar.set_postfix({
                    'goal':
                    '%r' % (env.goal),
                    'episode':
                        '%d' % (num_episodes* i + i_episode + 1),
                    'return':
                    '%.3f' % np.array(return_list[-1]),
                    "lr": agent.actor_optimizer.param_groups[0][
                                "lr"],
                    "info:is success": info['is_success'],
                })
            pbar.update(1)
    torch.save(agent.actor.state_dict(), "./model/ddpg_her_ur5_%d.pkl" % i)
    sim_params['is_train'] = False
    test_env  = UR5Env(sim_params, robot_params,visual_sensor_params)
    evluation_policy(env=test_env, state_dim=agent.state_dim,
                     action_dim = agent.action_dim,
                     hidden_dim=agent.hidden_dim, 
                     device=agent.device,
                     model_num=i)
    test_env.close()
    del test_env
    sim_params['is_train'] = True

env.close()
del env
with open('ddpg_her_buffer.pkl', 'wb') as file:
    pickle.dump(her_buffer, file)
episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DDPG with HER on {}'.format('UR5'))
plt.show()
