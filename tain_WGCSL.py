import numpy as np
from pick_place_env import PickPlace_UR5Env
import random
import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from WGCSL import WGCSL, ReplayBuffer_Trajectory, Trajectory,PolicyNet
import math
import pickle
import rl_utils
import collections

def evluation_policy(env, state_dim, action_dim,hidden_dim, device, model_num):
    model = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
    model.load_state_dict(torch.load("./model/wgcsl_her_ur5_pick_%d.pkl" % model_num))
    model.eval()
    episode_return = 0
    state,_,_ = env.reset()
    done = False
    while not done:
        state = torch.tensor(state, dtype=torch.float).to(device)
        mu,_ = model(state)
        action = mu.detach().cpu().numpy()
        state, reward, terminated, truncated, info,_ = env.step(action)
        done = terminated or truncated
        episode_return += reward
    print("Test rawrd of the model %d is %.3f and info: is_success: %r, goal is %r" % (model_num, episode_return, info['is_success'],env.goal))



seed = 3407
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
reset_arm_poses = [0, -math.pi/2, math.pi*4/9, -math.pi*4/9,
                            -math.pi/2, 0]
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
              'gripper_enable':True,
              'is_train':True,
              'distance_threshold':0.05,}
# env_kwargs_dict = {"sim_params":sim_params, "robot_params": robot_params, "visual_sensor_params": visual_sensor_params}

use_expert_data = False

env = PickPlace_UR5Env(sim_params, robot_params,visual_sensor_params)

state_dim = env.observation_space['observation'].shape[0]+env.observation_space['desired_goal'].shape[0]+env.observation_space['achieved_goal'].shape[0]
action_dim = env.action_space.shape[0]



actor_lr = 1e-3
critic_lr = 1e-3
num_episodes = 100
hidden_dim = 128
gamma = 0.99999
lmbda = 0.95
buffer_size = 100000
minimal_episodes = 5
n_train = 5
baw_delta = 0.05
geaw_M = 10
epochs = 100
batch_size = 512
B_capacity = n_train*num_episodes*epochs*2
state_len = env.observation_space['observation'].shape[0]
achieved_goal_len = env.observation_space['achieved_goal'].shape[0]
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "mps")




if use_expert_data:
    with open('ur5_pickplace_10000_expert_data_WGCSL.pkl', 'rb') as f:
    # 读取并反序列化数据
        her_buffer = pickle.load(f)
    f.close()
else:
    her_buffer = ReplayBuffer_Trajectory(capacity= buffer_size, 
                                        dis_threshold=sim_params['distance_threshold'], 
                                        use_her=True,
                                        batch_size=batch_size,
                                        state_len=state_len,
                                        achieved_goal_len=achieved_goal_len,)
    
agent = WGCSL(state_dim, hidden_dim, action_dim,
                 actor_lr, critic_lr, lmbda, gamma, baw_delta, geaw_M, epochs, device)

load_agent = False 
agent_num = 52
if load_agent:
    agent.actor.load_state_dict(torch.load("./model/wgcsl_her_ur5_pick_%d.pkl" % agent_num))
B_buffer = collections.deque(maxlen=B_capacity)


return_list = []
for i in range(100):
    agent.lr_decay(i)
    agent.percentile_num_update(i)
    with tqdm(total=int(num_episodes), desc='Iteration %d' % i) as pbar:
        success_count = 0
        for i_episode in range(num_episodes):
            episode_return = 0
            state,_,_ = env.reset()
            traj = Trajectory(state.copy())
            done = False
            while not done:
                with torch.no_grad():
                    action = agent.take_action(state)
                state, reward, terminated, truncated, info,_ = env.step(action)
                done = terminated or truncated
                episode_return += reward
                traj.store_step(action.copy(), state.copy(), reward, done)
            her_buffer.add_trajectory(traj)
            return_list.append(episode_return)
            if info['is_success'] == True:
                success_count+=1
                # her_buffer.add_trajectory(traj)
            if her_buffer.size() >= minimal_episodes:
                # her_buffer_len_ls = her_buffer.buffer[-1].length
                # her_buffer_minlen_ls = [her_buffer.buffer[i].length for i in range(her_buffer.size())]
                # her_ratio = (her_buffer_len_ls-1)/env.time_limitation
                her_ratio = 1
                for _ in range(n_train):
                    transition_dict = her_buffer.sample(her_ratio)
                    agent.update(transition_dict,B_buffer)
                # B_buffer_len = [len(B_buffer[i]) for i in range(len(B_buffer))]
                pbar.set_postfix({
                    # 'goal':
                    # '%r' % (env.goal),
                    # 'her_bf_min_len': min(her_buffer_minlen_ls),
                    # 'B_buffer_len':sum(B_buffer_len),
                    'percentile_num': '%.3f' % agent.percentile_num,
                    'her_size':her_buffer.size(),
                    'episode':
                        '%d' % (num_episodes* i + i_episode + 1),
                    "her dones":np.count_nonzero(transition_dict['dones']),
                    'return':
                    '%.3f' % np.mean(return_list[:]),
                    "lr": agent.actor_optimizer.param_groups[0][
                                "lr"],
                    "is success count": success_count,
                    # "HER ratio":her_ratio
                })
            else:
                pbar.set_postfix({
                    'goal':
                    '%r' % (env.goal),
                    'percentile_num': '%.3f' % agent.percentile_num,
                    'episode':
                        '%d' % (num_episodes* i + i_episode + 1),
                    'return':
                    '%.3f' % np.mean(return_list[:]),
                    "lr": agent.actor_optimizer.param_groups[0][
                                "lr"],
                    "is success count": success_count,
                })
            pbar.update(1)
    torch.save(agent.actor.state_dict(), "./model/wgcsl_her_ur5_pick_%d.pkl" % i)
    sim_params['is_train'] = False
    # sim_params['use_gui'] = True
    test_env  = PickPlace_UR5Env(sim_params, robot_params,visual_sensor_params)
    evluation_policy(env=test_env, state_dim=agent.state_dim,
                     action_dim = agent.action_dim,
                     hidden_dim=agent.hidden_dim, 
                     device=agent.device,
                     model_num=i)
    test_env.close()
    del test_env
    sim_params['is_train'] = True
    # sim_params['use_gui'] = False

env.close()
del env
with open('wgcsl_her_buffer_pickplace_all.pkl', 'wb') as file:
    pickle.dump(her_buffer, file)
episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('WGCSL with HER on {}'.format('UR5'))
plt.show()

mv_return = rl_utils.moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('WGCSL on {}'.format('UR5'))
plt.show()