import torch
import torch.nn.functional as F
import numpy as np
import random
from tqdm import tqdm
import collections
import matplotlib.pyplot as plt
from utilize import distance


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)

class Trajectory:
    ''' 用来记录一条完整轨迹 '''
    def __init__(self, init_state):
        self.states = [init_state]
        self.actions = []
        self.rewards = []
        self.dones = []
        self.length = 0

    def store_step(self, action, state, reward, done):
        self.actions.append(action)
        self.states.append(state)
        self.rewards.append(reward)
        self.dones.append(done)
        self.length += 1


class ReplayBuffer_Trajectory:
    ''' 存储轨迹的经验回放池 '''
    def __init__(self, capacity, dis_threshold, use_her, batch_size,state_len,achieved_goal_len):
        self.buffer = collections.deque(maxlen=capacity)
        self.dis_threshold = dis_threshold
        self.use_her = use_her
        self.batch_size = batch_size
        self.state_len = state_len
        self.achieved_goal_len = achieved_goal_len
    def add_trajectory(self, trajectory):
        self.buffer.append(trajectory)

    def size(self):
        return len(self.buffer)

    def sample(self,her_ratio):
        batch = dict(states=[],
                     actions=[],
                     next_states=[],
                     rewards=[],
                     dones=[])
        for _ in range(self.batch_size):
            traj = random.sample(self.buffer, 1)[0]
            step_state = np.random.randint(traj.length)
            state = traj.states[step_state]
            next_state = traj.states[step_state + 1]
            action = traj.actions[step_state]
            reward = traj.rewards[step_state]
            done = traj.dones[step_state]


            if self.use_her and sum(traj.rewards) < 1:
                step_goal = np.random.randint(step_state + 1, traj.length + 1)
                goal = traj.states[step_goal][self.state_len:self.state_len+self.achieved_goal_len].copy()   # 使用HER算法的future方案设置目标
                dis = distance(next_state[self.state_len:self.state_len+self.achieved_goal_len], goal)
                reward = 0 if dis > self.dis_threshold else 1
                done = False if dis > self.dis_threshold else True
                state = np.hstack((state[:self.state_len+self.achieved_goal_len], goal)).copy() 
                next_state = np.hstack((next_state[:self.state_len+self.achieved_goal_len], goal)).copy() 

            batch['states'].append(state.copy())
            batch['next_states'].append(next_state.copy())
            batch['actions'].append(action.copy())
            batch['rewards'].append(reward)
            batch['dones'].append(done)

        batch['states'] = np.array(batch['states'])
        batch['next_states'] = np.array(batch['next_states'])
        batch['actions'] = np.array(batch['actions'])
        return batch


# class PolicyNet(torch.nn.Module):
#     def __init__(self, state_dim, hidden_dim, action_dim):
#         super(PolicyNet, self).__init__()
#         self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
#         self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
#         self.fc3 = torch.nn.Linear(hidden_dim, action_dim)

#     def forward(self, x):
#         x = F.relu(self.fc2(F.relu(self.fc1(x))))
#         return torch.tanh(self.fc3(x))
class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_std = torch.nn.Linear(hidden_dim, action_dim)
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("mps")
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = torch.tanh(self.fc_mu(x))
        std = F.softplus(self.fc_std(x))
        std = std + 1e-8 * torch.ones(size=std.shape).to(self.device)
        return mu, std

class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class WGCSL:
    ''' DDPG算法 '''
    def __init__(self, state_dim, hidden_dim, action_dim,
                 actor_lr, critic_lr, lmbda, gamma, device):
        self.action_dim = action_dim
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda  # 高斯噪声的标准差,均值直接设为0
        self.device = device
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.percentile_num = 0
        self.baw_delta = 0.05
        self.geaw_M = 10
        self.epochs = 100
        self.lr_a = actor_lr
        self.lr_c = critic_lr

    def take_action(self, state):
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        mu, sigma = self.actor(state)
        action_dist = torch.distributions.Normal(mu, sigma)
        action = action_dist.sample()
        action = action.clamp(-1.0, 1.0)
        return action.detach().cpu().numpy()

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) +
                                    param.data * self.tau)

    def update(self, transition_dict,B_buffer):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'],
                               dtype=torch.float).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        # action_next,_ = self.actor(next_states)
        # next_q_values = self.target_critic(next_states, action_next)
        # q_targets = rewards + self.gamma * next_q_values * (1 - dones)
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        # MSE损失函数
        critic_loss = torch.mean(
            F.mse_loss(self.critic(states), td_target))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        advantage = compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)
        B_buffer.append(advantage.detach().cpu().numpy().copy())
        all_advantage_list = []
        for i in range(len(B_buffer)):
            all_advantage_list += list(B_buffer[i].flatten())
        all_advantage_np = np.array(all_advantage_list)
        A_threshold = np.percentile(all_advantage_np,self.percentile_num)
        baw = self.BAW_compute(advantage.detach().cpu().numpy().copy(), A_threshold)
        geaw = torch.clip(torch.exp(advantage),0,self.geaw_M)

        # 策略网络就是为了使Q值最大化
        actor_loss = -torch.mean(self.critic(states, self.actor(states)))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.actor, self.target_actor)  # 软更新策略网络
        self.soft_update(self.critic, self.target_critic)  # 软更新价值网络

    def lr_decay(self, total_steps):
        lr_a_now = self.lr_a * (1 - total_steps / self.epochs)
        lr_c_now = self.lr_c * (1 - total_steps / self.epochs)
        for p in self.actor_optimizer.param_groups:
            p["lr"] = lr_a_now
        for p in self.critic_optimizer.param_groups:
            p["lr"] = lr_c_now
    def BAW_compute(self, advantage:np.ndarray, A_threshold):
        baw = np.zeros_like(advantage)
        for i in range(advantage.shape[0]):
            if advantage[i] >=A_threshold:
                baw[i] = 1
            else:
                baw[i] = self.baw_delta
        return torch.tensor(baw)