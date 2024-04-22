import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import os
from torch.utils.data import DataLoader
from utilize import distance


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(np.array(advantage_list), dtype=torch.float)

def her_process(transition_dict,state_len,achieved_goal_len,distance_threshold):
    rewards = transition_dict['rewards']
    if rewards[-1] == 0:
        return transition_dict,'is_goal'
    else:
        initial_achieved_goal = transition_dict['states'][0][state_len:state_len+achieved_goal_len]
        achived_goal_all = np.array(transition_dict['next_states'])[:,state_len:state_len+achieved_goal_len]
        distances = np.linalg.norm(achived_goal_all-initial_achieved_goal,ord=2,axis=1)
        if np.all(distances<=distance_threshold):
            return transition_dict,'cant her'
        else:
            is_goal_np = distances<=distance_threshold
            step_goal = np.random.choice(np.where(is_goal_np==False)[0])
            goal = transition_dict['next_states'][step_goal][state_len:state_len+achieved_goal_len]
            new_transition_dict = {'states':[],'actions':[],'next_states':[],'rewards':[],'dones':[]}
            done = False
            i = 0
            while not done:
                state = np.concatenate([transition_dict['states'][i][:state_len+achieved_goal_len],goal])
                next_state = np.concatenate([transition_dict['next_states'][i][:state_len+achieved_goal_len],goal])
                reward = -1 if distance(next_state[state_len:state_len+achieved_goal_len],goal)>distance_threshold else 0
                done = reward==0
                new_transition_dict['states'].append(state)
                new_transition_dict['actions'].append(transition_dict['actions'][i])
                new_transition_dict['next_states'].append(next_state)
                new_transition_dict['rewards'].append(reward)
                new_transition_dict['dones'].append(done)
                i+=1
            return new_transition_dict,'her success'




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


class PPOContinuous:
    """处理连续动作的PPO算法"""

    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device, entropy_coef):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(
            device
        )
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.lr_a = actor_lr
        self.lr_c = critic_lr
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, eps=1e-5
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, eps=1e-5
        )
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device
        self.entropy_coef = entropy_coef
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.action_dim = action_dim

    def take_action(self, state):
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        mu, sigma = self.actor(state)
        action_dist = torch.distributions.Normal(mu, sigma)
        action = action_dist.sample()
        action = action.clamp(-1.0, 1.0)
        return action.detach().cpu().numpy()

    def update(self, transition_dict):
        states = torch.tensor(np.array(transition_dict['states']),
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(np.array(transition_dict['actions']),
                               dtype=torch.float).to(self.device)
        rewards = torch.tensor(np.array(transition_dict['rewards']),
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(transition_dict['next_states']),
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(np.array(transition_dict['dones']),
                             dtype=torch.float).view(-1, 1).to(self.device)
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        advantage = compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(
            self.device
        )
        mu, std = self.actor(states)
        action_dists = torch.distributions.Normal(mu.detach(), std.detach())
        # 动作是正态分布
        old_log_probs = action_dists.log_prob(actions)

        updata_size = 8
        for _ in range(updata_size):
            mu, std = self.actor(states)
            action_dists = torch.distributions.Normal(mu, std)
            dist_entropy = action_dists.entropy() # 计算熵
            log_probs = action_dists.log_prob(actions)
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
            actor_loss = (
                -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy
            )  # 计算actor的损失加入了熵
            critic_loss = torch.mean(
                F.mse_loss(self.critic(states), td_target.detach())
            )
            self.actor_optimizer.zero_grad()
            actor_loss.mean().backward()
            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

    def lr_decay(self, total_steps):
        lr_a_now = self.lr_a * (1 - total_steps / self.epochs)
        lr_c_now = self.lr_c * (1 - total_steps / self.epochs)
        for p in self.actor_optimizer.param_groups:
            p["lr"] = lr_a_now
        for p in self.critic_optimizer.param_groups:
            p["lr"] = lr_c_now





