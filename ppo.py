import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import os
from torch.utils.data import DataLoader


def evluation_policy(env, hidden_dim, device, model_num,horizon,test_dataloader):
    state_dim = len(env.state_lower)+horizon*2  # 离散状态空间
    action_dim = 2  # 连续动作空间
    agent_test = PolicyNetContinuous(state_dim, hidden_dim, action_dim,horizon).to(device)
    agent_test.load_state_dict(torch.load("./model/ppo_continuous_%d.pkl" % model_num))
    agent_test.eval()

    for test_data_each in test_dataloader:
        state, DHEV = env.reset(test_data_each.numpy()[0])
        break
    done = False
    reward_ls = []
    num = 0
    while not done:
        state = torch.tensor([state], dtype=torch.float).to(device)
        action, _ = agent_test(state)
        action = action.view(-1, 2)
        action = action.clamp(-1.0, 1.0)
        action = action.cpu().detach().numpy().tolist()
        done_d = 0
        roll = 0
        d_reward_total = 0
        while not done_d:
            action_each = 1
            reward_d, done_d, info_d = DHEV.step(action_each)
            roll += 1
            d_reward_total += reward_d
        
        next_state, next_DHEV, _, done, info = env.step(action[0])
        state = next_state
        DHEV = next_DHEV
        reward_ls.append(d_reward_total)
        num += 1

    print("reward: ", np.mean(reward_ls), "num: ", num, "info: ", info)


class PolicyNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, horizon):
        super(PolicyNetContinuous, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim * (horizon + 1))
        self.fc_std = torch.nn.Linear(hidden_dim, action_dim * (horizon + 1))
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("mps")
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        mu = torch.tanh(self.fc_mu(x))
        std = F.softplus(self.fc_std(x))
        std = std + 1e-8 * torch.ones(size=std.shape).to(self.device)
        return mu, std


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, horizon):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = torch.nn.Linear(hidden_dim, horizon + 2)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return self.fc5(x)


class PPOContinuous:
    """处理连续动作的PPO算法"""

    def __init__(
        self,
        state_dim,
        hidden_dim,
        action_dim,
        actor_lr,
        critic_lr,
        lmbda,
        epochs,
        eps,
        gamma,
        device,
        entropy_coef,
        horizon,
    ):
        self.actor = PolicyNetContinuous(state_dim, hidden_dim, action_dim, horizon).to(
            device
        )
        self.critic = ValueNet(state_dim, hidden_dim, horizon).to(device)
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
        self.horizon = horizon

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        mu, sigma = self.actor(state)
        mu = mu.view(-1, 2)
        sigma = sigma.view(-1, 2)
        action_dist = torch.distributions.Normal(mu, sigma)
        action = action_dist.sample()
        action = action.clamp(-1.0, 1.0)
        # print(action)
        return action.cpu().numpy().tolist()

    def update(self, transition_dict):
        state = torch.tensor(transition_dict["state"], dtype=torch.float).to(
            self.device
        )
        actions = (
            torch.tensor(transition_dict["actions"], dtype=torch.float)
            .view(-1, 2)
            .to(self.device)
        )
        rewards = (
            torch.tensor(transition_dict["rewards"], dtype=torch.float)
            .view(-1, 1)
            .to(self.device)
        )
        dones = (
            torch.tensor(transition_dict["dones"], dtype=torch.float)
            .view(-1, 1)
            .to(self.device)
        )
        rewards = rewards  # 和TRPO一样,对奖励进行修改,方便训练
        td_target = rewards + self.gamma * self.critic(state).view(-1, 1)[1:1+len(rewards)] * (
            1 - dones
        )
        td_delta = td_target - self.critic(state).view(-1, 1)[:len(rewards)]
        advantage = compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(
            self.device
        )
        mu, std = self.actor(state)
        mu = mu.view(-1, 2)
        std = std.view(-1, 2)
        mu = mu[:len(rewards),:]
        std = std[:len(rewards),:]
        action_dists = torch.distributions.Normal(mu.detach(), std.detach())
        # 动作是正态分布
        old_log_probs = action_dists.log_prob(actions)

        updata_size = 1
        for _ in range(updata_size):
            mu, std = self.actor(state)
            mu = mu.view(-1, 2)
            std = std.view(-1, 2)
            mu = mu[:len(rewards),:]
            std = std[:len(rewards),:]
            action_dists = torch.distributions.Normal(mu, std)
            dist_entropy = action_dists.entropy().sum(1, keepdim=True)  # 计算熵
            log_probs = action_dists.log_prob(actions)
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
            actor_loss = (
                -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy
            )  # 计算actor的损失加入了熵
            critic_loss = torch.mean(
                F.mse_loss(self.critic(state).view(-1, 1)[:len(rewards)], td_target.detach())
            )
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.mean().backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

    def lr_decay(self, total_steps):
        lr_a_now = self.lr_a * (1 - total_steps / self.epochs)
        lr_c_now = self.lr_c * (1 - total_steps / self.epochs)
        for p in self.actor_optimizer.param_groups:
            p["lr"] = lr_a_now
        for p in self.critic_optimizer.param_groups:
            p["lr"] = lr_c_now


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    adv = torch.tensor(np.array(advantage_list), dtype=torch.float).view(-1, 1)
    # adv = ((adv - adv.mean()) / (adv.std() + 1e-5))
    return adv


def train_on_policy_agent(env, agent, Iteration,train_data,test_data):
    transition_dict_d = {
        "state": [],
        "actions": [],
        "rewards": [],
        "dones": [],
    }
    for i in range(Iteration):
        agent.lr_decay(i)
        np.random.seed(1)
        train_dataloader = DataLoader(train_data, batch_size=1, shuffle=True)
        test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True)
        with tqdm(total=len(train_dataloader), desc="Iteration %d" % i) as pbar:
            info_display = {
                "zeta": 0,
                "MG2,t,n": 0,
                "fc": 0,
                "reach": 0,
            }
            return_list = []
            # num_ls = []
            num_driving_cycles = 0
            for train_data_each in train_dataloader:
                
                state, DHEV = env.reset(train_data_each.numpy()[0])

                done = False
                num = 0
                episode_return = []
                while not done:
                    # print("state: ", state)
                    action = agent.take_action(state)
                    transition_dict_d["state"].append(state)
                    done_d = 0
                    roll = 0
                    d_reward_total = 0
                    while not done_d:
                        action_each = 1
                        reward_d, done_d, info_d = DHEV.step(action_each)
                        transition_dict_d["actions"].append(action[roll])
                        transition_dict_d["rewards"].append(reward_d)
                        transition_dict_d["dones"].append(done_d)
                        roll += 1
                        d_reward_total += reward_d
                    agent.update(transition_dict_d)
                    transition_dict_d = {
                        "state": [],
                        "actions": [],
                        "rewards": [],
                        "dones": [],
                    }
                    next_state, next_DHEV, _, done, info = env.step(action[0])
                    state = next_state
                    DHEV = next_DHEV
                    episode_return.append(d_reward_total)
                    num += 1

                return_list.append(np.mean(episode_return))
                # num_ls.append(num)

                if info == "MG1 speed is out of range":
                    info_display["nMG1"] += 1
                elif info == "MG2 torque is out of range":
                    info_display["TMG2"] += 1
                elif info == "motor torque n can't find":
                    info_display["MG2,t,n"] += 1
                elif info == "zeta is out of range":
                    info_display["zeta"] += 1
                elif info == "ICE fuel consumption can't find":
                    info_display["fc"] += 1
                elif info == "reach the end of the road":
                    info_display["reach"] += 1
                pbar.set_postfix(
                    {
                        "n_dc": "%d" % np.array(num_driving_cycles+1),
                        "rt": "%.3f" % np.array(return_list[-1]),
                        "ns": "%.3f" % np.array(num),
                        "lr": agent.actor_optimizer.param_groups[0][
                            "lr"
                        ],
                        "info": info_display,
                    }
                )
                pbar.update(1)
                num_driving_cycles += 1
        # saveclass(env.state_norm, "./model/text_file/state_norm_%d" % i)
        torch.save(agent.actor.state_dict(), "./model/ppo_continuous_%d.pkl" % i)
        evluation_policy(env, agent.hidden_dim, agent.device, i,agent.horizon,test_dataloader)
    return return_list
