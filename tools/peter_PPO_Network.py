import torch
import torch.nn.functional as F
import numpy as np
import rl_utils


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class PPO:
    ''' PPO算法,采用截断方式 '''
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs  # 一条序列的数据用来训练轮数
        self.eps = eps  # PPO中截断范围的参数
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        print("probs: ",probs)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        td_target = rewards + self.gamma * self.critic(next_states) * (1 -
                                                                       dones)
        td_delta = td_target - self.critic(states)
        advantage = rl_utils.compute_advantage(self.gamma, self.lmbda,
                                               td_delta.cpu()).to(self.device)
        old_log_probs = torch.log(self.actor(states).gather(1,
                                                            actions)).detach()

        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(states).gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps,
                                1 + self.eps) * advantage  # 截断
            actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数
            critic_loss = torch.mean(
                F.mse_loss(self.critic(states), td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
    def save_model(self, path):
        # torch.save(self.actor.state_dict(), path + 'actor.pth')
        # torch.save(self.critic.state_dict(), path + 'critic.pth')
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
        },path)

    def load_model(self, path):
        # self.actor.load_state_dict(torch.load(path + 'actor.pth'))
        # self.critic.load_state_dict(torch.load(path + 'critic.pth'))
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])

class PolicyNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNetContinuous, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
        # self.fc_std = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_log_std = torch.nn.Linear(hidden_dim, action_dim)
        self.log_std = torch.nn.Parameter(torch.zeros(action_dim) * 0.1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = torch.tanh(self.fc_mu(x))
        # std = F.softplus(self.fc_std(x))+ 1e-5  # 添加极小值防止数值问题
        # # 生成对数标准差（可训练参数）
        # log_std = self.fc_log_std(x)
        # std = torch.exp(log_std) + 1e-5
        # 使用固定可训练参数 + 缩放，替代全连接层
        log_std = torch.tanh(self.log_std) * 0.5  # 限制log_std在[-0.5, 0.5]
        std = torch.exp(log_std) + 1e-5
        
        return mu, std


class PPOContinuous:
    ''' 处理连续动作的PPO算法 '''
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device):
        self.actor = PolicyNetContinuous(state_dim, hidden_dim,
                                         action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device

    def take_action(self, state):
        # state = torch.tensor([state], dtype=torch.float).to(self.device)
        state = torch.as_tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
        
        # mu, sigma = self.actor(state)
        # action_dist = torch.distributions.Normal(mu, sigma)
        mu, std = self.actor(state)
        # 创建对角协方差矩阵（各维度独立）
        cov = torch.diag_embed(std)
        action_dist = torch.distributions.MultivariateNormal(mu, cov)
        
        action = action_dist.sample()
        # print(action)
        # return [action.item()]

        B_b = action[:, 1]  
        M_b = action[:, 0]  
        w_b = action[:, 2]  
        R_b = action[:, 3] 

        M_b = (M_b + 1) * 25.0                        # [0, 50]
        M_b = torch.clamp(M_b, 0.0, 50.0)

        B_b = (B_b + 1) * 12.5 + 5.0               # [-1,1] → [5,30]
        B_b = torch.clamp(B_b, 5.0, 30.0)

        w_b = (w_b + 1) * 12.5 + 5.0 + 1e-3          # [5,30]
        w_b = torch.clamp(w_b, 5.0 + 1e-3, 30.0)

        R_b = (R_b + 1) * 15.0 + 10.0 + 1e-3          # [-1,1] → [10,40]
        R_b = torch.clamp(R_b, 10.0 + 1e-3, 40.0)

        scaled_action = torch.stack([M_b, B_b, w_b, R_b], dim=1)
        return scaled_action.cpu().numpy()

    def update(self, transition_dict):
        states = torch.tensor(np.array(transition_dict['states']),
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(np.array(transition_dict['actions']),
                               dtype=torch.float).view(-1, 4).to(self.device)
        rewards = torch.tensor(np.array(transition_dict['rewards']),
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(transition_dict['next_states']),
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        rewards = (rewards + 8.0) / 8.0  # 和TRPO一样,对奖励进行修改,方便训练
        td_target = rewards + self.gamma * self.critic(next_states) * (1 -
                                                                       dones)
        td_delta = td_target - self.critic(states)
        advantage = rl_utils.compute_advantage(self.gamma, self.lmbda,
                                               td_delta.cpu()).to(self.device)
        mu, std = self.actor(states)
        action_dists = torch.distributions.Normal(mu.detach(), std.detach())
        # 动作是正态分布
        old_log_probs = action_dists.log_prob(actions)

        for _ in range(self.epochs):
            mu, std = self.actor(states)
            action_dists = torch.distributions.Normal(mu, std)
            log_probs = action_dists.log_prob(actions)

            # 添加标准差正则化项
            log_std = torch.log(std)
            std_penalty = 0.01 * torch.mean(log_std**2)  # 正则化系数0.01

            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
            # actor_loss = torch.mean(-torch.min(surr1, surr2))
            actor_loss = torch.mean(-torch.min(surr1, surr2)) + std_penalty
            critic_loss = torch.mean(
                F.mse_loss(self.critic(states), td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

    def save_model(self, path):
        # torch.save(self.actor.state_dict(), path + 'actor.pth')
        # torch.save(self.critic.state_dict(), path + 'critic.pth')
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
        },path)

    def load_model(self, path):
        # self.actor.load_state_dict(torch.load(path + 'actor.pth'))
        # self.critic.load_state_dict(torch.load(path + 'critic.pth'))
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])