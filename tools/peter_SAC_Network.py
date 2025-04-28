import random
import gym
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.distributions import Normal
import matplotlib.pyplot as plt
import rl_utils

class PolicyNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNetContinuous, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_std = torch.nn.Linear(hidden_dim, action_dim)
        # self.action_bound = action_bound

        self.fc3 = torch.nn.Linear(hidden_dim, action_dim)
        torch.nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)  # 添加均匀初始化
        torch.nn.init.constant_(self.fc3.bias, 0.1)  # 避免初始输出为0

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        std = F.softplus(self.fc_std(x))+1e-6  # 添加最小值保护
        dist = Normal(mu, std)
        normal_sample = dist.rsample()  # rsample()是重参数化采样
        log_prob = dist.log_prob(normal_sample)
        # action = torch.tanh(normal_sample)
        action = torch.tanh(normal_sample)
        log_prob = log_prob - torch.log(1 - torch.tanh(action).pow(2) + 1e-7)
        # action = (action + 1) * 0.5  # Scale [0, 1] for positive-only actions
        # # Create scaling matrix based on your action space:
        # # [M_b(0-50), w_b(0-30), B_b(-30-30), R_b(0-40)]
        # action = action * torch.tensor([50.0, 30.0, 30.0, 40.0]).to(x.device)
        # action = action - torch.tensor([0.0, 0.0, 30.0, 0.0]).to(x.device)  # Offset for B_b
        # # # 计算tanh_normal分布的对数概率密度
        # # log_prob = log_prob - torch.log(1 - torch.tanh(action).pow(2) + 1e-7)
        # # action = action * self.action_bound
        print("action: ", action)
        # 解包动作分量进行单独处理
        M_b = action[:, 0]  
        B_b = action[:, 1]  
        w_b = action[:, 2]  
        R_b = action[:, 3]  
        # 使用弹性索引代替固定维度索引
        # M_b = action[..., 0]  
        # B_b = action[..., 1]
        # w_b = action[..., 2]
        # R_b = action[..., 3]

        # 分量缩放和约束
        M_b = (M_b + 1) * 25.0                        # [0, 50]
        M_b = torch.clamp(M_b, 0.0, 50.0)
        
        # B_b = B_b * 30.0                              # [-30, 30]
        # B_b = torch.clamp(B_b, -30.0, 30.0)

        B_b = (B_b + 1) * 12.5 + 5.0               # [-1,1] → [5,30]
        B_b = torch.clamp(B_b, 5.0, 30.0)

        # w_b = (w_b + 1) * 15.0 + 1e-3                # (0, 30]
        # w_b = torch.clamp(w_b, 1e-3, 30.0)

        w_b = (w_b + 1) * 12.5 + 5.0 + 1e-3          # [5,30]
        w_b = torch.clamp(w_b, 5.0 + 1e-3, 30.0)
        
        
        
        # R_b = (R_b + 1) * 19.5+1.0 + 1e-3                # [1, 40]
        # R_b = torch.clamp(R_b, 1.0+1e-3, 40.0)

        R_b = (R_b + 1) * 15.0 + 10.0 + 1e-3          # [-1,1] → [10,40]
        R_b = torch.clamp(R_b, 10.0 + 1e-3, 40.0)

        # 重新组合动作分量
        scaled_action = torch.stack([M_b, B_b, w_b, R_b], dim=1)
        return scaled_action, log_prob


class QValueNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNetContinuous, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        # self.fc_out = torch.nn.Linear(hidden_dim, 1)
        self.fc_out = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x, a):
        # print("x: ", x)
        # print("a: ", a)
        # print('dim: ', x.shape, a.shape)
        cat = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)

class SACContinuous:
    ''' 处理连续动作的SAC算法 '''
    def __init__(self, state_dim, hidden_dim, action_dim,
                 actor_lr, critic_lr, alpha_lr, target_entropy, tau, gamma,
                 device):
        self.actor = PolicyNetContinuous(state_dim, hidden_dim, action_dim).to(device)  # 策略网络
        self.critic_1 = QValueNetContinuous(state_dim, hidden_dim,
                                            action_dim).to(device)  # 第一个Q网络
        self.critic_2 = QValueNetContinuous(state_dim, hidden_dim,
                                            action_dim).to(device)  # 第二个Q网络
        self.target_critic_1 = QValueNetContinuous(state_dim,
                                                   hidden_dim, action_dim).to(
                                                       device)  # 第一个目标Q网络
        self.target_critic_2 = QValueNetContinuous(state_dim,
                                                   hidden_dim, action_dim).to(
                                                       device)  # 第二个目标Q网络
        # 令目标Q网络的初始参数和Q网络一样
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(),
                                                   lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(),
                                                   lr=critic_lr)
        # 使用alpha的log值,可以使训练结果比较稳定
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        self.log_alpha.requires_grad = True  # 可以对alpha求梯度
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=alpha_lr)
        self.target_entropy = target_entropy  # 目标熵的大小
        self.gamma = gamma
        self.tau = tau
        self.device = device

    # def take_action(self, state):
    #     state = torch.tensor([state], dtype=torch.float).to(self.device)
    #     action = self.actor(state)[0]
    #     return [action.item()]
    def take_action(self, state):
        state = torch.as_tensor(np.array(state), 
                            dtype=torch.float32,
                            device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state)[0]
        return action.cpu().numpy()

    def calc_target(self, rewards, next_states, dones):  # 计算目标Q值
        next_actions, log_prob = self.actor(next_states)
        entropy = -log_prob
        q1_value = self.target_critic_1(next_states, next_actions)
        q2_value = self.target_critic_2(next_states, next_actions)
        next_value = torch.min(q1_value,
                               q2_value) + self.log_alpha.exp() * entropy
        td_target = rewards + self.gamma * next_value * (1 - dones)
        return td_target

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) +
                                    param.data * self.tau)

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'],
                               dtype=torch.float).view(-1, 4).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        # # 修正动作张量的维度处理
        # actions = torch.tensor(transition_dict['actions'],
        #                      dtype=torch.float).to(self.device)
        
        # # 确保状态张量的维度正确
        # states = torch.tensor(transition_dict['states'],
        #                      dtype=torch.float).unsqueeze(1).to(self.device)  # 添加批次维度
        # next_states = torch.tensor(transition_dict['next_states'],
        #                           dtype=torch.float).unsqueeze(1).to(self.device)
        
        # # 修正其他张量的维度
        # rewards = torch.tensor(transition_dict['rewards'],
        #                       dtype=torch.float).view(-1, 1).to(self.device)
        # dones = torch.tensor(transition_dict['dones'],
        #                     dtype=torch.float).view(-1, 1).to(self.device)
        # 和之前章节一样,对倒立摆环境的奖励进行重塑以便训练
        # rewards = (rewards + 8.0) / 8.0

        # 更新两个Q网络
        td_target = self.calc_target(rewards, next_states, dones)
        critic_1_loss = torch.mean(
            F.mse_loss(self.critic_1(states, actions), td_target.detach()))
        critic_2_loss = torch.mean(
            F.mse_loss(self.critic_2(states, actions), td_target.detach()))
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # 更新策略网络
        new_actions, log_prob = self.actor(states)
        entropy = -log_prob
        q1_value = self.critic_1(states, new_actions)
        q2_value = self.critic_2(states, new_actions)
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy -
                                torch.min(q1_value, q2_value))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 更新alpha值
        alpha_loss = torch.mean(
            (entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)

    def save_model(self, save_path):
        """保存所有网络参数"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic_1': self.critic_1.state_dict(),
            'critic_2': self.critic_2.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_1_optimizer': self.critic_1_optimizer.state_dict(),
            'critic_2_optimizer': self.critic_2_optimizer.state_dict(),
            'log_alpha': self.log_alpha,
            'log_alpha_optimizer': self.log_alpha_optimizer.state_dict()
        }, save_path)

    def load_model(self, load_path):
        """加载已保存的参数"""
        checkpoint = torch.load(load_path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic_1.load_state_dict(checkpoint['critic_1']) 
        self.critic_2.load_state_dict(checkpoint['critic_2'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_1_optimizer.load_state_dict(checkpoint['critic_1_optimizer'])
        self.critic_2_optimizer.load_state_dict(checkpoint['critic_2_optimizer'])
        self.log_alpha = checkpoint['log_alpha']
        self.log_alpha_optimizer.load_state_dict(checkpoint['log_alpha_optimizer'])