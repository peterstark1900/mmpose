import random
import gym
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.distributions import Normal
import matplotlib.pyplot as plt
from rl_utils import ReplayBuffer, train_off_policy_agent, train_on_policy_agent, moving_average



from peter_env import Fish2DEnv
from peter_AC_Network import PolicyNet, ValueNet, ActorCritic
from peter_SAC_Network import SACContinuous, PolicyNetContinuous, QValueNetContinuous
from peter_detector import FishDetector

import time
import datetime

import threading
import queue

def env():
    # env_name = 'CartPole-v0'
    env_name = 'Pendulum-v1'
    env = gym.make(env_name)
    env.seed(0)
    torch.manual_seed(0)
    state = env.observation_space
    action = env.action_space
    print("state: ", state)
    print(type(state))
    print("action: ", action)
    print(type(action))
    reset_state = env.reset()
    print("reset_state: ", reset_state)
    print(type(reset_state))
    an_action = env.action_space.sample()
    print("an_action: ", an_action)
    print(type(an_action))
    # next_state, reward, done, _ = env.step(an_action)

    # state_dim = env.observation_space.shape[0]
    # action_dim = env.action_space.n
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    print("state_dim: ", state_dim)
    print("action_dim: ", action_dim)
    action_bound = env.action_space.high[0]  # 动作最大值
    print("action_bound: ", action_bound)

def network():
    state_dim = 7
    action_dim = 4


    random.seed(0)
    np.random.seed(0)

    torch.manual_seed(1)

    actor_lr = 1e-3
    critic_lr = 1e-2
    alpha_lr = 3e-4
    num_episodes = 100
    hidden_dim = 128
    gamma = 0.98
    tau = 0.005  # 软更新参数
    buffer_size = 100000
    minimal_size = 1000
    batch_size = 64
    target_entropy = -4

    max_episode_steps = 30
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "cpu")

    replay_buffer = ReplayBuffer(buffer_size)
    agent = SACContinuous(state_dim, hidden_dim, action_dim,
                      actor_lr, critic_lr, alpha_lr, target_entropy, tau,
                      gamma, device)
    # state = np.array([1403, \
    #                     340, \
    #                     960, \
    #                     540, \
    #                     1, \
    #                     1, \
    #                     1])
    # next_state = np.array([1000, \
    #                     430, \
    #                     690, \
    #                     450, \
    #                     2, \
    #                     2, \
    #                     2])
    state = np.expand_dims([1403, 340, 960, 540, 1, 1, 1], axis=0)
    next_state = np.expand_dims([1000, 430, 690, 450, 2, 2, 2], axis=0)
    print(state)
    print(type(state))
    action = agent.take_action(state)
    print(action)
    print(type(action))

    action_list = action.flatten().tolist()
    print(action_list)
    formatted_list = []
    print(action_list[0], action_list[1], action_list[2], action_list[3])
    for i in range(len(action_list)):
        rounded = round(action_list[i])
        if i == 1:
           formatted_list.append( f"{rounded + 30:02d}")
        else:
           formatted_list.append(f"{rounded:02d}")
        
    print(formatted_list)

    transition_dict = {'states': state, 'actions': np.array(action_list), 'next_states': next_state, 'rewards': [10], 'dones': [True]}
    agent.update(transition_dict)

    # actor_lr = 1e-3
    # critic_lr = 1e-2
    # num_episodes = 500
    # hidden_dim = 128
    # gamma = 0.98
    # lmbda = 0.95
    # epochs = 10
    # eps = 0.2
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    #     "cpu")

    # # env_name = 'CartPole-v0'
    # # env = gym.make(env_name)
    # # env.seed(0)
    # torch.manual_seed(0)
    # state_dim = env.observation_space.shape[0]
    # action_dim = env.action_space.n
    # agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda,
    #             epochs, eps, gamma, device)
def main():
    # env()
    network()

if __name__ == '__main__':
    main()