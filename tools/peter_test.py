import random
import gym
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.distributions import Normal
import matplotlib.pyplot as plt
import rl_utils

def main():
    # env_name = 'CartPole-v0'
    env_name = 'Pendulum-v1'
    env = gym.make(env_name)
    env.seed(0)
    torch.manual_seed(0)
    state = env.observation_space
    action = env.action_space
    print("state: ", state)
    print("action: ", action)
    # state_dim = env.observation_space.shape[0]
    # action_dim = env.action_space.n
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    print("state_dim: ", state_dim)
    print("action_dim: ", action_dim)
    action_bound = env.action_space.high[0]  # 动作最大值
    print("action_bound: ", action_bound)

if __name__ == '__main__':
    main()