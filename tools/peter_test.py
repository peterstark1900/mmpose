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
    env_name = 'CartPole-v0'
    env = gym.make(env_name)
    env.seed(0)
    torch.manual_seed(0)
    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape
    print("state_dim: ", state_dim)
    print("action_dim: ", action_dim)

if __name__ == '__main__':
    main()