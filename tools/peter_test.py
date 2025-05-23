import random
import gym
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.distributions import Normal
import matplotlib.pyplot as plt
import rl_utils
from rl_utils import ReplayBuffer, train_off_policy_agent, train_on_policy_agent, moving_average



# from peter_env import Fish2DEnv
from peter_AC_Network import PolicyNet, ValueNet, ActorCritic
from peter_SAC_Network import SACContinuous, PolicyNetContinuous, QValueNetContinuous
from peter_PPO_Network import PPO, PPOContinuous
from peter_detector import FishDetector

import time
import datetime

import threading
import queue
def convert_action(action):
    action_list = action.flatten().tolist()
    # print(action_list)
    formatted_list = []
    # print(action_list[0], action_list[1], action_list[2], action_list[3])
    for i in range(len(action_list)):
        rounded = round(action_list[i])
        formatted_list.append(f"{rounded:02d}")     
    print(formatted_list)
    
def env():
    env_name = 'CartPole-v0'
    # env_name = 'Pendulum-v1'
    env = gym.make(env_name)
    env.seed(0)
    torch.manual_seed(0)
    state_state = env.observation_space
    action_state = env.action_space
    print("state: ", state_state)
    print(type(state_state))
    print("action: ", action_state)
    print(type(action_state))
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
    # action_dim = env.action_space.shape[0]
    action_dim = env.action_space.n
    print("state_dim: ", state_dim)
    print("action_dim: ", action_dim)
    actor_lr = 1e-3
    critic_lr = 1e-2
    num_episodes = 500
    hidden_dim = 128
    gamma = 0.98
    lmbda = 0.95
    epochs = 10
    eps = 0.2
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda,epochs, eps, gamma, device)
    transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
    state1 = env.reset()
    state2 = env.reset()
    for i in range(30):
        # state = env.reset()
        done = i % 5 == 0
      
        action = agent.take_action(state1)
        # next_state, reward, done, _ = env.step(action)
        transition_dict['states'].append(state1)
        transition_dict['actions'].append(action)
        transition_dict['next_states'].append(state2)
        transition_dict['rewards'].append(10)
        transition_dict['dones'].append(done)
        # print(state)
        print("state: ", state1)
        print("action: ", action)
        print("next_state: ", state2)
        print("reward: ", 10)
        print("done: ", done)
        print("---")
            # state = next_state
        # agent.update(transition_dict)





    # env_name = 'Pendulum-v1'
    # env = gym.make(env_name)
    # state_dim = env.observation_space.shape[0]
    # action_dim = env.action_space.shape[0]
    # action_bound = env.action_space.high[0]  # 动作最大值
    # random.seed(0)
    # np.random.seed(0)
    # env.seed(0)
    # torch.manual_seed(0)

    # actor_lr = 3e-4
    # critic_lr = 3e-3
    # alpha_lr = 3e-4
    # num_episodes = 100
    # hidden_dim = 128
    # gamma = 0.99
    # tau = 0.005  # 软更新参数
    # buffer_size = 100000
    # minimal_size = 1000
    # batch_size = 64
    # target_entropy = -env.action_space.shape[0]
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    #     "cpu")

    # replay_buffer = rl_utils.ReplayBuffer(buffer_size)
    # agent = SACContinuous(state_dim, hidden_dim, action_dim, action_bound,actor_lr, critic_lr, alpha_lr, target_entropy, tau,gamma, device)
    
    
    # print("action_dim: ", action_dim)
    # print("action_bound: ", action_bound)
    # an_action = env.action_space.sample()
    # print("an_action: ", an_action)
    # print(type(an_action))

    # print("state_dim: ", state_dim)
    # a_state = env.observation_space.sample()
    # print("a_state: ", a_state)
    # print(type(a_state))
    # a_next_state = env.observation_space.sample()
    # print("a_next_state: ", a_next_state)
    # print(type(a_next_state))
    # a_reward = env.reward_range[0]
    # print("a_reward: ", a_reward)
    # print(type(a_reward))
    # a_done = False
    # print("a_done: ", a_done)
    # print(type(a_done))

    # select_action = agent.take_action(a_state)
    # print("select_action: ", select_action)
    # print(type(select_action))
    # agent.update(a_state, select_action, a_reward, a_next_state, a_done)


    # return_list = rl_utils.train_off_policy_agent(env, agent, num_episodes,
    #                                             replay_buffer, minimal_size,
    #                                             batch_size)

def network_SAC():
    state_dim = 10
    action_dim = 4


    # random.seed(0)
    # np.random.seed(0)

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
    state = np.array([551.5, \
                        326.0, \
                        630.0, \
                        364.5, \
                        -16.261203482894103, \
                        -10.441813660743295, \
                        -74.39016383661483, \
                        -47.768188239647465,\
                        1470, \
                        825])
    next_state = np.array([455.5, \
                        272.5, \
                        540.0, \
                        301.0, \
                        2.958600371969169, \
                        1.5897086297370844, \
                        -87.68785774187373, \
                        -47.11624641709709,\
                        1470, \
                        825])
    # state = np.expand_dims([1403, 340, 960, 540, 1, 1, 1], axis=0)
    # next_state = np.expand_dims([1000, 430, 690, 450, 2, 2, 2], axis=0)
    # print(state)
    # print(type(state))
    action = agent.take_action(state)
    convert_action(action)
    # print(action)
    # print(type(action))

    # action_list = action.flatten().tolist()
    # print(action_list)
    # formatted_list = []
    # print(action_list[0], action_list[1], action_list[2], action_list[3])
    # for i in range(len(action_list)):
    #     rounded = round(action_list[i])
    #     # if i == 1:
    #     #    formatted_list.append( f"{rounded + 30:02d}")
    #     # else:
    #     #    formatted_list.append(f"{rounded:02d}")
    #     formatted_list.append(f"{rounded:02d}")
        
    # print(formatted_list)

    transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
    transition_dict['states'].append(state)
    transition_dict['actions'].append(action)
    transition_dict['next_states'].append(next_state)
    transition_dict['rewards'].append(10)
    transition_dict['dones'].append(True)
    # print(transition_dict)
    agent.update(transition_dict)
    select_action = agent.take_action(state)
    convert_action(select_action)
    for i in range(30):
        # 添加噪声和状态变化
        noise = np.random.normal(0, 0.1, state.shape)
        dynamic_state = state + noise * i
        dynamic_next_state = next_state + noise * i
        
        transition_dict = {
            'states': [dynamic_state],
            'actions': [agent.take_action(dynamic_state)],
            'next_states': [dynamic_next_state],
            'rewards': [10 + i],  # 奖励动态变化
            'dones': [i % 5 == 0]  # 动态终止信号
        }
        agent.update(transition_dict)
        action = agent.take_action(state)
        convert_action(action)
    # print("select_action: ", select_action)
    # print("shape: ", select_action.shape)
    # print(type(select_action))
    # agent.update(transition_dict)

def newwork_PPO():
    # torch.manual_seed(1)
    state_dim = 10
    action_dim = 2
    actor_lr = 1e-3
    critic_lr = 1e-2
    num_episodes = 500
    hidden_dim = 128
    gamma = 0.98
    lmbda = 0.95
    epochs = 10
    eps = 0.2
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "cpu")
    agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda,epochs, eps, gamma, device)
    state = np.array([551.5, \
                        326.0, \
                        630.0, \
                        364.5, \
                        -16.261203482894103, \
                        -10.441813660743295, \
                        -74.39016383661483, \
                        -47.768188239647465,\
                        1470, \
                        825])
    state_mean = state.mean()
    state_std = state.std()
    state = (state - state_mean) / state_std
    next_state = np.array([455.5, \
                        272.5, \
                        540.0, \
                        301.0, \
                        2.958600371969169, \
                        1.5897086297370844, \
                        -87.68785774187373, \
                        -47.11624641709709,\
                        1470, \
                        825])
    next_state_mean = next_state.mean()
    next_state_std = next_state.std()
    next_state = (next_state - next_state_mean) / next_state_std
    
    # state = np.array([551.5, \
    #                     326.0, \
    #                     630.0, \
    #                     364.5])
    # next_state = np.array([455.5, \
    #                     272.5, \
    #                     540.0, \
    #                     301.0])

    # state = np.array([0.03132702,0.04127556,0.01066358,0.02294966])
    # next_state = np.array([0.0043625,0.04350724,0.03158535,-0.04972615])
    # state = np.array([0.03132702,0.04127556,0.01066358,0.02294966,0.03132702,0.04127556,0.01066358,0.02294966,0.03132702,0.04127556])
    # next_state = np.array([0.0043625,0.04350724,0.03158535,-0.04972615,0.0043625,0.04350724,0.03158535,-0.04972615,0.0043625,0.04350724])
    
    # action = agent.take_action(state)
    # print(action)
    # # convert_action(action)
    # # print(action)
    # # print(action.shape)
    # # print(type(action))
    # # action_list = action.flatten().tolist()
    # # # print(action_list)
    # # formatted_list = []
    # # # print(action_list[0], action_list[1], action_list[2], action_list[3])
    # # for i in range(len(action_list)):
    # #     rounded = round(action_list[i])
    # #     formatted_list.append(f"{rounded:02d}")     
    # # print(formatted_list)

    # transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
    # transition_dict['states'].append(state)
    # transition_dict['actions'].append(action)
    # transition_dict['next_states'].append(next_state)
    # transition_dict['rewards'].append(10)
    # transition_dict['dones'].append(True)
    # # print(transition_dict)
    # agent.update(transition_dict)
    # select_action = agent.take_action(state)
    # print(select_action)
    # convert_action(select_action)

    for i in range(10):
        # 添加噪声和状态变化
        noise = np.random.normal(0, 0.1, state.shape)
        dynamic_state = state + noise * i
        dynamic_next_state = next_state + noise * i
        action = agent.take_action(state)
        done = i % 5 == 0
        # print(action)
        
        transition_dict = {
            # 'states': [dynamic_state],
            
            # 'actions': [agent.take_action(dynamic_state)],
            # 'next_states': [dynamic_next_state],
            # 'rewards': [10 + i],  # 奖励动态变化

            'states': [state],
            'actions': [action],
            'next_states': [next_state],
            'rewards': [10],

            'dones': [done]  # 动态终止信号
            
        }
        print("state: ", state)
        print("action: ", action)
        print("next_state: ", next_state)
        print("reward: ", 10)
        print("done: ", done)
        print("---")
        agent.update(transition_dict)
       
        # convert_action(action)



    
    
def main():
    # env()
    # network_SAC()
    newwork_PPO()

if __name__ == '__main__':
    main()