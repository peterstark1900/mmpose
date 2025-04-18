import random
# import gym
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.distributions import Normal
import matplotlib.pyplot as plt
from rl_utils import ReplayBuffer, train_off_policy_agent, train_on_policy_agent, moving_average



from peter_env import Fish2DEnv
from peter_RLNetwork import PolicyNet, ValueNet, ActorCritic
from peter_detector import FishDetector

import time

import threading

# class Fish2DEnv(gym.Env):

def rl_train(detector):

    #############################  debug region  #######################
    i_episode = 0
    theta_current = None
    distance_current = None
    theta_last = None
    distance_last = None
    lambda_1 = 1
    lambda_2 = 1
    reach_threshold = 0.05
    change_start_point = True
    start_point = None
    while True:
        while not detector.get_train_flag():
            print("\r Waiting for training command...", end="")
            time.sleep(0.1)
            change_start_point = True
        # first time
        # if start_point is None:
            # start_point = detector.get_state()
        if theta_last is None:
            theta_last = detector.get_angle()
        if distance_last is None:
            distance_last = detector.get_distance()
        if theta_current is None:
            theta_current = detector.get_angle()
        if distance_current is None:
            distance_current = detector.get_distance()

        #  loop 
        # if change_start_point:
        #     start_point = detector.get_state()
        #     change_start_point = False

        if detector.is_in_rect():
            reward_pos = -10
            done = True
        else:
            distance_current = detector.get_distance()
            if distance_current <= reach_threshold:
                reward_pos = 10
                done = True
            else:
                done = False
                reward_pos = 0

            reward_appr = (distance_current - distance_last) * lambda_2

            theta_current = detector.get_angle()
            dot_theta = theta_current - theta_last
            reward_theta = -dot_theta * lambda_1

        reward = reward_pos + reward_appr + reward_theta

        theta_last = theta_current
        distance_last = distance_current
        # detector.get_state(start_point)
        # print("Training command received!")
        # print("start episode %d"%(i_episode))
        # i_episode += 1
        # time.sleep(5)
        # print('\r continue training: '+str(detector.is_in_rect())+' Distance: '+str(detector.get_distance())+' Theta: '+str(detector.get_angle()), end="")

        print('\r continue training: '+str(detector.is_in_rect())+' Distance: '+str(detector.get_distance())+' Theta: '+str(detector.get_angle())+' reward: '+ str(reward), end="")
        
    #############################  real region  #######################
    # env = Fish2DEnv(win11_capture_cfg_dict, win11_mmpose_cfg_dict, anno_cfg_dict, my_serial_config_dict, my_reward_cfg_dict, win11_save_cfg_dict)
    # actor_lr = 1e-3
    # critic_lr = 1e-2
    # num_episodes = 1000
    # hidden_dim = 128
    # gamma = 0.98
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    #     "cpu")

    # torch.manual_seed(0)
    # state_dim = env.observation_space.shape[0]
    # action_dim = env.action_space.n
    # agent = ActorCritic(state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
    #                     gamma, device)
    # # train_on_policy_agent(env, agent, num_episodes)
    # return_list = []
    # for i in range(10):
    #     with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
    #         for i_episode in range(int(num_episodes/10)):
    #             episode_return = 0
    #             transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
    #             while env.get_detector_train_flag():
    #                 print("\r Waiting for training command...",end="")
    #             print("Training command received!")
    #             print("start episode %d"%(i_episode))
    #             state = env.reset()
    #             done = False
    #             while not done:
    #                 action = agent.take_action(state)
    #                 next_state, reward, done, _ = env.step(action)
    #                 transition_dict['states'].append(state)
    #                 transition_dict['actions'].append(action)
    #                 transition_dict['next_states'].append(next_state)
    #                 transition_dict['rewards'].append(reward)
    #                 transition_dict['dones'].append(done)
    #                 state = next_state
    #                 episode_return += reward
    #             return_list.append(episode_return)
    #             agent.update(transition_dict)
    #             if (i_episode+1) % 10 == 0:
    #                 pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
    #             pbar.update(1)
    
def main():
    anno_cfg_dict = {
        'my_anno_flag': True,
        'current_rect': (100, 100, 200, 200),
        'target_x': 600,
        'target_y': 600,
        'drag_threshold': 5,
        'edit_button_color': (0, 0, 255),
        'target_button_color': (0, 0, 255),
        'detect_button_color': (0, 0, 255),
        'train_button_color': (0, 0, 255),
        'win_width': 2120,
        'win_height': 1080,
        'control_width': 200,
    }

    win11_capture_cfg_dict = {
        # 'detect_type': 'camera',
        # 'capture_type': 1,
        'detect_type': 'video',
        # 'input_vidoe_path' : r"C:\Users\peter\OneDrive\毕设\数据集\Fish-1222\fish-1222-demo20.mp4",
        # 'input_vidoe_path' : r"E:\Fish-0214\fish-0214-demo5.mp4",
        # 'input_vidoe_path' :r"E:\Fish-0223\fish-0223-demo13.mp4",
        'input_vidoe_path' : r"E:\fish-0414\0414-demo-4.mp4",
        # 'input_vidoe_path' : r"E:\openmmlab\mmpose\opencv_demo-19-30.mp4",
    }

    win11_mmpose_cfg_dict = {
        'my_pose_cfg': r"E:\openmmlab\mmpose\configs\fish_keypoints\fish-keypoints-1222.py",
        'my_pose_weights': r"E:\openmmlab\mmpose\work_dirs\fish-keypoints-1222\epoch_1200_0303_mmpose.pth",
        'my_detect_cfg' : r"E:\openmmlab\mmdetection\configs\fish\fish1210-rtmdet_tiny_8xb32-300e_coco.py",
        # 'my_detect_weights' : r"C:\Users\peter\OneDrive\毕设\demo_video\epoch_300.pth",
        'my_detect_weights' : r"E:\openmmlab\mmdetection\work_dirs\fish1222-rtmdet_tiny_8xb32-300e_coco\epoch_1200_0303_mmdet.pth",
        'my_kpt_thr' : 0.2,
        'my_real_num' : 1,
        'my_draw_flag' : True,
        'my_save_flag ': False,
        
        'output_path' : 'opencv_demo.mp4'
    }

    win11_save_cfg_dict = {
        'save_frame_flag': False,
        'video_output_path': '/video',
        'save_json_flag': False,
        'json_output_path': '/output/json', 
        'mix_anno_flag': False,
    }

    my_serial_config_dict = {
        "port": "/dev/COM7",
        "baudrate": 115200,
        "duration": 5,
        "control_type": "auto",
    }
    my_reward_cfg_dict = {
        'reach_threshold': 0.05,
        'lambda_1': 0.1,
        'lambda_2': 0.1,        
    }

    my_detector = FishDetector(win11_capture_cfg_dict, win11_mmpose_cfg_dict, anno_cfg_dict,win11_save_cfg_dict)
    training_thread = threading.Thread(target=rl_train, args=(my_detector,))
    training_thread.daemon = True
    training_thread.start()
    # my_detector.minimun_pipeline()
    my_detector.a_fish_pipeline()
    # while True:
    #     print('\r continue training: '+str(True)+' Distance: '+str(0.1)+' Theta: '+str(0.1), end="")





if __name__ == '__main__':
    main()

