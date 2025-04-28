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
from peter_AC_Network import PolicyNet, ValueNet, ActorCritic
from peter_SAC_Network import SACContinuous, PolicyNetContinuous, QValueNetContinuous
from peter_detector import FishDetector

import time
import datetime

import threading
import queue

# class Fish2DEnv(gym.Env):

def rl_train(detector, serial_cfg = None, reward_cfg = None):

    #############################  debug region  #######################
    i_episode = 0
    lambda_1 = 1
    lambda_2 = 1
    reach_threshold = 0.05
    num_episodes = 20
    # transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
    
    # while True:
    return_list = []
    max_episode_steps = 30
    sleep_time = 1
    # for i in range(10):
    #     with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
    #         for i_episode in range(int(num_episodes/10)):
    #             
    with tqdm(total=num_episodes) as pbar:
        for i_episode in range(num_episodes):
                episode_return = 0
                while not detector.get_train_flag():
                    # print("\r Waiting for training command...", end="")
                    time.sleep(0.1)
                # print("Training command received!")
                print("start episode %d"%(i_episode))
                detector.setup_episode_num(i_episode)

                # detector.setup_video_out()
                detector.setup_frame_stamps()
                # detector.set_save_state(True)
                done = False
                counts = 0
                while not done:
                    counts += 1
                    if counts >= max_episode_steps:
                        done = True
                        print("Episode steps reach the max!")
                        episode_return = counts
                        break
                    time.sleep(2)
                    if not detector.is_in_rect():
                        # debuging, disable this line temporarily 
                        print("Fish is out of the region!")
                        done = True 
                        reward_pos = -10
                        reward = reward_pos 
                    else:
                        # detector.calculate_distance()
                        # distance_current = detector.get_distance_current()
                        # distance_last = detector.get_distance_last()
                        # if distance_current <= reach_threshold:
                        #     done = True
                        #     reward_pos = 10
                        # else:
                        #     done = False
                        #     reward_pos = 0
                        
                        # reward_appr = (distance_current - distance_last)*lambda_2
                        # detector.calculate_theta_current()
                        # theta_current = detector.get_theta_current()
                        # detector.calculate_theta_dot()
                        # dot_theta = detector.get_theta_dot()
                        # reward_theta = -dot_theta*lambda_1
                        # reward = reward_pos + reward_appr + reward_theta
                        # episode_return += reward 
                    
                        # print('\r continue training: '+str(detector.is_in_rect())+' Distance: '+str(distance_current)+' Theta: '+str(theta_current)+' reward: '+ str(reward), end="")

                        detector.setup_get_state_flag(True)
                        temp_array = detector.get_state(2,2)
                        detector.reset_pos_list()
                        print("\n p_hx_avg: "+str(temp_array[0])\
                            +"\n p_hy_avg: "+str(temp_array[1])\
                            +"\n p_bx_avg: "+str(temp_array[2])\
                            +"\n p_by_avg: "+str(temp_array[3])\
                            +"\n theta_avg: "+str(temp_array[4])\
                            +"\n omega_avg: "+str(temp_array[5])\
                            +"\n displacement_avg: "+str(temp_array[6])\
                            +"\n velocity_avg: " +str(temp_array[7])\
                            +"\n l_x: " +str(temp_array[8])\
                            +"\n l_y: " +str(temp_array[9]))
                        detector.setup_get_state_flag(False)

                    # print('\r counts:  %d'%(counts), end="")
                    # time.sleep(sleep_time)
        
                # detector.set_save_state(False)
                detector.set_train_flag(False)
                # detector.export_current_video()
                detector.export_frame_stamps()
                return_list.append(episode_return)

                # if (i_episode+1) % 10 == 0:
                #     pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    
    # return return_list

    

    #############################  real region  #######################
    env = Fish2DEnv(detector, serial_cfg, reward_cfg)
    state_dim = 7
    action_dim = 4


    random.seed(0)
    np.random.seed(0)

    torch.manual_seed(0)

    actor_lr = 1e-3
    critic_lr = 1e-2
    alpha_lr = 3e-4
    num_episodes = 100
    hidden_dim = 128
    gamma = 0.98
    tau = 0.005  # 软更新参数
    buffer_size = 100000
    minimal_size = 3
    batch_size = 1
    target_entropy = -4
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "cpu")

    replay_buffer = ReplayBuffer(buffer_size)
    agent = SACContinuous(state_dim, hidden_dim, action_dim,
                      actor_lr, critic_lr, alpha_lr, target_entropy, tau,
                      gamma, device)
    return_list = []
    # for i in range(10):
    #     with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
    #         for i_episode in range(int(num_episodes/10)):
    #            
    with tqdm(total=num_episodes) as pbar:
        for i_episode in range(num_episodes):
                episode_return = 0
                while not detector.get_train_flag():
                    print("\r Waiting for training command...",end="")
                    time.sleep(0.1) #avoid busy waiting
                print("Training command received!")
                print("start episode %d"%(i_episode))
                detector.setup_episode_num(i_episode)
                # detector.setup_video_out()
                detector.setup_frame_stamps()
                # detector.set_save_state(True)
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    print("stepping...")
                    next_state, reward, done, _ = env.step(action)
                    print("step done!")
                    replay_buffer.add(state, action, reward, next_state, done)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                        agent.update(transition_dict)
                return_list.append(episode_return)
                # detector.set_save_state(False)
                detector.set_train_flag(False)
                # detector.export_current_video()
                detector.export_frame_stamps()

                # if (i_episode+1) % 10 == 0:
                #     pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)

    return return_list        

    # episodes_list = list(range(len(return_list)))
    # plt.style.use('bmh') 
    # plt.figure(figsize=(16, 10), dpi=600)
    # plt.plot(episodes_list, return_list)
    # plt.xlabel('Episodes')
    # plt.ylabel('Returns')
    # plt.title('Reward')
    # plt.grid(True)
    # # plt.show()  
    # filename = 'reward_'+datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')+'.png'
    # plt.savefig(filename,format='png',bbox_inches = 'tight')

    # current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    # save_path = f'sac_model_{current_time}.pth'
    # agent.save_model(save_path)
    # print(f'Model saved to {save_path}')

def main():
    anno_cfg_dict = {
        'my_anno_flag': True,
        'current_rect': (250, 125, 1720, 950),
        'target_x': 1300,
        'target_y': 700,
        'drag_threshold': 5,
        'edit_button_color': (0, 0, 255),
        'target_button_color': (0, 0, 255),
        'detect_button_color': (0, 0, 255),
        'train_button_color': (0, 0, 255),
        'exit_button_color': (0, 0, 255),
        'win_width': 2120,
        'win_height': 1080,
        'control_width': 200,
    }

    win11_capture_cfg_dict = {
        'detect_type': 'camera',
        # 'capture_type': 1,
        # 'detect_type': 'video',
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
        'save_video_flag': False,
        'video_output_path': '/output/video',
        'save_json_flag': True,
        'json_output_path': '/output/json', 
        'mix_anno_flag': True,
    }

    my_serial_config_dict = {
        "port": "/dev/COM7",
        "baudrate": 115200,
        "duration": 5,
        "control_type": "auto",
    }
    my_reward_cfg_dict = {
        'lambda_1': 0.1,
        'lambda_2': 0.1,
        'lambda_3': 0.1,  
    }

    my_detector = FishDetector(win11_capture_cfg_dict, win11_mmpose_cfg_dict, anno_cfg_dict,win11_save_cfg_dict)
    my_detector.set_save_state(False)
    result_queue = queue.Queue()
    training_thread = threading.Thread(target=rl_train, args=(my_detector,my_serial_config_dict,my_reward_cfg_dict))
    # training_thread = threading.Thread(target=rl_train, args=(my_detector, result_queue))
    training_thread.daemon = True
    training_thread.start()
    # my_detector.minimun_pipeline()
    my_detector.a_fish_pipeline()
    # training_thread.join()
    # return_list = result_queue.get()
    # episodes_list = list(range(len(return_list)))
    # plt.style.use('bmh') 
    # plt.figure(figsize=(16, 10), dpi=600)
    # plt.plot(episodes_list, return_list)
    # plt.xlabel('Episodes')
    # plt.ylabel('Returns')
    # plt.title('Reward')
    # plt.grid(True)
    # # plt.show()  
    # filename = 'reward_'+datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')+'.png'
    # plt.savefig(filename,format='png',bbox_inches = 'tight')
    # while True:
    #     print('\r continue training: '+str(True)+' Distance: '+str(0.1)+' Theta: '+str(0.1), end="")





if __name__ == '__main__':
    main()

