import random
# import gym
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.distributions import Normal
import matplotlib.pyplot as plt
from rl_utils import ReplayBuffer, train_off_policy_agent, train_on_policy_agent, moving_average
import os
import glob


from peter_env import Fish2DEnv
from peter_AC_Network import PolicyNet, ValueNet, ActorCritic
from peter_SAC_Network import SACContinuous, PolicyNetContinuous, QValueNetContinuous
from peter_PPO_Network import PPO
from peter_detector import FishDetector
from peter_serial import SerialAction

import time
import datetime

import threading
import queue

class e2e():
    def __init__(self,detector,serial_cfg,actor_pth):

        self.exit_flag = False
        self.detector = detector
        self.fish_control = SerialAction(serial_cfg)

        state_dim = 10
        action_dim = 3
        actor_lr = 1e-3
        critic_lr = 1e-2
        hidden_dim = 128
        gamma = 0.98
        lmbda = 0.95
        epochs = 10
        eps = 0.2
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
            "cpu")
        self.agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda,epochs, eps, gamma, device)
        # load model
        self.agent.load_model(actor_pth)

        self.times = 0
        self.reset_flag = False
        self.is_stopped = False
  
    def set_exit_flag(self,flag):
        self.exit_flag = flag

    def e2e_control(self,):
        while True:  
            if self.detector.get_exit_flag() or self.exit_flag == True:  # 添加退出标志检测
                self.exit_flag = True
            while not self.detector.get_train_flag() and not self.detector.get_exit_flag():
                print("\r Waiting for start...",end="")
                if not self.reset_flag:  # 检查是否已经停止
                    self.reset_flag = True
                    self.fish_control.send("CSE",None,None,None,None)
            if self.detector.get_exit_flag() or self.exit_flag == True:  # 再次检测退出标志
                self.exit_flag = True
                break
            if self.reset_flag == True:
                self.times += 1
                self.reset_flag = False
                print("Start command received!")
                print("start episode %d"%(self.times))
                self.detector.setup_episode_num(self.times)
                # detector.setup_video_out()
                self.detector.setup_frame_stamps()
                # detector.set_save_state(True)
                time.sleep(1)
            self.detector.setup_get_state_flag(True)
            state = self.detector.get_state()
            self.detector.reset_pos_list()
            self.detector.setup_get_state_flag(False)

            action = self.agent.take_action(state)

            # discrete action
            if action == 0:
                self.fish_control.send("CSE",None,None,None,None)
            elif action == 1:
                self.fish_control.send('CRE', '50', '30', '05', '40')
            elif action == 2:
                self.fish_control.send('CRE', '40', '15', '15', '10')
            print('sleep 1s')

            time.sleep(1)

                  

        
        
               

def main():
    anno_cfg_dict = {
        'my_anno_flag': True,
        'current_rect': (125, 50, 1795, 1025),
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
        "port": "COM7",
        "baudrate": 115200,
        "duration": 5,
        "control_type": "auto",
    }
    my_reward_cfg_dict = {
        'lambda_1': 0.01,
        'lambda_2': 0.1,
        'lambda_3': 0.05,  
    }
    

    my_detector = FishDetector(win11_capture_cfg_dict, win11_mmpose_cfg_dict, anno_cfg_dict,win11_save_cfg_dict)
    my_detector.set_save_state(False)
    actor_pth =  r"E:\openmmlab\mmpose\sac_model_20250502_141237.pth"
    my_e2e = e2e(my_detector,my_serial_config_dict,actor_pth)

    training_thread = threading.Thread(target=my_e2e.e2e_control)
    # training_thread = threading.Thread(target=rl_train_off_policy, args=(my_detector, result_queue))
    training_thread.daemon = True
    training_thread.start()

    # # my_detector.minimun_pipeline()
    # my_detector.a_fish_pipeline()

    # training_thread.join()
    try:
        my_detector.a_fish_pipeline()
    except Exception as e:
        print(f"Pipeline exited with error: {e}")
        pipeline_success = False
        my_e2e.set_exit_flag(True)
    else:
        pipeline_success = True

    # 仅在管道正常运行时等待训练线程
    if pipeline_success:
        training_thread.join()
        print("Training finished!")
    else:
        print("Skipping thread join due to pipeline failure")

    print("Training finished!")
    
    

    





if __name__ == '__main__':
    main()

