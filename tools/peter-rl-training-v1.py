import random
# import gym
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.distributions import Normal
import matplotlib.pyplot as plt
import rl_utils

# from gym import spaces

from mmpose.apis import MMPoseInferencer
import cv2
from peter_detector import FishDetector

# class Fish2DEnv(gym.Env):
class Fish2DEnv():
     
    def __init__(self,detector_config_dict):
        # self.action_space = spaces.Discrete(5) # 0, 1, 2，3，4: 不动，上下左右
        # self.observation_space = spaces.Box(np.array([-self.L, -self.L]), np.array([self.L, self.L]))
        self.state = None
        self.fish_detector = FishDetector(
            detector_config_dict.get('detect_type'),
            detector_config_dict.get('my_pose_cfg'),
            detector_config_dict.get('my_pose_weights'),
            detector_config_dict.get('my_detect_cfg'),
            detector_config_dict.get('my_detect_weights'),
            detector_config_dict.get('my_kpt_thr'),
            detector_config_dict.get('my_real_num'),
            detector_config_dict.get('my_draw_flag'),
            detector_config_dict.get('my_save_flag'),
            detector_config_dict.get('input_vidoe_path'),
            detector_config_dict.get('output_path')
        )

    def step(self, action):
        # assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        x, y = self.state
        if action == 0:
            x = x
            y = y
        if action == 1:
            x = x
            y = y + 1
        if action == 2:
            x = x
            y = y - 1
        if action == 3:
            x = x - 1
            y = y
        if action == 4:
            x = x + 1
            y = y
        self.state = np.array([x, y])
        self.counts += 1
            
        done = (np.abs(x)+np.abs(y) <= 1) or (np.abs(x)+np.abs(y) >= 2*self.L+1)
        done = bool(done)
        
        if not done:
            reward = -0.1
        else:
            if np.abs(x)+np.abs(y) <= 1:
                reward = 10
            else:
                reward = -50
            
        return self.state, reward, done, {}
    
    def reset(self):
        self.state = np.ceil(np.random.rand(2)*2*self.L)-self.L
        self.counts = 0
        return self.state
        
        
    def close(self):
        return None

win11_detector_config_dict = {
    'detect_type': 'video',
    'my_pose_cfg': r"E:\openmmlab\mmpose\configs\fish_keypoints\fish-keypoints-1222.py",
    'my_pose_weights': r"E:\openmmlab\mmpose\work_dirs\fish-keypoints-1222\epoch_1200_0303_mmpose.pth.pth",
    'my_detect_cfg' : r"E:\openmmlab\mmdetection\configs\fish\fish1210-rtmdet_tiny_8xb32-300e_coco.py",
    # 'my_detect_weights' : r"C:\Users\peter\OneDrive\毕设\demo_video\epoch_300.pth",
    'my_detect_weights' : r"E:\openmmlab\mmdetection\work_dirs\fish1222-rtmdet_tiny_8xb32-300e_coco\epoch_1200_0303_mmdet.pth",
    'my_kpt_thr' : 0.2,
    'my_real_num' : 1,
    'my_draw_flag' : True,
    'my_save_flag ': False,
    'input_vidoe_path' : r"C:\Users\peter\OneDrive\毕设\数据集\Fish-1222\fish-1222-demo20.mp4",
    # 'input_vidoe_path' : r"E:\Fish-0214\fish-0214-demo5.mp4",
    # 'input_vidoe_path' :r"E:\Fish-0223\fish-0223-demo13.mp4",
    'output_path' : 'opencv_demo.mp4'
}
