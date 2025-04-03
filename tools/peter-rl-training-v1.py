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
     
    def __init__(self,capture_cfg, mmpose_cfg, anno_cfg, writer_cfg = None, ):
        # self.action_space = spaces.Discrete(5) # 0, 1, 2，3，4: 不动，上下左右
        # self.observation_space = spaces.Box(np.array([-self.L, -self.L]), np.array([self.L, self.L]))
        self.state = None
        self.fish_detector = FishDetector(capture_cfg, mmpose_cfg, anno_cfg, writer_cfg)

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
        'control_width': 200
    }

    win11_capture_cfg_dict = {
        'detect_type': 'video',
        # 'capture_type': 1,
        'input_vidoe_path' : r"C:\Users\peter\OneDrive\毕设\数据集\Fish-1222\fish-1222-demo20.mp4",
        # 'input_vidoe_path' : r"E:\Fish-0214\fish-0214-demo5.mp4",
        # 'input_vidoe_path' :r"E:\Fish-0223\fish-0223-demo13.mp4",
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
        'save_frame_flag': True,
        'save_frame_path': 'opencv_demo.mp4',
        'save_json_flag': True,
        'save_json_path': 'opencv_demo.json',
        
    }

    my_detector = FishDetector(win11_capture_cfg_dict, win11_mmpose_cfg_dict, anno_cfg_dict,)
    my_detector.minimun_pipeline()



if __name__ == '__main__':
    main()

