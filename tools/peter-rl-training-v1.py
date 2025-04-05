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
        self.reach_threshold = 0.05
        self.start_point = None

    def step(self, action):
        '''
        next_state, reward, done, _ = env.step(action)
        '''

        self.counts += 1

        # check if reach target
        if self.is_touch_rect():
            done = True
            reward_pos = -10
        else:
            target_distance = self.fish_detector.get_distance()
            if target_distance <= self.reach_threshold:
                done = True
                reward_pos = 10
            else:
                done = False
                reward_pos = 0

        reward = reward_pos# check if out of boundary
            
        return self.fish_detector.get_state(), reward, done, {}
    
    def reset(self):
        self.start_point = self.fish_detector.get_start_point()
        self.counts = 0
        return self.state
        
        
    def close(self):
        return None
    
    def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size):
        return_list = []
        for i in range(10):
            with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
                for i_episode in range(int(num_episodes/10)):
                    episode_return = 0
                    state = env.reset()
                    done = False
                    while not done:
                        action = agent.take_action(state)
                        next_state, reward, done, _ = env.step(action)
                        replay_buffer.add(state, action, reward, next_state, done)
                        state = next_state
                        episode_return += reward
                        if replay_buffer.size() > minimal_size:
                            b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                            transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                            agent.update(transition_dict)
                    return_list.append(episode_return)
                    if (i_episode+1) % 10 == 0:
                        pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                    pbar.update(1)
        return return_list
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

