from peter_detector import FishDetector
from peter_serial import SerialAction

class Fish2DEnv():
     
    def __init__(self,fish_detector, serial_cfg, reward_cfg):
        # self.action_space = spaces.Discrete(5) # 0, 1, 2，3，4: 不动，上下左右
        # self.observation_space = spaces.Box(np.array([-self.L, -self.L]), np.array([self.L, self.L]))

        self.fish_detector =fish_detector

        self.fish_control = SerialAction(serial_cfg)

        self.lambda_1 = reward_cfg["lambda_1"]
        self.lambda_2 = reward_cfg["lambda_2"]
        self.reach_threshold = reward_cfg["reach_threshold"]
        self.counts = 0
        self.max_episode_steps = 1000


    def step(self, action):
        '''
        next_state, reward, done, _ = env.step(action)
        '''

        self.counts += 1
        # execute the action
        if action[0] == 0:
            motion_state = 'CSE'
        else:
            motion_state = 'CFE'
        self.fish_control.send(motion_state,action[1],action[2],action[3],action[4])

        # calculate the reward
        param_offset = action[2]
        param_ratio = action[4] 
        if param_offset > 0 and param_ratio > 0:
            reward_param = 10
        elif param_offset < 0 and param_ratio < 1:
            reward_param = 10
        elif param_offset == 0 and param_ratio == 1:
            reward_param = 10
        else:
            reward_param = -10

        if self.fish_detector.is_in_rect():
            done = True
            reward_pos = -10
            reward = reward_pos + reward_param
        else:
            self.fish_detector.calculate_distance()
            distance_current = self.fish_detector.get_distance_current()
            distance_last = self.fish_detector.get_distance_last()
            if distance_current <= self.reach_threshold:
                done = True
                reward_pos = 10
            else:
                done = False
                reward_pos = 0
            
            reward_appr = (distance_current - distance_last)*self.lambda_2
            self.fish_detector.calculate_theta_current()
            theta_current = self.fish_detector.get_theta_current()
            self.fish_detector.calculate_theta_dot()
            dot_theta = self.fish_detector.get_theta_dot()
            reward_theta = -dot_theta*self.lambda_1
            
            if self.counts >= self.max_episode_steps:
                done = True
                print("Episode steps reach the max!")
            reward = reward_pos + reward_appr + reward_theta + reward_param 
            
        return self.fish_detector.get_state(), reward, done, {}
    
    def reset(self):
        '''
        state = env.reset()
        '''
        # get the state for the agent
        state = self.fish_detector.get_state()
        self.counts = 0
        return state

    def close(self):
        return None