from peter_detector import FishDetector
from peter_serial import SerialAction

class Fish2DEnv():
     
    def __init__(self,capture_cfg, mmpose_cfg, anno_cfg, serial_cfg, reward_cfg,writer_cfg = None):
        # self.action_space = spaces.Discrete(5) # 0, 1, 2，3，4: 不动，上下左右
        # self.observation_space = spaces.Box(np.array([-self.L, -self.L]), np.array([self.L, self.L]))

        self.fish_detector = FishDetector(capture_cfg, mmpose_cfg, anno_cfg, writer_cfg)
        if writer_cfg is not None:
            self.save_flag = True
        else:
            self.save_flag = False
        
        self.start_point = None

        self.fish_control = SerialAction(serial_cfg)

        self.theta_current = None
        self.theta_last = None
        self.lambda_1 = reward_cfg["lambda_1"]
        
        self.distance = None
        self.distance_last = None
        self.lambda_2 = reward_cfg["lambda_2"]

        self.reach_threshold = reward_cfg["reach_threshold"]


    def step(self, action):
        '''
        next_state, reward, done, _ = env.step(action)
        '''

        self.counts += 1
        # execute the action
        if action[0] == 0:
            motion_state = 'CSE'
        elif action[0] == 1:
            motion_state = 'CFE'
        self.fish_control.send(motion_state,action[1],action[2],action[3],action[4])

        # save data
        if self.save_flag:
            self.fish_detector.save_data()

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

        if self.fish_detector.is_touch_rect():
            done = True
            reward_pos = -10
        else:
            self.distance_current = self.fish_detector.get_distance()
            if self.distance_current <= self.reach_threshold:
                done = True
                reward_pos = 10
            else:
                done = False
                reward_pos = 0
            
            reward_appr = (self.distance_current - self.distance_last)*self.lambda_2

            self.theta_current = self.fish_detector.get_theta()
            dot_theta = self.theta_current - self.theta_last
            reward_theta = -dot_theta*self.lambda_1

        reward = reward_pos + reward_appr + reward_theta + reward_param 
            
        return self.fish_detector.get_state(), reward, done, {}
    
    def reset(self):
        '''
        state = env.reset()
        '''
        # perpare for the coming episode
        self.counts = 0
        # self.timecode = datetime.datetime.now.strftime("%Y-%m-%d-%H-%M-%S")
        # self.fish_detector.reset_timecode(self.timecode)

        # update parameters

        if self.theata_last is None:
            self.theta_last = self.fish_detector.get_theta()
        else:
            self.theta_last = self.theta_current

        if self.distance_last is None:
            self.distance_last = self.fish_detector.get_distance()
        else:
            self.distance_last = self.distance_current


        # get the state for the agent
        state = self.fish_detector.get_state()
        return state

    def get_detector_train_flag(self):
        return self.fish_detector.get_train_flag()   
    def close(self):
        return None