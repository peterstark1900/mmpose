from peter_detector import FishDetector
from peter_serial import SerialAction

class Fish2DEnv():
     
    def __init__(self,fish_detector, serial_cfg, reward_cfg):
        # self.action_space = spaces.Discrete(5) # 0, 1, 2，3，4: 不动，上下左右
        # self.observation_space = spaces.Box(np.array([-self.L, -self.L]), np.array([self.L, self.L]))

        self.fish_detector =fish_detector

        # self.fish_control = SerialAction(serial_cfg)

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
        # if action[0] == 0:
        #     motion_state = 'CSE'
        # else:
        #     motion_state = 'CFE'
        # self.fish_control.send(motion_state,action[1],action[2],action[3],action[4])
        action_list = action.flatten().tolist()
        formatted_list = []
        for i in range(len(action_list)):
            action_list[i] = round(action_list[i])
            if i == 1:
                formatted_list.append( f"{action_list[i] + 30:02d}")
            else:
                formatted_list.append(f"{action_list[i]:02d}")

        # self.fish_control.send('CFE',formatted_list[0], formatted_list[1], formatted_list[2], formatted_list[3])

        # calculate the reward
        param_offset = action_list[1]
        param_ratio = action_list[3] 
        # param_ratio is 10 times of the real ratio
        if param_offset > 0 and param_ratio > 10:
            reward_param = 10
        elif param_offset < 0 and param_ratio < 10:
            reward_param = 10
        elif param_offset == 0 and param_ratio == 10:
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
        self.fish_detector.calculate_theta_current()
        self.fish_detector.calculate_theta_dot()
        self.fish_detector.detect_a_fish()
        self.fish_detector.calculate_v_x()
        self.fish_detector.calculate_v_y()
        state = self.fish_detector.get_state()
        print(state)
        print(type(state))
        self.counts = 0
        return state

    def close(self):
        return None