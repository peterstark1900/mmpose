from peter_detector import FishDetector
from peter_serial import SerialAction
import time

class Fish2DEnv():
     
    def __init__(self,fish_detector, serial_cfg, reward_cfg):
        # self.action_space = spaces.Discrete(5) # 0, 1, 2，3，4: 不动，上下左右
        # self.observation_space = spaces.Box(np.array([-self.L, -self.L]), np.array([self.L, self.L]))

        self.fish_detector =fish_detector

        self.fish_control = SerialAction(serial_cfg)

        self.lambda_1 = reward_cfg["lambda_1"]
        self.lambda_2 = reward_cfg["lambda_2"]
        self.lambda_3 = reward_cfg["lambda_3"]
        # self.reach_threshold = reward_cfg["reach_threshold"]
        self.counts = 0
        self.max_episode_steps = 1000

        self.theta_avg_total = 0
        self.last_theta_avg_total = 0
        self.theta_avg_trend = 0  # 趋势标记：1递增，-1递减，0初始状态
        self.theta_avg_mono = 1

        # self.theta_reward_coeff = 0.1  # 奖励系数，可调整
        self.last_omega_avg = None
        self.omega_trend = 0  # 趋势标记：1递增，-1递减，0初始状态
        self.omega_mono = 1


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
            rounded = round(action_list[i])
            formatted_list.append(f"{rounded:02d}")   
        # print(formatted_list)

        self.fish_control.send('CRE',formatted_list[0], formatted_list[1], formatted_list[2], formatted_list[3])

        time.sleep(3)


        # calculate the reward
        # param_offset = action_list[1]
        # param_ratio = action_list[3] 
        # # param_ratio is 10 times of the real ratio
        # if param_offset > 0 and param_ratio > 10:
        #     reward_param = 10
        # elif param_offset < 0 and param_ratio < 10:
        #     reward_param = 10
        # elif param_offset == 0 and param_ratio == 10:
        #     reward_param = 10
        # else:
        #     reward_param = -10

    
        # self.fish_detector.calculate_distance()
        # distance_current = self.fish_detector.get_distance_current()
        # distance_last = self.fish_detector.get_distance_last()
        # if distance_current <= self.reach_threshold:
        #     done = True
        #     reward_pos = 10
        # else:
        #     done = False
        #     reward_pos = 0
        
        # reward_appr = (distance_current - distance_last)*self.lambda_2
        # self.fish_detector.calculate_theta_current()
        # theta_current = self.fish_detector.get_theta_current()
        # self.fish_detector.calculate_theta_dot()
        # dot_theta = self.fish_detector.get_theta_dot()
        # reward_theta = -dot_theta*self.lambda_1
        
        # if self.counts >= self.max_episode_steps:
        #     done = True
        #     print("Episode steps reach the max!")

        self.fish_detector.setup_get_state_flag(False)
        state_array = self.fish_detector.get_state(2,2)
        self.fish_detector.reset_pos_list()

        theta_avg = state_array[4]
        omega_avg = state_array[5]
        displacement_avg = state_array[6]
        velocity_avg = state_array[7]

        self.theta_avg_total = self.theta_avg_total + theta_avg

        if self.last_theta_avg_total is None:
            # At the frist time 
            self.theta_avg_total_mono += 2  
            # update the omega_trend
            self.theta_avg_total_trend = 1 if self.theta_avg_total > 0 else (-1 if self.theta_avg_total < 0 else 0)
            # 1 if current angle (theta_avg) is greater than previous angle (last_theta_avg)
            # -1 if current angle is less than previous angle
            # 0 if angles are equal (no change)

        else:
            theta_avg_total_direction = 1 if self.theta_avg_total > self.last_theta_avg_total else (-1 if self.theta_avg_total < self.last_theta_avg_total else 0)
            # 1 if current angle (theta_avg) is greater than previous angle (last_theta_avg)
            # -1 if current angle is less than previous angle
            # 0 if angles are equal (no change)

            if theta_avg_total_direction == self.theta_avg_total_trend:
                # when theta_avg is 
                self.theta_avg_total_mono += 2  # 趋势持续时奖励递增
            else:
                self.theta_avg_total_mono = -1 # 趋势改变时奖励递减
            # update the omega_trend
            self.theta_avg_total_trend = theta_avg_total_direction     
            
        # update the last_theta_avg
        self.last_theta_avg_total = self.theta_avg_total
        
        if self.last_omega_avg is None:
            # At the frist time 
            self.omega_mono += 2  
            # update the omega_trend
            self.omega_trend = 1 if omega_avg > 0 else (-1 if omega_avg < 0 else 0)
            # 1 if current angle (theta_avg) is greater than previous angle (last_theta_avg)
            # -1 if current angle is less than previous angle
            # 0 if angles are equal (no change)

        else:
            omega_direction = 1 if omega_avg > self.last_omega_avg else (-1 if omega_avg < self.last_omega_avg else 0)
            # 1 if current angle (theta_avg) is greater than previous angle (last_theta_avg)
            # -1 if current angle is less than previous angle
            # 0 if angles are equal (no change)
            if omega_direction != 0:
                if omega_direction == self.omega_trend:
                    # when theta_avg is 
                    self.omega_mono += 2  # 趋势持续时奖励递增
                else:
                    self.omega_mono = -1 # 趋势改变时奖励递减
                # update the omega_trend
                self.omega_trend = omega_direction     
            else:
                # when theta_avg does not change, the reward will be the same as the last time
                # omega_trend would not update in this case
                self.omega_mono = 1  # 趋势不变时奖励不变
        # update the last_theta_avg
        self.last_omega_avg = omega_avg

        reward_theta = self.lambda_1*self.theta_avg_total_mono*self.theta_avg_total + self.lambda_2*self.omega_mono*omega_avg
        reward_dis = self.lambda_3*velocity_avg
        if self.fish_detector.is_in_rect():
            done = True
            if abs(self.theta_avg_total) > 175:
                reward_pos = 100
            else:
                reward_pos = -100
            reward = reward_pos 
        else:
            if self.theta_avg_total>175:
                done = True
                reward_pos = 100
            else:
                done = False
                reward_pos = 0

        reward = reward_pos + reward_theta - reward_dis
        return self.fish_detector.get_state(), reward, done, {}
    
    def reset(self):
        '''
        state = env.reset()
        '''
        # get the state for the agent
        # self.fish_detector.calculate_theta_current()
        # self.fish_detector.calculate_theta_dot()
        # self.fish_detector.detect_a_fish()
        # self.fish_detector.calculate_v_x()
        # self.fish_detector.calculate_v_y()
        self.fish_detector.setup_get_state_flag(True)
        state = self.fish_detector.get_state(2,2)
        self.fish_detector.reset_pos_list()
        self.theta_avg_total = 0
        self.last_theta_avg_total = 0
        self.theta_avg_trend = 0  # 趋势标记：1递增，-1递减，0初始状态
        self.theta_avg_mono = 1
        self.last_omega_avg = None
        self.omega_trend = 0  # 趋势标记：1递增，-1递减，0初始状态
        self.omega_mono = 1
        self.fish_detector.setup_get_state_flag(False)
        print(state)
        print(type(state))
        self.counts = 0
        return state

    def close(self):
        return None