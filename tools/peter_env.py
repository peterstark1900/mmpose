from peter_detector import FishDetector
from peter_serial import SerialAction
import time

class Fish2DEnv():
     
    def __init__(self,fish_detector, serial_cfg, reward_cfg):
        # self.action_space = spaces.Discrete(5) # 0, 1, 2，3，4: 不动，上下左右
        # self.observation_space = spaces.Box(np.array([-self.L, -self.L]), np.array([self.L, self.L]))

        self.fish_detector =fish_detector

        # self.fish_control = SerialAction(serial_cfg)

        self.lambda_1 = reward_cfg["lambda_1"]
        self.lambda_2 = reward_cfg["lambda_2"]
        self.lambda_3 = reward_cfg["lambda_3"]
        # self.reach_threshold = reward_cfg["reach_threshold"]
        self.elapsed_time = None
        self.max_episode_duration = 10
        self.episode_start_time = None

        self.theta_avg_total = 0
        self.last_theta_avg_total = 0
        self.theta_avg_total_trend = 0  # 趋势标记：1递增，-1递减，0初始状态
        self.theta_avg_mono = 1

        # self.theta_reward_coeff = 0.1  # 奖励系数，可调整
        self.last_omega_avg = None
        self.omega_trend = 0  # 趋势标记：1递增，-1递减，0初始状态
        self.omega_mono = 1

        self.last_theta_avg = None
        self.theta_trend = 0  # 趋势标记：1递增，-1递减，0初始状态
        self.theta_mono = 1

        self.done = False




    def step(self, action):
        '''
        next_state, reward, self.done, _ = env.step(action)
        '''

        self.elapsed_time =  time.time() - self.episode_start_time
        print(f"Elapsed time: {self.elapsed_time:.2f}s")

        # # continous action
        # action_list = action.flatten().tolist()
        # formatted_list = []
        # for i in range(len(action_list)):
        #     rounded = round(action_list[i])
        #     formatted_list.append(f"{rounded:02d}")   
        # print(formatted_list)

        # self.fish_control.send('CRE',formatted_list[0], formatted_list[1], formatted_list[2], formatted_list[3])
        print("action is: ",action)
        # # discrete action
        # if action == 0:
        #     self.fish_control.send("CSE",None,None,None,None)
        # elif action == 1:
        #     self.fish_control.send('CRE', '50', '30', '05', '40')
        # elif action == 2:
        #     self.fish_control.send('CRE', '40', '15', '15', '10')
        # print('sleep 1s')
        time.sleep(1)
        # print('sleep is over!')


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
        #     self.done = True
        #     reward_pos = 10
        # else:
        #     self.done = False
        #     reward_pos = 0
        
        # reward_appr = (distance_current - distance_last)*self.lambda_2
        # self.fish_detector.calculate_theta_current()
        # theta_current = self.fish_detector.get_theta_current()
        # self.fish_detector.calculate_theta_dot()
        # dot_theta = self.fish_detector.get_theta_dot()
        # reward_theta = -dot_theta*self.lambda_1
        print(self.elapsed_time >= self.max_episode_duration)
        if self.elapsed_time >= self.max_episode_duration:
            self.done = True
            # print("Episode steps reach the max!")
            print("Episode duration exceeded 5s!")

        self.fish_detector.setup_get_state_flag(True)
        state_array = self.fish_detector.get_state()
        self.fish_detector.reset_pos_list()
        self.fish_detector.setup_get_state_flag(False)

        theta_avg = state_array[4]
        omega_avg = state_array[5]
        displacement_avg = state_array[6]
        velocity_avg = state_array[7]

        self.theta_avg_total = self.theta_avg_total + theta_avg

        # if self.last_theta_avg_total is None:
        #     # At the frist time 
        #     self.theta_avg_total_mono += 2  
        #     # update the omega_trend
        #     self.theta_avg_total_trend = 1 if self.theta_avg_total > 0 else (-1 if self.theta_avg_total < 0 else 0)
        #     # 1 if current angle (theta_avg) is greater than previous angle (last_theta_avg)
        #     # -1 if current angle is less than previous angle
        #     # 0 if angles are equal (no change)

        # else:
        #     theta_avg_total_direction = 1 if self.theta_avg_total > self.last_theta_avg_total else (-1 if self.theta_avg_total < self.last_theta_avg_total else 0)
        #     # 1 if current angle (theta_avg) is greater than previous angle (last_theta_avg)
        #     # -1 if current angle is less than previous angle
        #     # 0 if angles are equal (no change)

        #     if theta_avg_total_direction == self.theta_avg_total_trend:
        #         # when theta_avg is 
        #         self.theta_avg_total_mono += 2  # 趋势持续时奖励递增
        #     else:
        #         self.theta_avg_total_mono = -1 # 趋势改变时奖励递减
        #     # update the omega_trend
        #     self.theta_avg_total_trend = theta_avg_total_direction     
            
        # # update the last_theta_avg
        # self.last_theta_avg_total = self.theta_avg_total
        
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
        reward_omega = self.lambda_2*abs(omega_avg)

        if self.last_theta_avg is None:
            # At the frist time 
            self.theta_mono += 2  
            # update the theta_trend
            self.theta_trend = 1 if theta_avg > 0 else (-1 if theta_avg < 0 else 0)
            # 1 if current angle (theta_avg) is greater than previous angle (last_theta_avg)
            # -1 if current angle is less than previous angle
            # 0 if angles are equal (no change)

        else:
            theta_direction = 1 if theta_avg > self.last_theta_avg else (-1 if theta_avg < self.last_theta_avg else 0)
            # 1 if current angle (theta_avg) is greater than previous angle (last_theta_avg)
            # -1 if current angle is less than previous angle
            # 0 if angles are equal (no change)
            if theta_direction != 0:
                if theta_direction == self.theta_trend:
                    # when theta_avg is 
                    self.theta_mono += 2  # 趋势持续时奖励递增
                else:
                    self.theta_mono = -1 # 趋势改变时奖励递减
                # update the theta_trend
                self.theta_trend = theta_direction     
            else:
                # when theta_avg does not change, the reward will be the same as the last time
                # theta_trend would not update in this case
                self.theta_mono = 1  # 趋势不变时奖励不变
        # update the last_theta_avg
        self.last_theta_avg = theta_avg

        # reward_theta = self.lambda_1*self.theta_avg_total_mono*abs(self.theta_avg_total) 
        # reward_omega = self.lambda_2*self.omega_mono*abs(omega_avg)
        # reward_dis = self.lambda_3*velocity_avg
        if not self.fish_detector.is_in_rect():
            self.done = True
            if abs(self.theta_avg_total) > 175:
                reward_pos = 10
                print("fish is out but finish!")
            else:
                reward_pos = -10
                print("fish is out of the rect!")
            reward = reward_pos 
        else:
            # if self.theta_avg_total>175:
            #     self.done = True
            #     reward_pos = 10
            #     print("fish is finish!")
            # else:
            if self.done == False:
                self.done = False
                reward_pos = 0
        # print("reward_pos = %f, reward_theta = %f, reward_omega = %f, reward_dis = %f" % (reward_pos, reward_theta, reward_omega, reward_dis))
        # reward = reward_pos + reward_theta + reward_omega - reward_dis
        # print("reward = %f" % reward)
        reward =  reward_omega
        # reward =  10
        print("elapsed_time = %f"%self.elapsed_time, "reward = %f" % reward, "theta_avg = %f" % theta_avg, "omega_avg = %f" % omega_avg, "displacement_avg = %f" % displacement_avg, "velocity_avg = %f" % velocity_avg)
        
        # if self.done == True:
        #     self.fish_control.send("CSE",None,None,None,None)
        
        mean_state_array = state_array.mean()
        std_state_array = state_array.std()
        standard_state_array = (state_array - mean_state_array) / (std_state_array)
        return standard_state_array, reward, self.done, {}
    
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
        state = self.fish_detector.get_state()
        self.fish_detector.reset_pos_list()
        self.theta_avg_total = 0
        self.last_theta_avg_total = 0
        self.theta_avg_trend = 0  # 趋势标记：1递增，-1递减，0初始状态
        self.theta_avg_mono = 1
        self.last_omega_avg = None
        self.omega_trend = 0  # 趋势标记：1递增，-1递减，0初始状态
        self.omega_mono = 1
        self.fish_detector.setup_get_state_flag(False)
        # print(state)
        # print(type(state))

        self.episode_start_time = time.time()  
        self.done = False

        state_mean = state.mean()
        state_std = state.std()
        standard_state = (state - state_mean) / (state_std)

        return standard_state

    def close(self):
        return None