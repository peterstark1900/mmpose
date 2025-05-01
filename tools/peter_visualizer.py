import json
import cv2
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt
import os


class Visualizer():
    def __init__(self, vis_cofig_dict):
        self.output_folder = vis_cofig_dict['output_folder']
        if self.output_folder is not None:
            self.export_flag = True
        else:
            self.export_flag = False
        self.fps = vis_cofig_dict['fps']
        self.add_fps = vis_cofig_dict['add_fps']
        self.keypoints = vis_cofig_dict['keypoints']
        self.point_colors = vis_cofig_dict['point_colors']
        if len(self.keypoints) != len(self.point_colors):
            raise ValueError("The number of keypoints and the number of point colors must be the same!")
        self.line_thickness = vis_cofig_dict['line_thickness']
        self.line_colors = vis_cofig_dict['line_colors']
        self.draw_rect = vis_cofig_dict['draw_rect']
        self.rect_color = vis_cofig_dict['rect_color']
        self.rect_thickness = vis_cofig_dict['rect_thickness']
        self.draw_target = vis_cofig_dict['draw_target']
        self.target_color = vis_cofig_dict['target_color']
        self.background_color = vis_cofig_dict['background_color']
        
        # create a blank background
        self.frame = np.full((1080, 1920, 3), self.background_color, dtype=np.uint8) 
        self.window_name = 'Keypoints Visualization'

        self.data = None
        self.file_list = []
        self.folder_path = None
    
    def load_data_file(self,file_path):
        with open(file_path, 'r') as f:
            self.data = json.load(f)
        self.window_name = file_path.split('\\')[-1].split('.')[0]

    def load_data_folder(self, folder_path):
        self.folder_path = folder_path
        # self.file_list = os.listdir(folder_path)
        # self.file_list = [f for f in os.listdir(folder_path) if f != '.DS_Store']
        self.file_list = [f for f in os.listdir(folder_path) 
                     if f.endswith('.json')]
        self.file_list.sort()
        print(self.file_list)

    def merge_data(self):
        self.data = {
            'frame_stamps': [],
            'time_stamps': [],
            'state_stamps': [],
            'info': []
        }
        for file_name in self.file_list:
            file_path = os.path.join(self.folder_path, file_name)
            print(file_path)
            # with open(file_path, 'r') as f:
            with open(file_path, 'r', encoding='utf-8-sig') as f:
                print(file_path)
                temp_data = json.load(f)
                # print(temp_data['time_stamps'])
            self.data['frame_stamps'].extend(temp_data['frame_stamps'])
            self.data['time_stamps'].extend(temp_data['time_stamps'])
            self.data['state_stamps'].extend(temp_data['state_stamps'])
            self.data['info'].extend(temp_data['info'])

    def setup_video_out(self,original_json_path):
        self.vidoe_name = self.output_folder+'\\'+original_json_path.split('\\')[-1].split('.')[0] + '.mp4'
        # self.vidoe_name = self.output_folder+'/'+original_json_path.split('/')[-1].split('.')[0] + '.mp4'

        print(self.vidoe_name)
        self.video_out = cv2.VideoWriter(self.vidoe_name, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (1920, 1080))
    def calculate_avg_fps(self):
        if self.data is None:
            print("Please load the data first!")
            return
        else:
            # 将字符串时间戳转换为datetime对象
            timestamps = [datetime.strptime(ts, "%Y-%m-%d %H:%M:%S.%f") for ts in self.data['time_stamps']]
            
            # 计算总持续时间和平均帧间隔
            total_duration = (timestamps[-1] - timestamps[0]).total_seconds()
            average_interval = total_duration / (len(timestamps) - 1)
            self.fps = round(1 / average_interval, 2)
            print('fps is update to: '+str(round(1 / average_interval, 2)))  # 保留两位小数

    def draw_raw_theta(self):
        if self.data is None:
            print("Please load the data first!")
            return
        else:
            vector_lists = []
            angle_in_degree = []
            for frame_stamp in self.data['frame_stamps']:
                head_pos = frame_stamp['head']
                body_pos = frame_stamp['body']
                vector = [body_pos[0] - head_pos[0], body_pos[1] - head_pos[1]]
                vector_lists.append(vector)
            print(len(vector_lists))
            for vector in vector_lists:
                # calculate the angle between the vector and the x-axis
                angle = np.arctan2(vector[1], vector[0])
                # transform the angle to degree
                angle = np.degrees(angle)
                angle_in_degree.append(angle)
            print(len(angle_in_degree))
            # plot the angle
            plt.style.use('bmh')
            plt.figure(figsize=(16, 10), dpi=600)
            plt.plot(angle_in_degree)
            plt.title("Angle")
            plt.xlabel("frame")
            plt.ylabel("angle")
            plt.grid(True)
            plt.savefig(self.output_folder+'/'+'raw_angle.png')

    def plot_raw_duration(self):
        if self.data is None:
            print("Please load the data first!")
            return
        else:
            duration_list = []
            # 将字符串时间戳转换为datetime对象
            timestamps = [datetime.strptime(ts, "%Y-%m-%d %H:%M:%S.%f") for ts in self.data['time_stamps']]
            for i in range(len(timestamps) - 1):
                # 计算相邻时间戳之间的时间差
                duration = (timestamps[i + 1] - timestamps[i]).total_seconds()
                duration_list.append(duration)
            # 绘制时间差的折线图
            plt.style.use('bmh')
            plt.figure(figsize=(16, 10), dpi=600)
            plt.plot(duration_list)
            plt.title("Duration")
            plt.xlabel("frame")
            plt.ylabel("duration")
            plt.grid(True)
            plt.savefig(self.output_folder+'/'+'raw_duration.png')

    def calculate_state(self):
        if self.data is None:
            print("Please load the data first!")
            return
        else:
            # 计算每个状态平均角速度
            avg_theta_list = []
            avg_omega_list = []
            old_theta_list = []
            old_omega_list = []

            for state_stamp in self.data['state_stamps']:
                head_pos_list = state_stamp['head_pos_list']
                body_pos_list = state_stamp['body_pos_list']
                duration = state_stamp['duration']
                vector_lists = []
                
                # for i in range(len(head_pos_list)):
                #     head_pos = head_pos_list[i]
                #     body_pos = body_pos_list[i]
                #     vector = [body_pos[0] - head_pos[0], body_pos[1] - head_pos[1]]
                #     vector_lists.append(vector)
                # theta_total = 0
                # for vector in vector_lists:
                #     # calculate the angle between the vector and the x-axis
                #     angle = np.arctan2(vector[1], vector[0])    
                #     # transform the angle to degree
                #     angle = np.degrees(angle)
                #     theta_total += angle
                # avg_theta = theta_total / len(vector_lists)
                # avg_omega = avg_theta / duration
                # avg_theta_list.append(avg_theta)
                # avg_omega_list.append(avg_omega)

                
                h_start = head_pos_list[:2]
                ph_start_pos = (sum(x for x, y in h_start)/2, sum(y for x, y in h_start)/2)

                b_start = body_pos_list[:2]
                pb_start_pos = (sum(x for x, y in b_start)/2, sum(y for x, y in b_start)/2)

                h_end = head_pos_list[-2:]
                ph_end_pos = (sum(x for x, y in h_end)/2, sum(y for x, y in h_end)/2)

                b_end = body_pos_list[-2:]
                pb_end_pos = (sum(x for x, y in b_end)/2, sum(y for x, y in b_end)/2)

                vec_initial = [ph_start_pos[0] - pb_start_pos[0], ph_start_pos[1] - pb_start_pos[1]]
                vec_end = [ph_end_pos[0] - pb_end_pos[0], ph_end_pos[1] - pb_end_pos[1]]
                initial_rad = np.arctan2(vec_initial[1], vec_initial[0])
                end_rad = np.arctan2(vec_end[1], vec_end[0])
                theta_avg = np.degrees(end_rad) - np.degrees(initial_rad)
                avg_theta_list.append(theta_avg)
                omega_avg = theta_avg / duration
                avg_omega_list.append(omega_avg)

                half_length = len(head_pos_list) // 2
                h_start = head_pos_list[:half_length]
                ph_start_pos = (sum(x for x, y in h_start)/half_length, sum(y for x, y in h_start)/half_length)

                b_start = body_pos_list[:half_length]
                pb_start_pos = (sum(x for x, y in b_start)/half_length, sum(y for x, y in b_start)/half_length)

                h_end = head_pos_list[-half_length:]
                ph_end_pos = (sum(x for x, y in h_end)/half_length, sum(y for x, y in h_end)/half_length)

                b_end = body_pos_list[-half_length:]
                pb_end_pos = (sum(x for x, y in b_end)/half_length, sum(y for x, y in b_end)/half_length)

                vec_initial = [ph_start_pos[0] - pb_start_pos[0], ph_start_pos[1] - pb_start_pos[1]]
                vec_end = [ph_end_pos[0] - pb_end_pos[0], ph_end_pos[1] - pb_end_pos[1]]
                initial_rad = np.arctan2(vec_initial[1], vec_initial[0])
                end_rad = np.arctan2(vec_end[1], vec_end[0])
                old_theta_avg = np.degrees(end_rad) - np.degrees(initial_rad)
                old_theta_list.append(old_theta_avg)
                old_omega_avg = old_theta_avg / duration
                old_omega_list.append(old_omega_avg)





                
            
            plt.style.use('bmh')
            plt.figure(figsize=(16, 10), dpi=600)
            plt.plot(avg_theta_list)
            plt.plot(avg_omega_list)
            plt.plot(old_theta_list)
            plt.plot(old_omega_list)
            plt.legend(['avg_theta', 'avg_omega', 'old_theta', 'old_omega'])
            plt.title("State")
            plt.xlabel("step")
            plt.ylabel("value")
            plt.grid(True)
            plt.savefig(self.output_folder+'/'+'state.png')

    def draw_static_analysis(self,split_list=None,merge_list=None):
        if self.data is None:
            print("请先加载数据!")
            return
        print(len(self.data['time_stamps']))
        print(len(self.data['frame_stamps']))
        print(len(self.data['state_stamps']))
        print(len(self.data['state_stamps'][0]['body_pos_list']))
        theta_avg_list = []
        omega_avg_list = []
        displacement_list = []
        velocity_list = []
        duration_list = []
        for state_stamp in self.data['state_stamps']:
            theta_avg_list.append(state_stamp['theta_avg'])
            omega_avg_list.append(state_stamp['omega_avg'])
            displacement_list.append(state_stamp['displacement_avg'])
            velocity_list.append(state_stamp['velocity_avg'])
            duration_list.append(state_stamp['duration'])
        if split_list is not None:
            for operate in split_list:
                plt.style.use('bmh')
                plt.figure(figsize=(16, 10), dpi=600)
                if operate == 'theta':
                    plt.plot(theta_avg_list, label='theta_avg')
                    plt.title("theta_avg")
                    plt.xlabel("frame")
                    plt.ylabel("theta_avg")
                if operate == 'omega':
                    plt.plot(omega_avg_list, label='omega_avg')
                    plt.title("omega_avg")
                    plt.xlabel("frame")
                if operate == 'displacement':
                    plt.plot(displacement_list, label='displacement_avg')
                    plt.title("displacement")
                    plt.xlabel("frame")
                if operate == 'velocity':
                    plt.plot(velocity_list, label='velocity_avg')
                    plt.title("velocity")
                    plt.xlabel("frame")
                if operate == 'duration':
                    plt.plot(duration_list, label='duration')
                    plt.title("duration")
                    plt.xlabel("frame")
                
                plt.grid(True)
                plt.legend()

                if operate == 'theta':
                    plt.savefig(self.output_folder+'\\'+'theta_avg.png')
                    # plt.savefig(self.output_folder+'/'+'theta_avg.png')
                if operate == 'omega':
                    plt.savefig(self.output_folder+'\\'+'omega_avg.png')
                    # plt.savefig(self.output_folder+'/'+'omega_avg.png')
                if operate == 'displacement':
                    plt.savefig(self.output_folder+'\\'+'displacement.png')
                    # plt.savefig(self.output_folder+'/'+'displacement.png')
                if operate =='velocity':
                    plt.savefig(self.output_folder+'\\'+'velocity.png')
                    # plt.savefig(self.output_folder+'/'+'velocity.png')
                if operate == 'duration':
                    plt.savefig(self.output_folder+'\\'+'duration.png')
                    # plt.savefig(self.output_folder+'/'+'duration.png')
                # plt.close()
        if merge_list is not None:
            plt.style.use('bmh')
            plt.figure(figsize=(16, 10), dpi=600)
            for operate in merge_list:
                if operate == 'theta':
                    plt.plot(theta_avg_list, label='theta_avg')
                    plt.title("theta_avg")
                    plt.xlabel("frame")
                    plt.ylabel("theta_avg")
                if operate == 'omega':
                    plt.plot(omega_avg_list, label='omega_avg')
                    plt.title("omega_avg")
                    plt.xlabel("frame")
                if operate == 'displacement':
                    plt.plot(displacement_list, label='displacement_avg')
                    plt.title("displacement")
                    plt.xlabel("frame")
                if operate =='velocity':
                    plt.plot(velocity_list, label='velocity_avg')
                    plt.title("velocity")
                    plt.xlabel("frame")
                if operate == 'duration':
                    plt.plot(duration_list, label='duration')
                    plt.title("duration")
                    plt.xlabel("frame")

                plt.grid(True)
                plt.legend()
                # plt.savefig(self.output_folder+'/'+'total_state.png')
                plt.savefig(self.output_folder+'\\'+'total_state.png')
                # plt.close()
    def draw_trajectory(self,selected_kp=None):
        if self.data is None:
            print("请先加载数据!")
            return
        
        plt.figure(figsize=(12, 8), dpi=150)
        plt.gca().invert_yaxis()  # 反转Y轴匹配图像坐标系
        
         # 修改轨迹提取逻辑
        trajectories = {}
        if selected_kp:  # 单个关键点模式
            trajectories[selected_kp] = [frame[selected_kp] for frame in self.data['frame_stamps']]
        else:  # 原有全关键点模式
            trajectories = {k: [] for k in self.keypoints}
            for frame in self.data['frame_stamps']:
                for kp in self.keypoints:
                    trajectories[kp].append(frame[kp])
        
         # 修改绘制逻辑
        if selected_kp:  # 绘制单个关键点
            coords = trajectories[selected_kp]
            x, y = zip(*coords)
            color = [c/255 for c in self.point_colors[0][::-1]]  # 使用第一个关键点的颜色
            plt.plot(x, y, 
                color=color,
                linewidth=2,
                linestyle='-',
                alpha=0.6,
                label=f'{selected_kp}_trajectory')
            # plt.scatter(x[::10], y[::10],
            #         color=color,
            #         s=30,
            #         edgecolors='black')
             # 标记起点和终点
            plt.scatter(x[0], y[0], 
                    color=color, 
                    marker='^', 
                    s=100,
                    edgecolor='black',
                    label='start')
            plt.scatter(x[-1], y[-1],
                    color=color,
                    marker='s',
                    s=100,
                    edgecolor='black',
                    label='end')
        else:  # 原有绘制逻辑
            for i, (kp, coords) in enumerate(trajectories.items()):
                x, y = zip(*coords)
                color = [c/255 for c in self.point_colors[i][::-1]]
                plt.plot(x, y, 
                        color=color,
                        linewidth=2,
                        linestyle='-',
                        alpha=0.6,
                        label=f'{kp}_trajectory')
                # plt.scatter(x[::10], y[::10],
                #         color=color,
                #         s=30,
                #         edgecolors='black')
                plt.scatter(x[0], y[0], 
                       color=color,
                       marker='^',
                       s=100,
                       edgecolor='black',
                       zorder=3)  # 提高图层级
                plt.scatter(x[-1], y[-1],
                        color=color,
                        marker='s',
                        s=100,
                        edgecolor='black',
                        zorder=3)

        
        # 绘制目标点
        if self.draw_target:
            target = (self.data['info']['target_x'], 
                    self.data['info']['target_y'])
            target_color = [c/255 for c in self.target_color[::-1]]  # BGR转RGB
            plt.scatter(*target, s=200, marker='*',
                    color=target_color,
                    label='target loaction')
        
        plt.title("keypoints trajectory")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)
        plt.legend()
        
        filename = f'trajectory_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(filename, bbox_inches='tight')
        print(f"Trajectory has save to: {filename}")
    

    def show_animation(self):
        if self.data is None:
            print("Please load the data first!")
            return
        else:
            print("Start showing the animation!")
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

            history_trajectory = []

            for frame_stamp in self.data['frame_stamps']:

                # 记录第一个关键点（head）的位置
                current_head = frame_stamp[self.keypoints[0]]
                history_trajectory.append(current_head)
                
                # # 绘制历史轨迹（最近50帧）
                # for point in history_trajectory[-50:]:
                #     cv2.circle(self.frame, point, 2, (0, 0, 255), -1)  # 黄色小点

                # 修改为绘制连续线段（保留最近50帧）
                if len(history_trajectory) >= 2:
                    for i in range(len(history_trajectory)-1):
                        # 绘制连接线段，颜色逐渐变淡
                        alpha = 0.3 + 0.7 * (i / len(history_trajectory))
                        color = (0, 0, int(255 * alpha))
                        cv2.line(self.frame, 
                                history_trajectory[i], 
                                history_trajectory[i+1], 
                                color, 
                                thickness=2)
            
                # 绘制关键点
                for i in range(len(self.keypoints)):
                    keypoint = frame_stamp[self.keypoints[i]]
                    cv2.circle(self.frame, keypoint, 5, self.point_colors[i], -1)
                    if i < len(self.keypoints) - 1:
                        cv2.line(self.frame, keypoint, frame_stamp[self.keypoints[i+1]], self.line_colors, self.line_thickness)
                if self.draw_rect:
                    x1, y1, x2, y2  = self.data['info']['current_rect']
                    cv2.rectangle(self.frame,  (min(x1,x2), min(y1,y2)),
                    (max(x1,x2), max(y1,y2)), self.rect_color, self.rect_thickness)
                if self.draw_target:
                    target_x, target_y =self.data['info']['target_x'],self.data['info']['target_y']
                    cv2.circle(self.frame, (target_x,target_y), 5, self.target_color, -1)
                if self.add_fps:
                    cv2.putText(self.frame, str(self.fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # head = frame_stamp['head']
                # body = frame_stamp['body']
                # joint = frame_stamp['joint']
                # tail = frame_stamp['tail']
                # # 绘制头部关键点
                # cv2.circle(frame, head, 5, (0, 255, 0), -1)
                # # 绘制身体关键点
                # cv2.circle(frame, body, 5, (0, 255, 0), -1)
                # # 绘制关节关键点
                # cv2.circle(frame, joint, 5, (0, 255, 0), -1)
                # # 绘制尾部关键点
                # cv2.circle(frame, tail, 5, (0, 0, 255), -1)
                # # 用线段连接关键点
                # cv2.line(frame, head, body, (255, 0, 0), 2)
                # cv2.line(frame, body, joint, (255, 0, 0), 2)
                # cv2.line(frame, joint, tail, (255, 0, 0), 2)

                self.frame = cv2.resize(self.frame, (1920, 1080))
                cv2.imshow(self.window_name, self.frame)
                delay = int(1000 / self.fps)
                if self.export_flag:
                    self.video_out.write(self.frame)
                if cv2.waitKey(delay) & 0xFF == ord('q'):
                    break
                # clear the frame
                self.frame = np.full((1080, 1920, 3), self.background_color, dtype=np.uint8)

            cv2.destroyAllWindows()
            print("Animation is over!")

    def mini_pipeline(self,file_path):
        self.load_data_file(file_path)
        self.calculate_avg_fps()
        if self.export_flag:
            self.setup_video_out(file_path)
        self.show_animation()
        # self.draw_static_analysis()
        self.draw_trajectory(selected_kp=self.keypoints[0])

        # self.load_data_folder(file_path)
        # self.merge_data()
        # self.draw_static_analysis(split_list=['theta','omega','displacement','velocity','duration'],merge_list=['theta','omega','displacement','velocity','duration'])

    def mac_pipeline(self,file_cfg):
        # self.load_data_folder(file_cfg)
        # self.merge_data()
        # self.draw_static_analysis(split_list=['theta','omega','displacement','velocity','duration'],merge_list=['theta','omega','displacement','velocity','duration'])

        # self.load_data_file(file_cfg)
        # self.draw_static_analysis(split_list=['theta','omega','displacement','velocity','duration'],merge_list=['theta','omega','displacement','velocity','duration'])

        self.load_data_file(file_cfg)
        self.calculate_state()

        # self.load_data_folder(file_cfg)
        # self.merge_data()
        # self.calculate_state()
        
        
        # self.load_data_file(file_cfg)
        # self.draw_raw_theta()
        # self.plot_raw_duration()

        # self.load_data_file(file_cfg)
        # self.calculate_avg_fps()
        # if self.export_flag:
        #     self.setup_video_out(file_cfg)
        # self.show_animation()
        # # self.draw_static_analysis()
        # self.draw_trajectory(selected_kp=self.keypoints[0])

        
       
    # def json_to_mp4(self):
def main():

    vis_cofig_dict = {
        'output_folder': r"E:\output\debug", 
        # 'output_folder': "/Users/peter/code/debug", 
        # 'output_folder': None, 
        'fps': 30,  # 每秒帧数    
        'add_fps': False, # 是否添加fps信息
        'keypoints': ['head', 'body', 'joint', 'tail'],
        'point_colors': [(0, 0, 255), (0, 51, 102), (0, 255, 0), (204, 0, 102)],
        'line_thickness': 2,
        'line_colors': (0, 0, 0),
        'draw_rect': True,
        'rect_color': (255, 0, 0),
        'rect_thickness': 2,
        'draw_target': False,
        'target_color': (0, 0, 255),
        'background_color': (255, 255, 255)
    }

    # file_path = r"E:\output\json\2025-04-29-20-36-07_1.json"
    # file_path = r"E:\output\debug\2025-05-01-16-36-14_1.json"
    file_path = r"E:\output\debug\2025-05-01-16-37-19_4.json"
    # file_path = "/Users/peter/code/debug/2025-04-30-21-56-48_0.json"
    # file_path = "/Users/peter/code/debug/2025-04-30-21-57-45_2.json"
    # file_path ="/Users/peter/code/debug/2025-04-30-21-59-27_7.json"
    # file_path = "/Users/peter/code/debug/2025-04-30-22-00-07_9.json"
    # file_folder = "/Users/peter/code/debug"
    file_folder =r"E:\output\debug"


    
    visualizer = Visualizer(vis_cofig_dict)
    visualizer.mini_pipeline(file_path)
    # visualizer.mini_pipeline(file_folder)
    # visualizer.mac_pipeline(file_folder)
    # visualizer.mac_pipeline(file_path)
    

if __name__ == '__main__':
    main()