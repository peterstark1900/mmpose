import json
import cv2
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt


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
    
    def load_data(self,file_path):
        with open(file_path, 'r') as f:
            self.data = json.load(f)
        self.window_name = file_path.split('\\')[-1].split('.')[0]
    def setup_video_out(self,original_json_path):
        self.vidoe_name = self.output_folder+'\\'+original_json_path.split('\\')[-1].split('.')[0] + '.mp4'
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

    def draw_static_analysis(self):
        if self.data is None:
            print("请先加载数据!")
            return
        print(len(self.data['time_stamps']))
        print(len(self.data['frame_stamps']))
        print(len(self.data['state_stamps']))
        print(len(self.data['state_stamps'][0]['body_pos_list']))
        theta_avg_list = []
        omega_avg_list = []
        for state_stamp in self.data['state_stamps']:
            theta_avg_list.append(state_stamp['theta_avg'])
            omega_avg_list.append(state_stamp['omega_avg'])
        
        # print(self.data['state_stamps'][0]['theta_avg'])
        # # 提取时间序列和角速度数据
        # timestamps = [datetime.strptime(ts, "%Y-%m-%d %H:%M:%S.%f") 
        #             for ts in self.data['time_stamps']]
        # time_sec = [(t - timestamps[0]).total_seconds() for t in timestamps]
        
        # # 从数据中提取角速度参数
        # omega_values = [frame.get('omega_avg', 0) for frame in self.data['frame_stamps']]
        # theta_values = [frame.get('theta_avg_total', 0) for frame in self.data['frame_stamps']]

        # # 创建绘图画布
        # plt.figure(figsize=(12, 6), dpi=150)
        # plt.style.use('bmh')
        
        # # 绘制角速度曲线
        # plt.plot(time_sec, omega_values, label='角速度 (deg/s)', color='royalblue')
        
        # # 绘制角度变化曲线
        # plt.plot(time_sec, theta_values, '--', label='累计转角 (deg)', color='darkorange')
        
        # # 添加标注和样式
        # plt.title("鱼头转向运动分析")
        # plt.xlabel("时间 (秒)")
        # plt.ylabel("数值")
        # plt.grid(True)
        # plt.legend()
        
        # # 计算并显示平均角速度
        # avg_omega = np.mean(np.abs(omega_values))
        # plt.annotate(f'平均角速度: {avg_omega:.2f} deg/s', 
        #             xy=(0.7, 0.9), xycoords='axes fraction',
        #             fontsize=10, bbox=dict(boxstyle="round", fc="white"))

        # # 保存图像
        # filename = f'analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        # plt.savefig(filename, bbox_inches='tight')
        # print(f"分析图表已保存至: {filename}")
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
    
    def mini_pipeline(self,file_path):
        self.load_data(file_path)
        self.calculate_avg_fps()
        if self.export_flag:
            self.setup_video_out(file_path)
        self.show_animation()
        # self.draw_static_analysis()
        self.draw_trajectory(selected_kp=self.keypoints[0])
        
       
    # def json_to_mp4(self):
def main():

    vis_cofig_dict = {
        'output_folder': r"E:\output\json", 
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

    file_path = r"E:\output\json\2025-04-29-20-36-07_1.json"

    
    visualizer = Visualizer(vis_cofig_dict)
    visualizer.mini_pipeline(file_path)

if __name__ == '__main__':
    main()