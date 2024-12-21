from mmpose.apis import MMPoseInferencer
import numpy as np
import cv2

# def average_pos(pos_list):
#     # pos_list: [[x1,y1],[x2,y2],[x3,y3],[x4,y4],[x5,y5],[x6,y6]]
#     # 提取 x 坐标
#     x_coords = [pos[0] for pos in pos_list]
#     x_coords_array = np.array(x_coords)

#     # 提取 y 坐标
#     y_coords = [pos[1] for pos in pos_list]
#     y_coords_array = np.array(y_coords)

#     # 计算平均值
#     x_mean = np.mean(x_coords_array)
#     y_mean = np.mean(y_coords_array)

#     return [x_mean, y_mean]



# # inferencer = MMPoseInferencer(
   
    
#     # pose2d='/home/peter/mmpose/configs/fish_keypoints/fish-keypoints-1001.py',
#     # pose2d_weights='/home/peter/mmpose/work_dirs/fish-keypoints-1001/epoch_220.pth',
#     # det_cat_ids=[0],
#     # det_model='/home/peter/mmdetection/configs/fish/peter-rtmdet_tiny_8xb32-300e_coco.py', 
#     # det_weights='/home/peter/mmdetection/work_dirs/peter-rtmdet_tiny_8xb32-300e_coco/epoch_300.pth',
#     # device='cuda:0',
#     pose2d='/home/peter/mmpose/configs/fish_keypoints/fish-keypoints-1004.py',
#     pose2d_weights='/home/peter/mmpose/work_dirs/fish-keypoints-1004/best_coco_AP_epoch_220.pth',
#     det_cat_ids=[0],
#     det_model='/home/peter/mmdetection/configs/fish/peter-rtmdet_tiny_8xb32-300e_coco.py', 
#     det_weights='/home/peter/mmdetection/work_dirs/peter-rtmdet_tiny_8xb32-300e_coco/epoch_300.pth',
#     device='cuda:0'
# # )

# # img_path = '/home/peter/Desktop/Fish-Dataset/gold-fish-0921-v2-test/images/Test/frame_000673.PNG'
# # img_path = '/home/peter/Desktop/Fish-Dataset/视频/第三组/白.mp4'
# # img_path = '/home/peter/mmpose/data/Fish-Tracker-0924/images/Train/fish_10_frame_000019.PNG'
# # img_path = '/home/peter/Desktop/Fish-Dataset/视频/第一组/45&0.5.mp4'
# # img_path = '/home/peter/Desktop/Fish-Dataset/视频/第二组/亮.mp4'
# # img_path = '/home/peter/Desktop/Fish-Dataset/Mix-Small/M40W10-Mix-Small.mp4'
# # img_path = '/home/peter/Desktop/Fish-Dataset/Fish-1001/M50W10-v1.mp4'
# img_path = '/home/peter/Desktop/Fish-Dataset/Fish-1001/goldfish2-1080-v4.mp4'
# # img_path = '/home/peter/Desktop/Fish-Dataset/Fish-1001/goldfish-1080-v5.mp4'
# # img_path = '/home/peter/Desktop/Fish-Dataset/Fish-1001/goldfish2.png',
# # img_path = '/home/peter/Desktop/Fish-Dataset/Fish-1001/goldfish1.png',
# # img_path = '/home/peter/mmpose/data/HumanArt/fes2024-v2.jpeg'



 
# # result_generator = inferencer(img_path, vis_out_dir='/home/peter/Desktop/Fish-Dataset/test-output-0929',kpt_thr = 0.2, draw_bbox = True,draw_heatmap = True)

# # result_generator = inferencer(img_path, show=True)
# # result = next(result_generator)

# # results = [result for result in result_generator]
# # results = []


# # last_len = 0
# # current_len = 0
# # reset_counter = 6
# # counter = 0
# # period = 0
# # first_time_flag = True
# # for result in result_generator:
    
# #     predictions = result['predictions'][0]
# #     # print(predictions)  
# #     key_points = []
# #     for prediction in predictions:
# #         key_points.append(prediction['keypoints'])
# #     print(len(key_points))
# #     print(' ')
# #     # current_len = len(key_points)
# #     # if first_time_flag == True:
# #     #     first_time_flag = False
# #     #     last_len = current_len
# #     #     key_points_buffer = [[] for i in range(current_len)]
# #     #     average_pos_list = [[]]
# #     # if current_len == last_len:
# #     #     print('current_len')
# #     #     print(current_len)
# #     #     last_len = current_len
# #     #     for i in range(current_len):
# #     #         key_points_buffer[i].append(key_points[i])
# #     #     print(key_points)
# #     #     print(key_points_buffer[0])
# #     #     # print(len(key_points))
# #     #     print(' ')
# #     # if counter == reset_counter:
# #     #     counter = 0
# #     #     for temp_key_points in key_points_buffer:
# #     #         # average_pos_list[period].append(average_pos(temp_key_points))
# #     #         print('average_pos_list:')
# #     #         # print(average_pos_list[period])
# #     #         print(average_pos(temp_key_points))
# #     #         period += 1  
# #     #     print(' ')
# #     results.append(result)
# #     # counter += 1




# # 打开视频文件
# # cap = cv2.VideoCapture(img_path)
# cap = cv2.VideoCapture(1)
# if not cap.isOpened():
#     print("无法打开视频文件")
#     exit()

# # 初始化 VideoWriter 对象
# output_path = 'opencv_demo.mp4'
# fps = cap.get(cv2.CAP_PROP_FPS)
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 编码器
# out = cv2.VideoWriter(output_path,fourcc,fps,(width,height))

# # 设置显示窗口的名称和属性
# window_name = 'Video'
# cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # 允许调整窗口大小

# # 设置窗口的初始大小
# initial_width = 1920
# initial_height = 1080
# real_num = 2
# vis_period = 10
# counter = 0
# while True:
#     # 读取视频的每一帧
#     ret, frame = cap.read()
#     if not ret:
#         print("无法读取帧 (视频结束?). Exiting ...")
#         break


#     # if (counter % vis_period) == 0:
#     #     result_generator = inferencer(frame,kpt_thr = 0.2)
#     #     for result in result_generator:
#     #         predictions = result['predictions'][0]
#     #         # print(predictions)  
#     #         key_points = []
#     #         for prediction in predictions:
#     #             key_points.append(prediction['keypoints'])
#     #         # print(len(key_points))
#     #         # print(key_points)
#     #         # print(' ')
#     #         # 绘制关键点和线段
#     #         if len(key_points) == real_num:
#     #             for fish in key_points:
#     #                 head = tuple(map(int, fish[0]))
#     #                 tail = tuple(map(int, fish[1]))
#     #                 # 绘制头部关键点
#     #                 cv2.circle(frame, head, 5, (0, 255, 0), -1)  # 绿色圆点
#     #                 # 绘制尾部关键点
#     #                 cv2.circle(frame, tail, 5, (0, 0, 255), -1)  # 红色圆点
#     #                 # 用线段连接头部和尾部关键点
#     #                 cv2.line(frame, head, tail, (255, 0, 0), 2)  # 蓝色线段
#     #         else:
#     #             print('The number of fish is not correct.')
#     # counter += 1
#     # 调整帧的大小
#     resized_frame = cv2.resize(frame, (initial_width, initial_height))

#     # 显示调整后的帧
#     cv2.imshow(window_name, resized_frame)

#     # 写入处理后的帧到输出视频文件
#     out.write(frame)

#     # # 显示帧
#     # cv2.imshow('Video', frame)

#     # 按 'q' 键退出
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # 释放视频捕获对象并关闭所有窗口
# cap.release()
# out.release()
# cv2.destroyAllWindows()


class FishDetector():
    def __init__(self,detect_type, my_pose_cfg, my_pose_weights, my_detect_cfg, my_detect_weights, my_kpt_thr, my_real_num, my_draw_flag, my_save_flag,input_vidoe_path = None, output_path = None):
        # 初始化 VideoCapture 对象
        self.detect_type = detect_type
        if self.detect_type == 'camera':
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("无法打开摄像头")
                exit()
        elif self.detect_type == 'video':
            self.cap = cv2.VideoCapture(input_vidoe_path)
            if not self.cap.isOpened():
                print("无法打开视频文件")
                exit()
        
        #初始化保存参数
        self.save_flag = my_save_flag
        if self.save_flag:
            # 初始化 VideoWriter 对象
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 编码器
            self.out = cv2.VideoWriter(output_path,self.fourcc,self.fps,(self.width,self.height))

        # 设置显示窗口的名称和属性
        self.window_name = 'Video'
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)  # 允许调整窗口大小
        self.frame = None


        #初始化检测参数
        self.kpt_thr = my_kpt_thr
        self.inferencer = MMPoseInferencer(
            pose2d=my_pose_cfg,
            pose2d_weights=my_pose_weights,
            det_cat_ids=[0],
            det_model=my_detect_cfg,
            det_weights=my_detect_weights,
            device='cuda:0'
        )

        self.draw_flag = my_draw_flag
        self.real_num = my_real_num
        self.key_points = []

    def detect_in_frame(self):
        result_generator = self.inferencer(self.frame,self.kpt_thr)
        for result in result_generator:
            predictions = result['predictions'][0]
            # print(predictions)  
            self.key_points = []
            for prediction in predictions:
                self.key_points.append(prediction['keypoints'])
            # print(len(key_points))
            # print(key_points)
            # print(' ')

    def draw_in_frame(self):
        # 绘制关键点和线段
        if len(self.key_points) == self.real_num:
            for fish in self.key_points:
                head = tuple(map(int, fish[0]))
                body = tuple(map(int, fish[1]))
                joint = tuple(map(int, fish[2]))
                tail = tuple(map(int, fish[3]))
                # 绘制头部关键点
                cv2.circle(self.frame, head, 5, (0, 255, 0), -1)
                # 绘制身体关键点
                cv2.circle(self.frame, body, 5, (0, 255, 0), -1)
                # 绘制关节关键点
                cv2.circle(self.frame, joint, 5, (0, 255, 0), -1)
                # 绘制尾部关键点
                cv2.circle(self.frame, tail, 5, (0, 0, 255), -1)
                # 用线段连接关键点
                cv2.line(self.frame, head, body, (255, 0, 0), 2)
                cv2.line(self.frame, body, joint, (255, 0, 0), 2)
                cv2.line(self.frame, joint, tail, (255, 0, 0), 2)

        
    def frame_pipeline(self):
        while True:
            # 读取视频的每一帧
            ret, ori_frame = self.cap.read()
            if not ret:
                print("无法读取帧 (视频结束?). Exiting ...")
                break

            # 调整帧的大小
            self.frame = cv2.resize(ori_frame, (self.initial_width, self.initial_height))
            # 进行检测
            self.detect_in_frame()
            # 进行绘制
            if self.draw_flag:
                self.draw_in_frame()
            # 显示帧
            cv2.imshow(self.window_name, self.frame)

            # 写入处理后的帧到输出视频文件
            if self.save_flag:
                self.out.write(self.frame)

            if self.detect_type == 'camera':
                # 按 'q' 键退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        # 释放视频捕获对象并关闭所有窗口
        self.cap.release()
        if self.save_flag:
            self.out.release()
        cv2.destroyAllWindows()

def main():
    # detect_type = 'video'
    detect_type = 'camera'
    my_pose_cfg = '/home/peter/mmpose/configs/fish_keypoints/fish-keypoints-1004.py'
    my_pose_weights = '/home/peter/mmpose/work_dirs/fish-keypoints-1004/best_coco_AP_epoch_220.pth'
    my_detect_cfg = '/home/peter/mmdetection/configs/fish/peter-rtmdet_tiny_8xb32-300e_coco.py'
    my_detect_weights = '/home/peter/mmdetection/work_dirs/peter-rtmdet_tiny_8xb32-300e_coco/epoch_300.pth'
    my_kpt_thr = 0.2
    my_real_num = 1
    my_draw_flag = True
    my_save_flag = True
    input_vidoe_path = '/home/peter/Desktop/Fish-Dataset/Fish-1001/goldfish2-1080-v4.mp4'
    output_path = 'opencv_demo.mp4'
    fish_detector = FishDetector(detect_type, my_pose_cfg, my_pose_weights, my_detect_cfg, my_detect_weights, my_kpt_thr, my_real_num, my_draw_flag, my_save_flag, input_vidoe_path, output_path)
    fish_detector.frame_pipeline()

if __name__ == '__main__':
    main()



         