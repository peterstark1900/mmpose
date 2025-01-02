


from mmpose.apis import MMPoseInferencer

import numpy as np
import cv2

def average_pos(pos_list):
    # pos_list: [[x1,y1],[x2,y2],[x3,y3],[x4,y4],[x5,y5],[x6,y6]]
    # 提取 x 坐标
    x_coords = [pos[0] for pos in pos_list]
    x_coords_array = np.array(x_coords)

    # 提取 y 坐标
    y_coords = [pos[1] for pos in pos_list]
    y_coords_array = np.array(y_coords)

    # 计算平均值
    x_mean = np.mean(x_coords_array)
    y_mean = np.mean(y_coords_array)

    return [x_mean, y_mean]



inferencer = MMPoseInferencer(
   
    
    # pose2d='/home/peter/mmpose/configs/fish_keypoints/fish-keypoints-1001.py',
    # pose2d_weights='/home/peter/mmpose/work_dirs/fish-keypoints-1001/epoch_220.pth',
    # det_cat_ids=[0],
    # det_model='/home/peter/mmdetection/configs/fish/peter-rtmdet_tiny_8xb32-300e_coco.py', 
    # det_weights='/home/peter/mmdetection/work_dirs/peter-rtmdet_tiny_8xb32-300e_coco/epoch_300.pth',
    # device='cuda:0',
    pose2d='/home/peter/mmpose/configs/fish_keypoints/fish-keypoints-1004.py',
    pose2d_weights='/home/peter/mmpose/work_dirs/fish-keypoints-1004/best_coco_AP_epoch_220.pth',
    det_cat_ids=[0],
    det_model='/home/peter/mmdetection/configs/fish/peter-rtmdet_tiny_8xb32-300e_coco.py', 
    det_weights='/home/peter/mmdetection/work_dirs/peter-rtmdet_tiny_8xb32-300e_coco/epoch_300.pth',
    device='cuda:0'
)

# img_path = '/home/peter/Desktop/Fish-Dataset/gold-fish-0921-v2-test/images/Test/frame_000673.PNG'
# img_path = '/home/peter/Desktop/Fish-Dataset/视频/第三组/白.mp4'
# img_path = '/home/peter/mmpose/data/Fish-Tracker-0924/images/Train/fish_10_frame_000019.PNG'
# img_path = '/home/peter/Desktop/Fish-Dataset/视频/第一组/45&0.5.mp4'
# img_path = '/home/peter/Desktop/Fish-Dataset/视频/第二组/亮.mp4'
# img_path = '/home/peter/Desktop/Fish-Dataset/Mix-Small/M40W10-Mix-Small.mp4'
# img_path = '/home/peter/Desktop/Fish-Dataset/Fish-1001/M50W10-v1.mp4'
img_path = '/home/peter/Desktop/Fish-Dataset/Fish-1001/goldfish2-1080-v4.mp4'
# img_path = '/home/peter/Desktop/Fish-Dataset/Fish-1001/goldfish-1080-v5.mp4'
# img_path = '/home/peter/Desktop/Fish-Dataset/Fish-1001/goldfish2.png',
# img_path = '/home/peter/Desktop/Fish-Dataset/Fish-1001/goldfish1.png',
# img_path = '/home/peter/mmpose/data/HumanArt/fes2024-v2.jpeg'



 
result_generator = inferencer(img_path, vis_out_dir='/home/peter/Desktop/Fish-Dataset/test-output-0929',kpt_thr = 0.2, draw_bbox = True,draw_heatmap = True)

# result_generator = inferencer(img_path, show=True)
# result = next(result_generator)

# results = [result for result in result_generator]
# results = []


# last_len = 0
# current_len = 0
# reset_counter = 6
# counter = 0
# period = 0
# first_time_flag = True
# for result in result_generator:
    
#     predictions = result['predictions'][0]
#     # print(predictions)  
#     key_points = []
#     bboxs = []
#     for prediction in predictions:
#         key_points.append(prediction['keypoints'])
#         bboxs.append(prediction['bbox'][0])
#     # print(len(key_points))
#     print(bboxs)
#     # print(' ')
    
#     # current_len = len(key_points)
#     # if first_time_flag == True:
#     #     first_time_flag = False
#     #     last_len = current_len
#     #     key_points_buffer = [[] for i in range(current_len)]
#     #     average_pos_list = [[]]
#     # if current_len == last_len:
#     #     print('current_len')
#     #     print(current_len)
#     #     last_len = current_len
#     #     for i in range(current_len):
#     #         key_points_buffer[i].append(key_points[i])
#     #     print(key_points)
#     #     print(key_points_buffer[0])
#     #     # print(len(key_points))
#     #     print(' ')
#     # if counter == reset_counter:
#     #     counter = 0
#     #     for temp_key_points in key_points_buffer:
#     #         # average_pos_list[period].append(average_pos(temp_key_points))
#     #         print('average_pos_list:')
#     #         # print(average_pos_list[period])
#     #         print(average_pos(temp_key_points))
#     #         period += 1  
#     #     print(' ')
#     results.append(result)
#     # counter += 1




# 打开视频文件
cap = cv2.VideoCapture(img_path)
if not cap.isOpened():
    print("无法打开视频文件")
    exit()

# 初始化 VideoWriter 对象
output_path = '/home/peter/Desktop/Fish-Dataset/test-output-0929/opencv_bbox.mp4'
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 编码器
out = cv2.VideoWriter(output_path,fourcc,fps,(width,height))

# 设置显示窗口的名称和属性
window_name = 'Video'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # 允许调整窗口大小

# 设置窗口的初始大小
initial_width = 1920
initial_height = 1080
real_num = 2
vis_period = 15
counter = 0
while True:
    # 读取视频的每一帧
    ret, frame = cap.read()
    if not ret:
        print("无法读取帧 (视频结束?). Exiting ...")
        break
   
    result_generator = inferencer(frame,kpt_thr = 0.2)
    for result in result_generator:
        predictions = result['predictions'][0]
        print(predictions) 
        print
        key_points = []
        bboxs = []
        for prediction in predictions:
            key_points.append(prediction['keypoints'])
            bboxs.append(prediction['bbox'][0])
        # print(len(key_points))
        # print(key_points)
        # print(' ')
        for bbox in bboxs:
            # 绘制边界框
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
        # 绘制关键点和线段
        if (counter % vis_period) == 0:
            if len(key_points) == real_num:
                plot_key_points = key_points
            else:
                print('The number of fish is not correct.')
        for fish in plot_key_points:
            head = tuple(map(int, fish[0]))
            tail = tuple(map(int, fish[1]))
            # 绘制头部关键点
            cv2.circle(frame, head, 5, (0, 255, 0), -1)  # 绿色圆点
            # 绘制尾部关键点
            cv2.circle(frame, tail, 5, (0, 0, 255), -1)  # 红色圆点
            # 用线段连接头部和尾部关键点
            cv2.line(frame, head, tail, (255, 0, 0), 2)  # 蓝色线段
                
            
    counter += 1
    # 调整帧的大小
    resized_frame = cv2.resize(frame, (initial_width, initial_height))

    # 显示调整后的帧
    cv2.imshow(window_name, resized_frame)

    # 写入处理后的帧到输出视频文件
    out.write(resized_frame)

    # # 显示帧
    # cv2.imshow('Video', frame)

    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放视频捕获对象并关闭所有窗口
cap.release()
out.release()
cv2.destroyAllWindows()