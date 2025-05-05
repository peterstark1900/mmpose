from mmpose.apis import MMPoseInferencer
import numpy as np
import cv2
import json
import datetime
'''
predictions:
<PoseDataSample(

    META INFORMATION
    img_shape: (1080, 1920)
    input_scale: array([278.9734, 371.9645], dtype=float32)
    pad_shape: (256, 192)
    img_path: '000000.jpg'
    flip_indices: [0, 1, 2, 3]
    dataset_name: 'fish_1210'
    ori_shape: (1080, 1920)
    input_center: array([1053.9369,  689.4169], dtype=float32)
    input_size: (192, 256)
    batch_input_shape: (256, 192)

    DATA FIELDS
    pred_instances: <InstanceData(

            META INFORMATION

            DATA FIELDS
            bboxes: array([[ 942.34753,  623.7214 , 1165.5262 ,  755.1124 ]], dtype=float32)
            keypoint_scores: array([[0.31500334, 0.38769138, 1.2576126 , 0.23234713]], dtype=float32)
            keypoints_visible: array([[0.31500334, 0.38769138, 1.2576126 , 0.23234713]], dtype=float32)
            keypoints: array([[[1121.5007 ,  643.6478 ],
                        [1075.7317 ,  723.5621 ],
                        [1024.8772 ,  720.65607],
                        [ 986.37305,  739.5449 ]]], dtype=float32)
            bbox_scores: array([0.69327235], dtype=float32)
        ) at 0x1e45a326df0>
    gt_instances: <InstanceData(

            META INFORMATION

            DATA FIELDS
            bboxes: array([[ 942.34753,  623.7214 , 1165.5262 ,  755.1124 ]], dtype=float32)
            bbox_scores: array([0.69327235], dtype=float32)
            keypoints_visible: array([[[1.]]])
            bbox_scales: array([[278.9734, 371.9645]], dtype=float32)
        ) at 0x1e45a2f2c10>
) at 0x1e45a326b80>

---------------------------------
'''

class FishDetector():
    def __init__(self,detect_type, my_pose_cfg, my_pose_weights, my_detect_cfg, my_detect_weights, my_kpt_thr, my_real_num, my_draw_flag, my_save_flag,input_vidoe_path = None, output_path = None):
        # 初始化 VideoCapture 对象
        self.detect_type = detect_type
        if self.detect_type == 'camera':
            self.cap = cv2.VideoCapture(1)
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
        self.frame_stamps= []
        self.keypoint_stamp = {}
        self.time_stamp = None
        self.time_stamps = []
        self.output_json_file = output_path.split('/')[-1].split('.')[0]
        self.output_json_path = output_path+self.output_json_file+'.json'

    def detect_in_frame(self):
        result_generator = self.inferencer(self.frame,self.kpt_thr)
        for result in result_generator:
            predictions = result['predictions'][0]
            self.key_points.append(predictions.pred_instances.keypoints)
        self.time_stamp = datetime.datetime.now()


    def draw_in_frame(self):
        # 绘制关键点和线段
        if len(self.key_points) == self.real_num:
            for fish in self.key_points:
                '''example:
                [[[1121.5007   643.6478 ]
                    [1075.7317   723.5621 ]
                    [1024.8772   720.65607]
                    [ 986.37305  739.5449 ]]]
                '''
                head = tuple(map(int, fish[0][0]))
                body = tuple(map(int, fish[0][1]))
                joint = tuple(map(int, fish[0][2]))
                tail = tuple(map(int, fish[0][3]))
                # print(head)
                # print(body)
                # print(joint)
                # print(tail)
                # print('---------------------------------')

                self.keypoint_stamp = {
                "head": head,
                "body": body,
                "joint": joint,
                "tail": tail
                }


                # 绘制头部关键点
                cv2.circle(self.frame, head, 5, (0, 0, 255), -1)
                # 绘制身体关键点
                cv2.circle(self.frame, body, 5, (0, 51, 102), -1)
                # 绘制关节关键点
                cv2.circle(self.frame, joint, 5, (0, 255, 0), -1)
                # 绘制尾部关键点
                cv2.circle(self.frame, tail, 5, (204, 0, 102), -1)
                # 用线段连接关键点
                cv2.line(self.frame, head, body, (0, 0, 0), 2)
                cv2.line(self.frame, body, joint, (0, 0, 0), 2)
                cv2.line(self.frame, joint, tail, (0, 0, 0), 2)

    def reset_key_points(self):
        self.key_points = []
        self.keypoint_stamp = {}
    
    def update_frame_stamps(self):
        
        self.frame_stamps.append(self.keypoint_stamp)
        self.time_stamps.append(str(self.time_stamp)) 


    def export_frame_stamps(self):
        info = {'total_frames': len(self.frame_stamps),
                'total_time_stamps': len(self.time_stamps)}
        data = {
            "info": info,
            "frame_stamps": self.frame_stamps,
            "time_stamps":self.time_stamps
        }
        with open(self.output_json_path, 'w') as f:
            json.dump(data, f, indent=4)

        
    def frame_pipeline(self):
        while True:
            # 读取视频的每一帧
            ret, ori_frame = self.cap.read()
            if not ret:
                print("无法读取帧 (视频结束?). Exiting ...")
                break

            # 调整帧的大小
            self.frame = cv2.resize(ori_frame, (1920, 1080))
            # 进行检测
            self.detect_in_frame()
            # 进行绘制
            if self.draw_flag:
                self.draw_in_frame()
            # 显示帧
            cv2.imshow(self.window_name, self.frame)
            cv2.waitKey(1)
            # 更新frame_stamps
            self.update_frame_stamps()
            # 重置key_points
            self.reset_key_points()

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

        # 导出frame_stamps
        self.export_frame_stamps()

class Visualizer():
    def __init__(self, file_path,file_type):
        self.file_type = file_type
        if self.file_type == 'json':
            self.file_path = file_path
            with open(self.file_path, 'r') as f:
                self.data = json.load(f)
    
    def show_animation(self):
        # 创建一个空白窗口
        cv2.namedWindow('Keypoints Visualization', cv2.WINDOW_NORMAL)

        for frame_stamp in self.data['frame_stamps']:
            # 创建一个空白图像
            frame = np.zeros((1080, 1920, 3), dtype=np.uint8)* 255
            head = frame_stamp['head']
            body = frame_stamp['body']
            joint = frame_stamp['joint']
            tail = frame_stamp['tail']
            # 绘制头部关键点
            cv2.circle(frame, head, 5, (0, 255, 0), -1)
            # 绘制身体关键点
            cv2.circle(frame, body, 5, (0, 255, 0), -1)
            # 绘制关节关键点
            cv2.circle(frame, joint, 5, (0, 255, 0), -1)
            # 绘制尾部关键点
            cv2.circle(frame, tail, 5, (0, 0, 255), -1)
            # 用线段连接关键点
            cv2.line(frame, head, body, (255, 0, 0), 2)
            cv2.line(frame, body, joint, (255, 0, 0), 2)
            cv2.line(frame, joint, tail, (255, 0, 0), 2)

            # 显示图像
            cv2.imshow('Keypoints Visualization', frame)
            cv2.waitKey(10)

            # 清空图像
            frame.fill(0)

            # # 等待一段时间以显示每一帧
            # if cv2.waitKey(500) & 0xFF == ord('q'):
            #     break
        
def peter_inferencer():
    detect_type = 'video'
    # detect_type = 'camera'
    my_pose_cfg = '/home/peter/mmpose/configs/fish_keypoints/fish-keypoints-1222.py'
    my_pose_weights = '/home/peter/mmpose/work_dirs/fish-keypoints-1222/epoch_1200_0303_mmpose.pth'
    my_detect_cfg = '/home/peter/mmdetection/configs/fish/fish1222-rtmdet_tiny_8xb32-300e_coco.py'
    my_detect_weights = '/home/peter/mmdetection/work_dirs/fish1222-rtmdet_tiny_8xb32-300e_coco/epoch_1200_0303_mmdet.pth'
    my_kpt_thr = 0.2
    my_real_num = 1
    my_draw_flag = True
    my_save_flag = True
    # input_vidoe_path = '/home/peter/Desktop/Fish-Dataset/fish-1222/fish-1222-demo19.mp4'
    # input_vidoe_path = '/home/peter/Desktop/Fish-Dataset/fish-0414/0414-demo-new-1.mp4'
    # input_vidoe_path = '/home/peter/Desktop/Fish-Dataset/fish-0502/40151510-3.mp4'
    # input_vidoe_path = '/home/peter/Desktop/Fish-Dataset/fish-0502/50300540-3.mp4'
    input_vidoe_path = '/home/peter/Desktop/Fish-Dataset/fish-0502/mix16-2.mp4'

    # output_path = '/home/peter/Desktop/Fish-Dataset/fish-0502/output_40151510-3.mp4'
    # output_path = '/home/peter/Desktop/Fish-Dataset/fish-0502/output_50300540-3.mp4'
    output_path = '/home/peter/Desktop/Fish-Dataset/fish-0502/output_mix16-2.mp4'
    
    fish_detector = FishDetector(detect_type, my_pose_cfg, my_pose_weights, my_detect_cfg, my_detect_weights, my_kpt_thr, my_real_num, my_draw_flag, my_save_flag, input_vidoe_path, output_path)
    fish_detector.frame_pipeline()

def peter_visualizer():
    # file_path = 'frame_stamps.json'
    file_path = 'fish-1222-demo19.json'
    file_type = 'json'
    visualizer = Visualizer(file_path,file_type)
    visualizer.show_animation()

def main():
    peter_inferencer()
    # peter_visualizer()


if __name__ == '__main__':
    main()




         