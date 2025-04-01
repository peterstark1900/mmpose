from mmpose.apis import MMPoseInferencer
import numpy as np
import cv2
import json

'''
detect_type, 
capture_num,
input_vidoe_path, 

---

save_frame_flag,
save_json_flag
frame_output_path = None
json_output_path = None

---

my_anno_flag,
current_rect
initial_target_x
initial_target_y
drag_threshold
edit_button_color
target_button_color
detect_button_color
train_button_color
win_width
win_height
control_width

---

my_pose_cfg, 
my_pose_weights, 
my_detect_cfg, 
my_detect_weights, 
my_kpt_thr, 
my_real_num, 
my_device
'''


class FishDetector():
    def __init__(self,capture_cfg, mmpose_cfg, writer_cfg = None, anno_cfg = None):

        # Step 1: Initialize VideoCapture object
        if capture_cfg is None:
            print("Please provide the capture configuration")
            exit()
        self.detect_type = capture_cfg.get('detect_type')
        if self.detect_type == 'camera':
            self.cap = cv2.VideoCapture(capture_cfg.get('capture_num'))
            if not self.cap.isOpened():
                print("Failed to open camera!")
                exit()
        elif self.detect_type == 'video':
            self.cap = cv2.VideoCapture(capture_cfg.get('input_vidoe_path'))
            if not self.cap.isOpened():
                print("Failed to open video!")
                exit()

        
        
        # Step 2: Initialize VideoWriter object
        if writer_cfg is not None:
            if writer_cfg.get('save_frame_flag') == False:
                print("Warning: Save flag is False, no video will be saved")
            if writer_cfg.get('output_path') is None:
                print("Warning: Output path is None, no video will be saved")
                self.save_frame_flag = False
            if writer_cfg.get('save_frame_flag') == True and writer_cfg.get('output_path') is not None:
                self.save_frame_flag = writer_cfg.get('save_frame_flag')
                self.fps = self.cap.get(cv2.CAP_PROP_FPS)
                self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 编码器
                self.out = cv2.VideoWriter(writer_cfg.get('output_path'),self.fourcc,self.fps,(self.width,self.height))

            if writer_cfg.get('save_json_flag') == False:
                print("Warning: Save json flag is False, no json file will be saved")
            if writer_cfg.get('json_output_path') is None:
                print("Warning: Json output path is None, no json file will be saved")
                self.save_json_flag = False
            if writer_cfg.get('save_json_flag') == True and writer_cfg.get('json_output_path') is not None:
                self.save_json_flag = writer_cfg.get('save_json_flag')
                self.json_output_path = writer_cfg.get('json_output_path')
        else:
            print("Warning: Writer configuration is None, no video and json file will be saved")
            self.save_frame_flag = False
            self.save_json_flag = False

        
        # Step 3: Initialize annotation object
        if anno_cfg is not None:
            if anno_cfg.get('my_anno_flag') == False:
                print("Warning: Annotation flag is False, no annotation will be displayed")

            if anno_cfg.get('my_anno_flag') == True:

                self.current_rect = anno_cfg.get('current_rect')
                self.target_x = anno_cfg.get('target_x')
                self.target_y = anno_cfg.get('target_y')
                self.drag_threshold = anno_cfg.get('drag_threshold')
                self.edit_button_color = anno_cfg.get('edit_button_color')
                self.target_button_color = anno_cfg.get('target_button_color')
                self.detect_button_color = anno_cfg.get('detect_button_color')
                self.train_button_color = anno_cfg.get('train_button_color')

                self.win_width = anno_cfg.get('win_width')
                self.win_height = anno_cfg.get('win_height')
                self.control_width = anno_cfg.get('control_width')

                self.dargging = False
                self.selected_corner = None
                self.edit_rect_mode = False
                self.set_target_mode = False
                self.train_flag = False
                self.detect_flag = False

                self.window_name = 'Frame'
                cv2.namedWindow(self.window_name)
                cv2.setMouseCallback(self.window_name, self.mouse_callback)
        else:
            print("Warning: Annotation configuration is None, no annotation will be displayed")
            self.my_anno_flag = False

        # Step 4: Initialize MMPoseInferencer object
        if mmpose_cfg is None:
            print("Please provide the mmpose configuration")
            exit()

        self.kpt_thr = mmpose_cfg.get('my_kpt_thr')
        self.inferencer = MMPoseInferencer(
            pose2d=mmpose_cfg.get('my_pose_cfg'),
            pose2d_weights=mmpose_cfg.get('my_pose_weights'),
            det_cat_ids=[0],
            det_model=mmpose_cfg.get('my_detect_cfg'),
            det_weights=mmpose_cfg.get('my_detect_weights'),
            device=mmpose_cfg.get('my_device')
        )

        self.real_num = mmpose_cfg.get('my_real_num')

        self.key_points = []
        self.frame_stamps= []
        self.keypoint_stamp = {}
        self.frame = None

    def mouse_callback(self, event, x, y, flags, param):

        if x < (self.win_width - self.control_width):

            if self.edit_mode:
                x1, y1, x2, y2 = self.current_rect
                if event == cv2.EVENT_LBUTTONDOWN:

                    dist_to_tl = (x - x1)**2 + (y - y1)**2
                    dist_to_br = (x - x2)**2 + (y - y2)**2

                    if dist_to_tl < (self.drag_threshold*2)**2:
                        self.selected_corner = 1
                        self.dragging = not self.dragging
                        print(f"Top-left corner {'selected' if self.dragging else 'released'}")
                    elif dist_to_br < (self.drag_threshold*2)**2:
                        self.selected_corner = 2
                        self.dragging = not self.dragging
                        print(f"Bottom-right corner {'selected' if self.dragging else 'released'}")

                elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
                    if self.selected_corner == 1:
                        self.current_rect = (x, y, x2, y2)
                    elif self.selected_corner == 2:
                        self.current_rect = (x1, y1, x, y)

            if self.set_target_mode and event == cv2.EVENT_LBUTTONDOWN:
                dist_to_traget = (x - self.target_x)**2 + (y - self.target_y)**2
                if dist_to_traget < (self.drag_threshold*2)**2:
                    self.dragging = not self.dragging
                    print(f"Target point {'selected' if self.dragging else'released'}")
                elif self.dragging:
                    self.target_x, self.target_y = x, y
                    print(f"Target point moved to ({self.target_x}, {self.target_y})")

        else:
            if event == cv2.EVENT_LBUTTONDOWN:
                # 编辑按钮区域 (10,10)-(110,60)
                if 10 <= x-(self.win_width-self.control_width) <= 110 and 10 <= y <= 60:
                    self.edit_mode = not self.edit_mode
                    self.edit_button_color = (0, 255, 0) if self.edit_mode else (0, 0, 255)
                    print(f"Edit mode: {'ON' if self.edit_mode else 'OFF'}")
                # 目标按钮区域 (10,70)-(110,120)
                if 10 <= x-(self.win_width-self.control_width) <= 110 and 70 <= y <= 120:
                    self.set_target_mode = not self.set_target_mode
                    self.target_button_color = (0, 255, 0) if self.set_target_mode else (0, 0, 255)
                    print(f"Target mode: {'ON' if self.set_target_mode else 'OFF'}")
                # 检测按钮区域 (10,130)-(110,180)
                if 10 <= x-(self.win_width-self.control_width) <= 110 and 130 <= y <= 180:
                    self.detect_flag = not self.detect_flag
                    self.detect_button_color = (0, 255, 0) if self.detect_flag else (0, 0, 255)
                    print(f"Detect mode: {'ON' if self.detect_flag else 'OFF'}")    
                # 训练按钮区域 (10,190)-(110,240)   
                if 10 <= x-(self.win_width-self.control_width) <= 110 and 190 <= y <= 240:
                    self.train_flag = not self.train_flag
                    self.train_button_color = (0, 255, 0) if self.train_flag else (0, 0, 255)
                    print(f"Train mode: {'ON' if self.train_flag else 'OFF'}")
        
    def detect_in_frame(self):
        result_generator = self.inferencer(self.frame,self.kpt_thr)
        for result in result_generator:
            predictions = result['predictions'][0]
            self.key_points.append(predictions.pred_instances.keypoints)


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

    def reset_key_points(self):
        self.key_points = []
        self.keypoint_stamp = {}
    
    def update_frame_stamps(self):
        self.frame_stamps.append(self.keypoint_stamp)

    def export_frame_stamps(self):
        info = {'total_frames': len(self.frame_stamps)}
        data = {
            "frame_stamps": self.frame_stamps,
            "info": info
        }
        with open('fish-1222-demo19.json', 'w') as f:
            json.dump(data, f, indent=4)

        
    def minimun_pipeline(self):
        while True:
            # 读取视频的每一帧
            ret, ori_frame = self.cap.read()
            if not ret:
                print("No video streaming. Exiting ...")
                break
            
            self.frame = cv2.resize(ori_frame, (1920, 1080))

            if self.detect_flag:
                self.detect_in_frame()
                # 进行绘制
                self.draw_in_frame()

            # Resize video frame
            if self.my_anno_flag:
                video_frame = cv2.resize(self.frame, (self.win_width - self.control_width, self.win_height))
                # Create composite image
                combined = np.zeros((self.win_height, self.win_width, 3), dtype=np.uint8)
                combined[:, :self.win_width-self.control_width] = video_frame  # Left video area
                # Right control panel
                controls = np.zeros((self.win_height, self.control_width, 3), dtype=np.uint8)
                cv2.rectangle(controls, (10,10), (110,60), self.edit_button_color, -1)
                cv2.putText(controls, 'EDIT', (20,45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                cv2.rectangle(controls, (10,70), (110,120), self.target_button_color, -1)
                cv2.putText(controls, 'Target', (20,105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                cv2.rectangle(controls, (10,130), (110,180), self.detect_button_color, -1)
                cv2.putText(controls, 'Detect', (20,165), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                cv2.rectangle(controls, (10,190), (110,240), self.train_button_color, -1)
                cv2.putText(controls, 'Train', (20,225), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                combined[:, self.win_width-self.control_width:] = controls

                # Draw interactive rectangle
                x1, y1, x2, y2 = self.current_rect
                cv2.rectangle(combined, 
                            (min(x1,x2), min(y1,y2)),
                            (max(x1,x2), max(y1,y2)),
                            (0,255,0), 2)
                cv2.circle(combined, (x1, y1), 8, (255,0,0), -1)
                cv2.circle(combined, (x2, y2), 8, (255,0,0), -1)
                cv2.circle(combined, (self.target_x, self.target_y), 8, (0,0,255), -1)
            
            # 显示帧
            cv2.imshow(self.window_name, self.frame)
            cv2.waitKey(1)
            if self.save_frame_flag:
                self.out.write(self.frame)
            if self.save_json_flag:
                self.update_frame_stamps()
            self.reset_key_points()
            if self.detect_type == 'camera':
                # 按 'q' 键退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        # 释放视频捕获对象并关闭所有窗口
        self.cap.release()
        if self.save_frame_flag:
            self.out.release()
        
        if self.save_json_flag:
            self.export_frame_stamps()

        cv2.destroyAllWindows()

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