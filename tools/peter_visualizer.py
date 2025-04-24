import json
import cv2
import numpy as np



class Visualizer():
    def __init__(self, vis_cofig_dict):
        self.output_folder = vis_cofig_dict['output_folder']
        if self.output_folder is not None:
            self.export_flag = True
        else:
            self.export_flag = False
        self.fps = vis_cofig_dict['fps']
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
    def setup_video_out(self,original_json_path):
        self.vidoe_name = original_json_path.split('\\')[-1].split('.')[0] + '.mp4'
        self.video_out = cv2.VideoWriter(self.vidoe_name, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (1920, 1080))
    def show_animation(self):
        if self.data is None:
            print("Please load the data first!")
            return
        else:
            print("Start showing the animation!")
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            for frame_stamp in self.data['frame_stamps']:
            
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
        self.load_data(file_path)
        if self.export_flag:
            self.setup_video_out(file_path)
        self.show_animation()
            
       
    # def json_to_mp4(self):
def main():

    vis_cofig_dict = {
        # 'output_folder': r"E:\output\json", 
        'output_folder': None, 
        'fps': 30,  # 每秒帧数    
        'keypoints': ['head', 'body', 'joint', 'tail'],
        'point_colors': [(0, 0, 255), (0, 51, 102), (0, 255, 0), (204, 0, 102)],
        'line_thickness': 2,
        'line_colors': (0, 0, 255),
        'draw_rect': True,
        'rect_color': (255, 0, 0),
        'rect_thickness': 2,
        'draw_target': True,
        'target_color': (0, 0, 255),
        'background_color': (255, 255, 255)
    }

    file_path = r"E:\output\json\2025-04-24-10-13-09_1.json"
    
    visualizer = Visualizer(vis_cofig_dict)
    visualizer.mini_pipeline(file_path)

if __name__ == '__main__':
    main()