import cv2
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt


import torch

import mmcv
from mmcv import imread
import mmengine
from mmengine.registry import init_default_scope

from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples

from mmdet.apis import inference_detector, init_detector

# 有 GPU 就用 GPU，没有就用 CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device', device)

img_path = '/home/peter/Desktop/Fish-Dataset/Fish-1001/goldfish2.png'

# Faster R CNN
detector = init_detector(
    '/home/peter/mmdetection/configs/fish/peter-rtmdet_tiny_8xb32-300e_coco.py',
    '/home/peter/mmdetection/work_dirs/peter-rtmdet_tiny_8xb32-300e_coco/epoch_300.pth',
    device=device
)
pose_estimator = init_pose_estimator(
    '/home/peter/mmpose/configs/fish_keypoints/fish-keypoints-1004.py',
    '/home/peter/mmpose/work_dirs/fish-keypoints-1004/epoch_420.pth',
    device=device,
    cfg_options={'model': {'test_cfg': {'output_heatmaps': True}}}
)
# pose_estimator = init_pose_estimator(
#     '/home/peter/mmpose/configs/fish_keypoints/fish-keypoints-1001.py',
#     '/home/peter/mmpose/work_dirs/fish-keypoints-1001/peter-best_EPE_epoch_140.pth',
#     device=device,
#     cfg_options={'model': {'test_cfg': {'output_heatmaps': True}}}
# )

init_default_scope(detector.cfg.get('default_scope', 'mmdet'))

# 获取目标检测预测结果
detect_result = inference_detector(detector, img_path)
print(detect_result.keys())

# 预测类别
print(detect_result.pred_instances.labels)
# 置信度
print(detect_result.pred_instances.scores)

# 置信度阈值
CONF_THRES = 0.5

pred_instance = detect_result.pred_instances.cpu().numpy()
bboxes = np.concatenate((pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
bboxes = bboxes[np.logical_and(pred_instance.labels == 0, pred_instance.scores > CONF_THRES)]
bboxes = bboxes[nms(bboxes, 0.3)][:, :4]
print('bboxes') 
print(bboxes)

# 获取每个 bbox 的关键点预测结果
pose_results = inference_topdown(pose_estimator, img_path, bboxes)

print(len(pose_results))

# 把多个bbox的pose结果打包到一起
data_samples = merge_data_samples(pose_results)
print('data_samples.keys() are:')
print(data_samples.keys())
# 每个人 17个关键点 坐标
print(data_samples.pred_instances.keypoints.shape)

# 索引为 0 的人，每个关键点的坐标
print('data_samples.pred_instances.keypoints[0,:,:]')
print(data_samples.pred_instances.keypoints[0,:,:])

# 每一类关键点的预测热力图
print(data_samples.pred_fields.heatmaps.shape)
idx_point = 0
heatmap = data_samples.pred_fields.heatmaps[idx_point,:,:]
print(heatmap.shape)
# 索引为 idx 的关键点，在全图上的预测热力图
plt.imshow(heatmap)
plt.show()

# 半径
pose_estimator.cfg.visualizer.radius = 10
# 线宽
pose_estimator.cfg.visualizer.line_width = 8
visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
# 元数据
visualizer.set_dataset_meta(pose_estimator.dataset_meta)

img = mmcv.imread(img_path)
img = mmcv.imconvert(img, 'bgr', 'rgb')

img_output = visualizer.add_datasample(
            'result',
            img,
            data_sample=data_samples,
            draw_gt=False,
            draw_heatmap=True,
            draw_bbox=True,
            show_kpt_idx=True,
            show=False,
            wait_time=0,
            out_file='outputs/B2.jpg'
)
print(img_output.shape)

plt.figure(figsize=(10,10))
plt.imshow(img_output)
plt.show()