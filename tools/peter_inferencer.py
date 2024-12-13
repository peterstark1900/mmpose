# from mmcv.image import imread

# from mmpose.apis import inference_topdown, init_model
# from mmpose.registry import VISUALIZERS
# from mmpose.structures import merge_data_samples

# from mmpose.apis import visualize

# # model_cfg = 'configs/fish_keypoints/fish-keypoints-0909.py'
# model_cfg = '/home/peter/mmpose/configs/body_2d_keypoint/topdown_heatmap/humanart/td-hm_ViTPose-base_8xb64-210e_humanart-256x192.py'

# # ckpt = 'work_dirs/fish-keypoints-0909/best_AUC_epoch_50.pth'
# ckpt = '/home/peter/.cache/torch/hub/checkpoints/td-hm_ViTPose-base-simple_8xb64-210e_coco-256x192-0b8234ea_20230407.pth'
# device = 'cuda'

# # 使用初始化接口构建模型
# model = init_model(model_cfg, ckpt, device=device)

# # # img_path = '/home/peter/Desktop/Fish-Dataset/Mix-Small/temp1.png'
# # # img_path = '/home/peter/Desktop/Fish-Dataset/test1.png'
# # # img_path = '/home/peter/mmpose/data/Fish-Tracker-0908/images/Test/frame_000058.PNG'
# # # img_path = '/home/peter/mmpose/data/Fish-Tracker-0908/images/Train/frame_000228.PNG'
# # img_path = '/home/peter/Desktop/Fish-Dataset/gold-fish-0921-v2-test/images/Test/frame_000673.PNG'
# img_path = '/home/peter/mmpose/data/HumanArt/fes2024.jpeg'

# # 单张图片推理
# batch_results = inference_topdown(model, img_path)

# # pred_instances = batch_results[0].pred_instances
# # print(pred_instances.keypoints)

# # 将推理结果打包
# results = merge_data_samples(batch_results)

# # 初始化可视化器
# visualizer = VISUALIZERS.build(model.cfg.visualizer)

# # 设置数据集元信息
# visualizer.set_dataset_meta(model.dataset_meta)

# img = imread(img_path, channel_order='rgb')

# # 可视化
# visualizer.add_datasample(
#     'result',
#     img,
#     data_sample=results,
#     show=True)


# # pred_instances = batch_results[0].pred_instances

# # keypoints = pred_instances.keypoints
# # keypoint_scores = pred_instances.keypoint_scores

# # metainfo = 'config/_base_/datasets/coco.py'

# # visualize(
# #     img_path,
# #     keypoints,
# #     keypoint_scores,
# #     metainfo=metainfo,
# #     show=True)


from mmpose.apis import MMPoseInferencer

inferencer = MMPoseInferencer(
    # 假设姿态估计器是在自定义数据集上训练的
    # pose2d='/home/peter/mmpose/configs/fish_keypoints/fish-keypoints-0922.py',
    # pose2d_weights='/home/peter/mmpose/work_dirs/fish-keypoints-0922/epoch_250.pth',
    # pose2d='/home/peter/mmpose/configs/fish_keypoints/fish-keypoints-0924.py',
    # pose2d_weights='/home/peter/mmpose/work_dirs/fish-keypoints-0924/best_PCK_epoch_30.pth',
    # det_cat_ids=[0],
    # pose2d='/home/peter/mmpose/configs/fish_keypoints/fish-keypoints-0930.py',
    # pose2d_weights='/home/peter/mmpose/work_dirs/fish-keypoints-0930/best_AUC_epoch_40.pth',
    # det_cat_ids=[0],
    # pose2d='/home/peter/mmpose/configs/fish_keypoints/fish-keypoints-0932.py',
    # pose2d_weights='/home/peter/mmpose/work_dirs/fish-keypoints-0932/best_PCK_epoch_10.pth',
    # det_cat_ids=[0],
    # pose2d='/home/peter/mmpose/configs/fish_keypoints/fish-keypoints-0933.py',
    # pose2d_weights='/home/peter/mmpose/work_dirs/fish-keypoints-0933/best_AUC_epoch_210.pth',
    # det_cat_ids=[0],
    # pose2d='/home/peter/mmpose/configs/fish_keypoints/fish-keypoints-0934.py',
    # pose2d_weights='/home/peter/mmpose/work_dirs/fish-keypoints-0934/best_EPE_epoch_120.pth',
    # det_cat_ids=[0],
    # pose2d='/home/peter/mmpose/configs/fish_keypoints/fish-keypoints-1001.py',
    # pose2d_weights='/home/peter/mmpose/work_dirs/fish-keypoints-1001/epoch_220.pth',
    # det_cat_ids=[0],
    # det_model='/home/peter/mmdetection/configs/fish/peter-rtmdet_tiny_8xb32-300e_coco.py', 
    # det_weights='/home/peter/mmdetection/work_dirs/peter-rtmdet_tiny_8xb32-300e_coco/epoch_300.pth',
    # device='cuda:0',
    # pose2d='/home/peter/mmpose/configs/fish_keypoints/fish-keypoints-1002.py',
    # pose2d_weights='/home/peter/mmpose/work_dirs/fish-keypoints-1002/best_AUC_epoch_200.pth',
    # det_cat_ids=[0],
    # pose2d='/home/peter/mmpose/configs/fish_keypoints/fish-keypoints-1003.py',
    # pose2d_weights='/home/peter/mmpose/work_dirs/fish-keypoints-1003/epoch_50.pth',
    # det_cat_ids=[0],

    # pose2d='/home/peter/mmpose/configs/fish_keypoints/fish-keypoints-1004.py',
    # pose2d_weights='/home/peter/mmpose/work_dirs/fish-keypoints-1004/best_coco_AP_epoch_220.pth',
    # det_cat_ids=[0],
    # det_model='/home/peter/mmdetection/configs/fish/peter-rtmdet_tiny_8xb32-300e_coco.py', 
    # det_weights='/home/peter/mmdetection/work_dirs/peter-rtmdet_tiny_8xb32-300e_coco/epoch_300.pth',
    # device='cuda:0'
    pose2d='/home/peter/mmpose/configs/fish_keypoints/fish-keypoints-1210.py',
    pose2d_weights='/home/peter/mmpose/work_dirs/fish-keypoints-1210/best_coco_AP_epoch_340.pth',
    det_cat_ids=[0],
    det_model='/home/peter/mmdetection/configs/fish/fish1210-rtmdet_tiny_8xb32-300e_coco.py', 
    det_weights='/home/peter/mmdetection/work_dirs/fish1210-rtmdet_tiny_8xb32-300e_coco/epoch_300.pth',
    device='cuda:0'
    
    # pose2d='/home/peter/mmpose/configs/fish_keypoints/fish-keypoints-1005.py',
    # pose2d_weights='/home/peter/mmpose/work_dirs/fish-keypoints-1005/best_EPE_epoch_100.pth',
    # det_cat_ids=[0],
    # pose2d='/home/peter/mmpose/configs/fish_keypoints/fish-keypoints-1007.py',
    # pose2d_weights='/home/peter/mmpose/work_dirs/fish-keypoints-1007/best_EPE_epoch_110.pth',
    # det_cat_ids=[0],
    # det_model='rtmdet-m',
    # pose2d='/home/peter/mmpose/configs/fish_keypoints/peter-td-hm_ViTPose-base_8xb64-210e_humanart-256x192.py',
    # pose2d_weights='/home/peter/.cache/torch/hub/checkpoints/td-hm_ViTPose-base_8xb64-210e_humanart-256x192-b417f546_20230611.pth',
    # pose2d_weights='/home/peter/mmpose/work_dirs/peter-td-hm_ViTPose-base_8xb64-210e_humanart-256x192/epoch_210.pth',
    # det_cat_ids=[0],
    # 'td-hm_ViTPose-base_8xb64-210e_humanart-256x192',
    # pose2d='vitpose',


)

# img_path = '/home/peter/Desktop/Fish-Dataset/gold-fish-0921-v2-test/images/Test/frame_000673.PNG'
# img_path = '/home/peter/Desktop/Fish-Dataset/视频/第三组/白.mp4'
# img_path = '/home/peter/mmpose/data/Fish-Tracker-0924/images/Train/fish_10_frame_000019.PNG'
# img_path = '/home/peter/Desktop/Fish-Dataset/视频/第一组/45&0.5.mp4'

# img_path = '/home/peter/Desktop/Fish-Dataset/视频/第二组/亮.mp4'
img_path = '/home/peter/Desktop/Fish-Dataset/fish-1210/VID_20241210_160343.mp4'
# img_path = '/home/peter/Desktop/Fish-Dataset/fish-1210/VID_20241210_160129.mp4'
# img_path = '/home/peter/Desktop/Fish-Dataset/fish-1210/fish-1210-demo2.mp4'

# img_path = '/home/peter/Desktop/Fish-Dataset/Mix-Small/M40W10-Mix-Small.mp4'
# img_path = '/home/peter/Desktop/Fish-Dataset/Fish-1001/M50W10-v1.mp4'
# img_path = '/home/peter/Desktop/Fish-Dataset/Fish-1001/goldfish2-1080-v4.mp4'
# img_path = '/home/peter/Desktop/Fish-Dataset/Fish-1001/goldfish-1080-v5.mp4'
# img_path = '/home/peter/Desktop/Fish-Dataset/Fish-1001/goldfish2.png',
# img_path = '/home/peter/Desktop/Fish-Dataset/Fish-1001/goldfish1.png',
# img_path = '/home/peter/mmpose/data/HumanArt/fes2024-v2.jpeg'


 
# result_generator = inferencer(img_path, vis_out_dir='/home/peter/Desktop/Fish-Dataset/test-output-0929',kpt_thr = 0.2, draw_bbox = True,draw_heatmap = True)
result_generator = inferencer(img_path, vis_out_dir='/home/peter/Desktop/Fish-Dataset/fish-1210/test-output',kpt_thr = 0.2, draw_bbox = True,draw_heatmap = True)

# result_generator = inferencer(img_path, show=True)
# result = next(result_generator)

results = [result for result in result_generator]


# from mmpose.apis import MMPoseInferencer

# inferencer = MMPoseInferencer(
#     # 假设姿态估计器是在自定义数据集上训练的
#     pose2d='/home/peter/mmpose/configs/fish_keypoints/fish-keypoints-0909.py',
#     pose2d_weights='/home/peter/mmpose/work_dirs/fish-keypoints-0909/best_AUC_epoch_50.pth',
#     det_cat_ids=[0],
# )

# # img_path = '/home/peter/Desktop/Fish-Dataset/gold-fish-0921-v2-test/images/Test/frame_000673.PNG'
# img_path = '/home/peter/Desktop/Fish-Dataset/Mix-Small/M40W30-Mix-Small.mp4'

# result_generator = inferencer(img_path, vis_out_dir='/home/peter/Desktop/Fish-Dataset/test-output-0922-v1')

# # result_generator = inferencer(img_path, show=True)
# # result = next(result_generator)
# results = [result for result in result_generator]