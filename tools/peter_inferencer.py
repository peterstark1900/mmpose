# from mmcv.image import imread

# from mmpose.apis import inference_topdown, init_model
# from mmpose.registry import VISUALIZERS
# from mmpose.structures import merge_data_samples

# # model_cfg = 'configs/fish_keypoints/fish-keypoints-0909.py'
# model_cfg = 'configs/fish_keypoints/fish-keypoints-0922.py'

# # ckpt = 'work_dirs/fish-keypoints-0909/best_AUC_epoch_50.pth'
# ckpt = '/home/peter/mmpose/work_dirs/fish-keypoints-0922/epoch_250.pth'
# device = 'cuda'

# # 使用初始化接口构建模型
# model = init_model(model_cfg, ckpt, device=device)

# # img_path = '/home/peter/Desktop/Fish-Dataset/Mix-Small/temp1.png'
# # img_path = '/home/peter/Desktop/Fish-Dataset/test1.png'
# # img_path = '/home/peter/mmpose/data/Fish-Tracker-0908/images/Test/frame_000058.PNG'
# # img_path = '/home/peter/mmpose/data/Fish-Tracker-0908/images/Train/frame_000228.PNG'
# img_path = '/home/peter/Desktop/Fish-Dataset/gold-fish-0921-v2-test/images/Test/frame_000673.PNG'

# # 单张图片推理
# batch_results = inference_topdown(model, img_path)

# pred_instances = batch_results[0].pred_instances
# print(pred_instances.keypoints)

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
    # pose2d_weights='/home/peter/mmpose/work_dirs/fish-keypoints-1001/best_EPE_epoch_110.pth',
    # det_cat_ids=[0],
    pose2d='/home/peter/mmpose/configs/fish_keypoints/fish-keypoints-1002.py',
    pose2d_weights='/home/peter/mmpose/work_dirs/fish-keypoints-1002/best_AUC_epoch_200.pth',
    det_cat_ids=[0],
)

# img_path = '/home/peter/Desktop/Fish-Dataset/gold-fish-0921-v2-test/images/Test/frame_000673.PNG'
# img_path = '/home/peter/Desktop/Fish-Dataset/视频/第三组/白.mp4'
# img_path = '/home/peter/mmpose/data/Fish-Tracker-0924/images/Train/fish_10_frame_000019.PNG'
# img_path = '/home/peter/Desktop/Fish-Dataset/视频/第一组/45&0.5.mp4'
# img_path = '/home/peter/Desktop/Fish-Dataset/视频/第二组/亮.mp4'
# img_path = '/home/peter/Desktop/Fish-Dataset/Mix-Small/M40W10-Mix-Small.mp4'
# img_path = '/home/peter/Desktop/Fish-Dataset/Fish-1001/M50W10-v1.mp4'
img_path = '/home/peter/Desktop/Fish-Dataset/Fish-1001/goldfish2-1080-v4.mp4'



result_generator = inferencer(img_path, vis_out_dir='/home/peter/Desktop/Fish-Dataset/test-output-0929')

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