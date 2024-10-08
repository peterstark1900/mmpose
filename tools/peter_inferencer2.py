from mmpose.apis import MMPoseInferencer

img_path = '/home/peter/mmpose/data/HumanArt/fes2024.jpeg'   # 将img_path替换给你自己的路径

# 使用模型别名创建推理器
inferencer = MMPoseInferencer('vitpose-b')

# MMPoseInferencer采用了惰性推断方法，在给定输入时创建一个预测生成器
result_generator = inferencer(img_path, vis_out_dir='/home/peter/Desktop/Fish-Dataset/test-output-0929')
result = next(result_generator)