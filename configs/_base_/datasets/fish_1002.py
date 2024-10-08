dataset_info = dict(
    dataset_name='fish_1002',
    paper_info=dict(
        author='Graving, Jacob M and Chae, Daniel and Naik, Hemal and '
        'Li, Liang and Koger, Benjamin and Costelloe, Blair R and '
        'Couzin, Iain D',
        title='DeepPoseKit, a software toolkit for fast and robust '
        'animal pose estimation using deep learning',
        container='Elife',
        year='2019',
        homepage='https://github.com/jgraving/DeepPoseKit-Data',
    ),
    keypoint_info={
        0:
        dict(name='head', id=0, color=[40, 100, 40], type='', swap=''),
        1:
        dict(name='body', id=1, color=[61,61,245], type='', swap='')
        
    },
    skeleton_info={
        0: dict(link=('head', 'body'), id=0, color=[0,0,0])
    },
    joint_weights=[1.] * 2,
    sigmas=[0.025,0.025])
