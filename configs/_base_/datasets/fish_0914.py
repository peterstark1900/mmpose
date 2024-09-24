dataset_info = dict(
    dataset_name='fish_0914',
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
        dict(name='head', id=0, color=[250, 50, 83], type='', swap=''),
        1:
        dict(name='body', id=1, color=[250, 250, 55], type='', swap='')
        
    },
    skeleton_info={
        0: dict(link=('head', 'body'), id=0, color=[0,0,0])
    },
    joint_weights=[1.] * 2,
    sigmas=[])
