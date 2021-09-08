import torch
import os
import numpy as np
import cv2
from boedemo.get_keypoints import KPDetection

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import mmcv
import sys
import glob
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mmaction.apis import init_recognizer
from mmaction.datasets.pipelines import Compose

config_file = "configs/skeleton/posec3d/slowonly_r50_u48_240e_boefalla_keypoint.py"
checkpoint_file =  "boeweights/boefall.pth"
data_dir = "example_videos/boefall"

# assign the desired device.
device = 'cuda:0' # or 'cpu'
device = torch.device(device)

 # build the model from a config file and a checkpoint file
config = mmcv.Config.fromfile(config_file)
config.dataset_type = 'PoseDataset'


model = init_recognizer(config, checkpoint_file, device=device,use_frames=False)
left_kp = [1, 3, 5, 7, 9, 11, 13, 15]
right_kp = [2, 4, 6, 8, 10, 12, 14, 16]

test_pipeline_cfg = [
    dict(
        type='UniformSampleFrames', clip_len=48, num_clips=1, test_mode=True,base_offset=-1),
    dict(type='MultiPersonProcess'),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(-1, 64)),
    dict(type='CenterCrop', crop_size=64),
    dict(
        type='GeneratePoseTarget',
        sigma=0.6,
        use_score=True,
        with_kp=True,
        with_limb=False,
        double=False,
        left_kp=left_kp,
        right_kp=right_kp),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
kpd = KPDetection()
test_pipeline = Compose(test_pipeline_cfg)
def read_preprocess_data(frames):

    total_frames = len(frames)
    keypoint = []
    keypoint_score = []
    img_shape = None
    for f in frames:
        img = cv2.imread(f)
        if img_shape is None:
            img_shape = img.shape[:2]
        ans = kpd(img)
        keypoint.append(ans[...,:2])
        keypoint_score.append(ans[...,2])

    data = dict(
        frame_dir=frames,
        total_frames=total_frames,
        label=-1,
        start_index=0,
        filename_tmpl='img_{:05}.jpg',
        modality="Pose",
        keypoint=keypoint,
        keypoint_score=keypoint_score,
        img_shape=img_shape,
        )

    data = test_pipeline(data)
    data = torch.unsqueeze(data['imgs'],0)
    data = data.to(device)
    return data


for x in os.listdir(data_dir):
    t_data_dir = os.path.join(data_dir,x)
    if not os.path.isdir(t_data_dir):
        continue
    files = glob.glob(os.path.join(t_data_dir,"*.jpg"))
    total_files_nr = len(files)
    new_files = []
    for i in range(total_files_nr):
        new_files.append(os.path.join(t_data_dir,f"img_{i+1:05d}.jpg"))
    data = read_preprocess_data(new_files)
    # forward the model
    with torch.no_grad():
        scores = model.forward_test(data)[0]

    labels = np.argsort(scores)[::-1][0]
    scores = scores[labels]
    print(x,labels,scores)

'''
Expected outputs:
S001C001P007R002A037_rgb 0 0.9999962
Office_video_(10)_126_269 1 0.95076156
'''
