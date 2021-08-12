import torch
import os
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = "3"

import mmcv
import sys
import glob
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mmaction.apis import init_recognizer
from mmaction.datasets.pipelines import Compose

config_file = "configs/recognition/tsm/tsm_k400_pretrained_r50_1x1x16_25e_boeoffice_rgb.py"
checkpoint_file =  "boeweights/boeoffice.pth"
data_dir = "example_videos/boeoffice"

# assign the desired device.
device = 'cuda:0' # or 'cpu'
device = torch.device(device)

 # build the model from a config file and a checkpoint file
config = mmcv.Config.fromfile(config_file)
config.dataset_type = 'RawframeDataset'


model = init_recognizer(config, checkpoint_file, device=device,use_frames=True)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
test_pipeline_cfg = [
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=16,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = Compose(test_pipeline_cfg)
def read_preprocess_data(frames):

    total_frames = len(frames)
    data = dict(
        frame_dir=frames,
        total_frames=total_frames,
        label=-1,
        start_index=1,
        filename_tmpl='img_{:05}.jpg',
        modality="RGB")

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
f_000_1888_2188 0 0.9972543
f_000_301_601 0 0.99525994
f_010_2861_3161 2 0.9996619
f_013_2378_2672 1 0.9639324
f_011_177_477 1 0.86652994
f_006_4717_4976 2 0.70645225
'''