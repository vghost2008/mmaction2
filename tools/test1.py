import torch
import mmcv
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mmaction.apis import init_recognizer, inference_recognizer

config_file = "configs/recognition/tsm/tsm_k400_pretrained_video_r50_1x1x16_25e_ucf101_rgb.py"
# download the checkpoint from model zoo and put it in `checkpoints/`
checkpoint_file =  "weights/tsm_k400_pretrained_r50_1x1x16_25e_ucf101_rgb_20210630-8df9c358.pth" 
#  "--out" "results.pkl" "--eval" "top_k_accuracy"]


# assign the desired device.
device = 'cuda:0' # or 'cpu'
device = torch.device(device)

 # build the model from a config file and a checkpoint file
config = mmcv.Config.fromfile(config_file)
config.dataset_type = 'VideoDataset'

model = init_recognizer(config, checkpoint_file, device=device)

# test a single video and show the result:
video = 'demo/demo.mp4'
labels = 'demo/label_map_k400.txt'
results = inference_recognizer(model, video, labels)

# show the results
print(f'The top-5 labels with corresponding scores are:')
for result in results:
    print(f'{result[0]}: ', result[1])