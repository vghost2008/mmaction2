import torch
import mmcv
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mmaction.apis import init_recognizer,predict_recognizer

config_file = "configs/recognition/tsm/tsm_k400_pretrained_r50_1x1x16_25e_boefalldown_rgb.py"
# download the checkpoint from model zoo and put it in `checkpoints/`
checkpoint_file =  "work_dirs/tsm_k400_pretrained_r50_1x1x16_boefalldown/epoch_120.pth"
#  "--out" "results.pkl" "--eval" "top_k_accuracy"]
file_path = "/home/wj/ai/mldata/Le2i/FallDown/test_rawframes.txt"
root_dir = "/home/wj/ai/mldata/Le2i/FallDown"


# assign the desired device.
device = 'cuda:0' # or 'cpu'
device = torch.device(device)

 # build the model from a config file and a checkpoint file
config = mmcv.Config.fromfile(config_file)
config.dataset_type = 'VideoDataset'

model = init_recognizer(config, checkpoint_file, device=device)

# test a single video and show the result:
with open(file_path) as f:
    for x in f.readlines():
        x = x.split(" ")[0]
        x = os.path.join(root_dir,x)
        if os.path.exists(x):
            results = predict_recognizer(model, x,use_frames=True)
            print(results,x)
        else:
            print(f"{x} not exists.")