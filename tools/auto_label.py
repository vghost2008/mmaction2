import torch
import mmcv
import sys
import os
import glob
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mmaction.apis import init_recognizer,predict_recognizerv2

config_file = "configs/recognition/tsm/tsm_k400_pretrained_r50_1x1x16_25e_boefalldown_rgb.py"
# download the checkpoint from model zoo and put it in `checkpoints/`
checkpoint_file =  "work_dirs/tsm_k400_pretrained_r50_1x1x16_boefalldown/latest.pth"
#  "--out" "results.pkl" "--eval" "top_k_accuracy"]
#data_dir = "/home/wj/ai/mldata/Le2i/Office/raw_frames"
#save_dir = "/home/wj/ai/mldata/Le2i/Office/Annotation_files"
data_dir = "/home/wj/ai/mldata/MultiCameras/raw_frames"
save_dir = '/home/wj/ai/mldata/MultiCameras/Annotation_files'
FRAMS_NR=75


# assign the desired device.
device = 'cuda:0' # or 'cpu'
device = torch.device(device)

 # build the model from a config file and a checkpoint file
config = mmcv.Config.fromfile(config_file)
config.dataset_type = 'VideoDataset'


model = init_recognizer(config, checkpoint_file, device=device)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for x in os.listdir(data_dir):
    t_data_dir = os.path.join(data_dir,x)
    cur_save_path = os.path.join(save_dir,x+".txt")
    if not os.path.isdir(t_data_dir):
        continue
    files = glob.glob(os.path.join(t_data_dir,"*.jpg"))
    total_files_nr = len(files)
    new_files = []
    for i in range(total_files_nr):
        new_files.append(os.path.join(t_data_dir,f"img_{i+1:05d}.jpg"))
    max_scores = -1.0
    index_range = None
    for i in range(0,total_files_nr+FRAMS_NR-1,FRAMS_NR):
        beg_idx = i
        end_idx = beg_idx+FRAMS_NR
        if end_idx>=total_files_nr:
            break
        cur_data = new_files[beg_idx:end_idx]
        results = predict_recognizerv2(model, cur_data)
        if results[0] == 1 and results[1]>max_scores:
            index_range=(beg_idx+1,end_idx)
            max_scores = results[1]

    if index_range is not None:
        with open(cur_save_path,"w") as f:
            f.write(str(index_range[0])+"\n")
            f.write(str(index_range[1])+"\n")
        print(x,index_range)
    else:
        print(x,"0,0")
        with open(cur_save_path,"w") as f:
            f.write("0\n")
            f.write("0\n")