import torch
import os
from collections import namedtuple
import wml_utils as wmlu
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mmcv
import glob
from mmaction.datasets.rawframe_dataset_with_bboxes import RawframeDatasetWithBBoxes
from mmaction.apis import init_recognizer,predict_recognizerv2

VideoInfo = namedtuple('VideoInfo',["dir_path","frames","label"])
'''config_file = "configs/recognition/tsm/tsm_k400_pretrained_r50_1x1x16_25e_boeoffice_rgb.py"
# download the checkpoint from model zoo and put it in `checkpoints/`
checkpoint_file =  wmlu.home_dir("ai/mldata/training_data/mmaction/work_dirs/tsm_k400_pretrained_r50_1x1x16_boeofficea/epoch_1.pth")
#  "--out" "results.pkl" "--eval" "top_k_accuracy"]
#data_dir = "/home/wj/ai/mldata/Le2i/Office/raw_frames"
#save_dir = "/home/wj/ai/mldata/Le2i/Office/Annotation_files"
data_file = "/home/wj/ai/mldata/boeoffice/train_rawframes1.txt"
#data_file = "/home/wj/ai/mldata/boeoffice/test_rawframes.txt"'''

config_file = "configs/recognition/tsm/tsm_k400_pretrained_r50_1x1x16_25e_boefalldown_rgb.py"
# download the checkpoint from model zoo and put it in `checkpoints/`
checkpoint_file =  wmlu.home_dir("ai/mldata/training_data/mmaction/work_dirs/tsm_k400_pretrained_r50_1x1x16_boefalldown/latest.pth")
data_file = "/home/wj/ai/mldata/Le2i/FallDown/test_rawframes.txt"

data_root = os.path.dirname(data_file)

# assign the desired device.
device = 'cuda:0' # or 'cpu'
device = torch.device(device)

 # build the model from a config file and a checkpoint file
config = mmcv.Config.fromfile(config_file)



model = init_recognizer(config, checkpoint_file, device=device,use_frames=True)

def read_ann_file(file_path):
    res = []
    with open(file_path,"r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if len(line)<5:
                continue
            line = line.split(" ")
            dir_path = os.path.join(data_root,line[0])
            frames = int(line[1])
            label = int(line[2])
            res.append(VideoInfo(dir_path,frames,label))

    return res

if __name__ == "__main__":
    video_infos = read_ann_file(data_file)
    num_classes = 3
    error_dict = {}
    total_error_nr = 0

    for i in range(num_classes):
        error_dict[i] = []

    for idx,x in enumerate(video_infos):
        t_data_dir = x.dir_path
        bboxes_path = t_data_dir+"_bboxes.txt"
        if not os.path.isdir(t_data_dir):
            continue
        files = glob.glob(os.path.join(t_data_dir,"*.jpg"))
        total_files_nr = x.frames
        new_files = []
        for i in range(total_files_nr):
            new_files.append(os.path.join(t_data_dir,f"img_{i+1:05d}.jpg"))
        max_scores = -1.0
        index_range = None
        results_list = []
        bboxes = RawframeDatasetWithBBoxes.read_bboxes(bboxes_path,total_files_nr)
        results = predict_recognizerv2(model, t_data_dir,bboxes=bboxes,total_frames=total_files_nr)
        if results[0] != x.label:
            error_dict[x.label].append([os.path.basename(t_data_dir),results[0]])
            total_error_nr += 1
        sys.stdout.write(f"\rProcess {idx}/{len(video_infos)}.")

    for k,v in error_dict.items():
        print(f"ERROR for label {k}:")
        wmlu.show_list(v)

    for k,v in error_dict.items():
        print(k,f"Error nr {len(v)}")
    print(f"Error {total_error_nr}/{len(video_infos)}")
