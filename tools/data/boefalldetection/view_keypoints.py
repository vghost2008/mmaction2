import os
import semantic.visualization_utils as svu
import wml_utils as wmlu
import sys
import pickle
import glob
import img_utils as wmli
from PIL import Image
import numpy as np

dir_path = "/home/wj/ai/mldata/Le2i/FallDown/training/images/1/MultiCameras_chute06_cam8_675_925"
save_dir = "/home/wj/ai/mldata/SisFall/tmp"
coco_part_labels = [
    'nose', 'eye_l', 'eye_r', 'ear_l', 'ear_r',
    'sho_l', 'sho_r', 'elb_l', 'elb_r', 'wri_l', 'wri_r',
    'hip_l', 'hip_r', 'kne_l', 'kne_r', 'ank_l', 'ank_r'
]
coco_part_idx = {
    b: a for a, b in enumerate(coco_part_labels)
}
coco_part_orders = [
    ('nose', 'eye_l'), ('eye_l', 'eye_r'), ('eye_r', 'nose'),
    ('eye_l', 'ear_l'), ('eye_r', 'ear_r'), ('ear_l', 'sho_l'),
    ('ear_r', 'sho_r'), ('sho_l', 'sho_r'), ('sho_l', 'hip_l'),
    ('sho_r', 'hip_r'), ('hip_l', 'hip_r'), ('sho_l', 'elb_l'),
    ('elb_l', 'wri_l'), ('sho_r', 'elb_r'), ('elb_r', 'wri_r'),
    ('hip_l', 'kne_l'), ('kne_l', 'ank_l'), ('hip_r', 'kne_r'),
    ('kne_r', 'ank_r')
]
points_pair = []
for x in coco_part_orders:
    a,b = x
    points_pair.append([coco_part_idx[a],coco_part_idx[b]])

if len(sys.argv) >= 2:
    dir_path = sys.argv[1]
if len(sys.argv)>=3:
    save_dir = sys.argv[2]

if dir_path[-1] == "/":
    dir_path = dir_path[:-1]

wmlu.create_empty_dir(save_dir,False)

pkl_path = dir_path+".pkl"

with open(pkl_path,"rb") as f:
    data = pickle.load(f)
keypoint = data['keypoint']
files = glob.glob(os.path.join(dir_path,"*.jpg"))
frames_nr = len(files)
for i in range(frames_nr):
    base_name = f"img_{i+1:05d}.jpg"
    file_path = os.path.join(dir_path,base_name)
    img = wmli.imread(file_path)
    img = svu.draw_keypoints_on_image_array(img,np.array(keypoint[i]),use_normalized_coordinates=False,points_pair=points_pair)
    img = np.array(img)
    save_path = os.path.join(save_dir,base_name)
    wmli.imwrite(save_path,img)


