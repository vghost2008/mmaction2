import os
import cv2
import glob 

data_dir = "/home/wj/ai/mldata/SisFall/raw_frames1"
save_dir = "/home/wj/ai/mldata/SisFall/raw_frames1_1"
skip_frame = 135
pos = [41,840,71,1136]

for x in os.listdir(data_dir):
    cur_data_dir = os.path.join(data_dir,x)
    if not os.path.isdir(cur_data_dir):
        continue
    cur_save_dir = os.path.join(save_dir,x)
    if not os.path.exists(cur_save_dir):
        os.makedirs(cur_save_dir)
    files = glob.glob(os.path.join(cur_data_dir,"*.jpg"))
    files.sort()
    files = files[skip_frame:]
    for i,f in enumerate(files):
        save_name = f"img_{i+1:05d}.jpg"
        save_path = os.path.join(cur_save_dir,save_name)
        img = cv2.imread(f)
        img = img[pos[0]:pos[1],pos[2]:pos[3]]
        cv2.imwrite(save_path,img)
