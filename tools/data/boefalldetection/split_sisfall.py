import copy
import os
MAX_FRAMES = 300
MIN_FRAMES = 50
osp = os.path

labelmap = {1:1}
def read_ann(ann_path):
    if not os.path.exists(ann_path):
        return None
    res = []
    with open(ann_path) as f:
        for line in f.readlines():
            line = line.strip().split(",")
            line = [int(x) for x in line]
            if line[0] in labelmap:
                res.append([labelmap[line[0]],line[1]+1,line[2]+1])  #the org data index start with 0, change to start with 1
    beg_idx = 1
    _res = copy.deepcopy(res)
    res = []
    for x in _res:
        if x[1]-beg_idx>MIN_FRAMES:
            res.append([0,beg_idx,x[1]])
        res.append(x)
        beg_idx = x[2]+1
    return res

def __save_action(ann_data,img_dir,save_dir,type=None):
    if img_dir[-1] == "/":
        img_dir = img_dir[:-1]
    base_name = osp.basename(img_dir)
    if type is None:
        save_dir = osp.join(save_dir,f"{ann_data[0]}",base_name+f"_{ann_data[1]}_{ann_data[2]}")
    else:
        save_dir = osp.join(save_dir, f"{ann_data[0]}", type+"_"+base_name + f"_{ann_data[1]}_{ann_data[2]}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for i,idx in enumerate(range(ann_data[1],ann_data[2]+1)):
        src_name = f"img_{idx:05d}.jpg"
        dst_name = f"img_{i+1:05d}.jpg"
        src_path = os.path.join(img_dir,src_name)
        dst_path = osp.join(save_dir,dst_name)
        if osp.exists(dst_path):
            print(f"Remove {dst_path}")
            os.remove(dst_path)
        if not osp.exists(src_path):
            print(f"ERROR: scr file {src_path} not exists.")
            break
        os.link(src_path,dst_path)

def save_action(ann_data,img_dir,save_dir):
    if ann_data[2]-ann_data[1]<MIN_FRAMES:
        return
    if ann_data[2]-ann_data[1]<=MAX_FRAMES:
        return __save_action(ann_data,img_dir,save_dir)
    for i in range(ann_data[1],ann_data[2],MAX_FRAMES):
        beg_idx = i
        end_idx = min(beg_idx+MAX_FRAMES,ann_data[2])
        if end_idx-beg_idx<MIN_FRAMES:
            break
        __save_action([ann_data[0],beg_idx,end_idx],img_dir,save_dir)

def split_dir(data_dir,save_dir):
    if data_dir[-1] == "/":
        data_dir = data_dir[:-1]
    base_name = osp.basename(data_dir)
    ppath = osp.dirname(data_dir)
    ann_path = osp.join(ppath,base_name+".txt")
    anns = read_ann(ann_path)
    if anns is None:
        return
    for ann in anns:
        save_action(ann,data_dir,save_dir)

def split_dirs(data_dir,save_dir):
    for x in os.listdir(data_dir):
        cur_dir = osp.join(data_dir,x)
        if osp.isdir(cur_dir):
            split_dir(cur_dir,save_dir)

if __name__ == "__main__":
    split_dirs("/home/wj/ai/mldata/SisFall/raw_frames1_1",
               "/home/wj/ai/mldata/SisFall/split")
