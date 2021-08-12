import copy
import os
MAX_FRAMES = 300
MIN_FRAMES = 100
osp = os.path

labelmap = {15:1,17:0}
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
    return res

def read_bboxes_data(path):
    if not os.path.exists(path):
        return None
    with open(path) as fp:
        res = list(fp.readlines())
        res = [x.strip() for x in res]
    return res

def __save_action(ann_data,bboxes_data,img_dir,save_dir,type=None):
    if img_dir[-1] == "/":
        img_dir = img_dir[:-1]
    base_name = osp.basename(img_dir)
    if bboxes_data is not None:
        bboxes_data = bboxes_data[ann_data[1]:ann_data[2]+1]
    if type is None:
        save_dir = osp.join(save_dir,f"{ann_data[0]}",base_name+f"_{ann_data[1]}_{ann_data[2]}")
    else:
        save_dir = osp.join(save_dir, f"{ann_data[0]}", type+"_"+base_name + f"_{ann_data[1]}_{ann_data[2]}")
    bboxes_info_path = save_dir+"_bboxes.txt"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(bboxes_info_path,"w") as fp:
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
            if bboxes_data is not None:
                if i>= len(bboxes_data):
                    bbox_data = ""
                    print(f"WARNING: loss bbox data for {dst_path}.")
                else:
                    bbox_data = bboxes_data[i]
                bbox_data = bbox_data.split(",")
                fp.write(f"{i+1}")
                if len(bbox_data)>1:
                    for x in bbox_data[1:]:
                        fp.write(","+x)
                fp.write("\n")

def save_action(ann_data,bboxes_data,img_dir,save_dir,type=None):
    if ann_data[2]-ann_data[1]<MIN_FRAMES:
        return
    if ann_data[2]-ann_data[1]<=MAX_FRAMES:
        return __save_action(ann_data,bboxes_data,img_dir,save_dir)
    for i in range(ann_data[1],ann_data[2],MAX_FRAMES):
        beg_idx = i
        end_idx = min(beg_idx+MAX_FRAMES,ann_data[2])
        if end_idx-beg_idx<MIN_FRAMES:
            break
        __save_action([ann_data[0],beg_idx,end_idx],bboxes_data,img_dir,save_dir)

def split_dir(data_dir,save_dir,type="side_view"):
    if data_dir[-1] == "/":
        data_dir = data_dir[:-1]
    base_name = osp.basename(data_dir)
    ppath = osp.dirname(osp.dirname(data_dir))
    ann_path = osp.join(ppath,type,base_name+".txt")
    bboxes_path = osp.join(osp.dirname(data_dir),base_name+"_bboxes.txt")
    bboxes_data = read_bboxes_data(bboxes_path)
    if bboxes_data is None:
        print(f"WARNING: find bboxes data {bboxes_path} faild.")
    anns = read_ann(ann_path)
    if anns is None:
        return
    for ann in anns:
        save_action(ann,bboxes_data,data_dir,save_dir,type=type)

def split_dirs(data_dir,save_dir,type):
    for x in os.listdir(data_dir):
        cur_dir = osp.join(data_dir,x)
        if osp.isdir(cur_dir):
            split_dir(cur_dir,save_dir,type)

if __name__ == "__main__":
    split_dirs("/home/wj/ai/mldata/kaggle_office_action/kaggle_office_action2/raw_frames",
               "/home/wj/ai/mldata/kaggle_office_action/kaggle_office_action2/splited1",
               "side_view")
    split_dirs("/home/wj/ai/mldata/kaggle_office_action/kaggle_office_action/raw_frames",
               "/home/wj/ai/mldata/kaggle_office_action/kaggle_office_action/splited1",
               "front_view")