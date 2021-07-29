import os
import shutil
import csv
import copy
import sys
from collections import namedtuple

Label = namedtuple('Label',["file_id","start_frame","end_frame","action"])

def split_video(dir_path,save_paths,labels,min_frame_nr):
    '''

    Args:
        dir_path: data path
        save_paths: list[...] [save_path for classes 0, save path for classes 1, ...]
        labels: list[tuple(3)] [start_frame,end_frame,classes_id]

    Returns:
        None
    '''
    if dir_path[-1] == "/":
        dir_path = dir_path[:-1]
    dir_name = os.path.basename(dir_path)
    for l in labels:
        sf = l.start_frame
        ef = l.end_frame
        id = l.action

        if ef-sf+1<min_frame_nr:
            continue
        save_dir = os.path.join(save_paths[id],dir_name+f"_{sf}_{ef}")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for idx,i in enumerate(range(sf,ef+1)):
            src_file_name = f"img_{i+1:05d}.jpg"
            dst_file_name = f"img_{idx+1:05d}.jpg"
            src_file = os.path.join(dir_path,src_file_name)
            dst_file = os.path.join(save_dir,dst_file_name)
            #shutil.copy(src_file,dst_file)
            os.link(src_file,dst_file)

def get_action(action):
    if 'phone' in action:
        return 1
    else:
        return 0

def merge_label(label0,label1):
    return Label(label0.file_id,label0.start_frame,label1.end_frame,label0.action)

def check_data_length(data,max_frame_nr=150):
    res = {}
    for k,v in data.items():
        ds = []
        for l in v:
            if l.end_frame-l.start_frame>max_frame_nr:
                for i in range(l.start_frame,l.end_frame,max_frame_nr):
                    tef = i+max_frame_nr
                    if tef<l.end_frame:
                        tl = Label(l.file_id,i,tef,l.action)
                        ds.append(tl)
            else:
                ds.append(l)
        res[k] = ds
    return res

def read_annotation(file_path,max_delta=150,max_merge_delta=75,default_action=2):
    res = {}
    with open(file_path) as f:
        last_data = Label("",0,0,0)
        for i,d in enumerate(csv.reader(f)):
            if 0 == i:
                continue
            file_id = d[1]
            start_frame_id = d[3]
            end_frame_id = d[4]
            action = get_action(d[5])
            cur_data = Label(file_id,int(start_frame_id),int(end_frame_id),action)
            if file_id not in  res:
                res[file_id] = []
            if last_data.file_id == cur_data.file_id:
                if cur_data.start_frame-last_data.end_frame>max_delta:
                    new_data = Label(cur_data.file_id,last_data.end_frame+1,cur_data.start_frame-1,default_action)
                    if last_data.action == new_data.action:
                        last_data = merge_label(last_data,new_data)
                    else:
                        res[file_id].append(last_data)

                        if cur_data.action == new_data.action:
                            new_data = merge_label(new_data,cur_data)
                            last_data = new_data
                            continue
                        else:
                            res[file_id].append(new_data)
                            last_data = cur_data
                elif cur_data.start_frame-last_data.end_frame<max_merge_delta and cur_data.action==last_data.action:
                    last_data = merge_label(last_data,cur_data)
            else:
                if last_data.file_id != "":
                    res[last_data.file_id].append(last_data)
                last_data = cur_data
    res = check_data_length(res)
    return res

if __name__ == "__main__":
    base_data_path = "/home/wj/ai/smldata/drive_and_act"
    data_path = os.path.join(base_data_path,"kinect_color_rgb")
    annotation_path = os.path.join(base_data_path,"activities_3s/kinect_color/midlevel.chunks_90.csv")
    save_path = os.path.join(base_data_path,"phone_x1")
    save_paths = [os.path.join(save_path,f"{x}") for x in range(3)]
    annotations = read_annotation(annotation_path)
    for i,(k,v) in enumerate(annotations.items()):
        sys.stdout.write(f"\n process {k} {i}/{len(annotations)}.")
        data_dir = os.path.join(data_path,k)
        split_video(data_dir,save_paths,v,min_frame_nr=30)
    print(f"Finish")
