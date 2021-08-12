import glob
import os
import random

name_to_id = {"0":0,"1":1}

def get_file_list(dir_path,suffix="*.jpg"):
    if isinstance(suffix,list):
        files = []
        for sx in suffix:
            tfs = glob.glob(os.path.join(dir_path,sx))
            files.extend(tfs)
    else:
        files = glob.glob(os.path.join(dir_path,suffix))
    return list(files)

def get_all_type_files(root_dir_path,dir_path,suffix=["*.jpg","*.png"],repeat={}):
    res = []
    for k,v in name_to_id.items():
        l0dir_path = os.path.join(root_dir_path,dir_path,k)
        _r_nr = repeat.get(k,1)
        if not os.path.exists(l0dir_path):
            continue
        for x in os.listdir(l0dir_path):
            l1dir_path = os.path.join(l0dir_path,x)
            if not os.path.exists(l1dir_path):
                continue
            if os.path.isdir(l1dir_path):
                files = get_file_list(l1dir_path,suffix)
                res.extend([[os.path.join(dir_path,k,x),len(files),v]]*_r_nr)
    return res

def write_file_list(file_path,file_list,shuffle=True):
    with open(file_path,"w") as f:
        if shuffle:
            random.shuffle(file_list)
        for fi in file_list:
            f.write(f"{fi[0]} {fi[1]} {fi[2]}\n")

if __name__ == "__main__":
    repeat = {"1":10}
    root_dir = "/home/wj/ai/mldata/Le2i/FallDown"
    train_save_path = os.path.join(root_dir,"train_rawframes.txt")
    test_save_path = os.path.join(root_dir,"test_rawframes.txt")
    file_list = get_all_type_files(root_dir,"training/images",repeat=repeat)
    write_file_list(train_save_path,file_list)
    file_list = get_all_type_files(root_dir,"test/images")
    write_file_list(test_save_path,file_list)