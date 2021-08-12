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

def get_all_type_files(root_dir_path,dir_path,suffix=["*.jpg","*.png"]):
    res = []
    for k,v in name_to_id.items():
        l0dir_path = os.path.join(root_dir_path,dir_path,k)
        for x in os.listdir(l0dir_path):
            l1dir_path = os.path.join(l0dir_path,x)
            if os.path.isdir(l1dir_path):
                files = get_file_list(l1dir_path,suffix)
                res.append([os.path.join(dir_path,k,x),len(files),v])
    return res

def write_file_list(file_path,file_list,repeat={},shuffle=True):
    with open(file_path,"w") as f:
        if shuffle:
            random.shuffle(file_list)
        for fi in file_list:
            label = fi[2]
            if label in repeat:
                for _ in range(repeat[label]):
                    f.write(f"{fi[0]} {fi[1]} {fi[2]}\n")
            else:
                f.write(f"{fi[0]} {fi[1]} {fi[2]}\n")

if __name__ == "__main__":
    repeat = {1:10}
    root_dir = "/home/wj/ai/mldata/Le2i/FallDown"
    train_save_path = os.path.join(root_dir,"train_rawframes.txt")
    test_save_path = os.path.join(root_dir,"test_rawframes.txt")
    file_list = get_all_type_files(root_dir,"training/images")
    write_file_list(train_save_path,file_list,repeat=repeat)
    file_list = get_all_type_files(root_dir,"test/images")
    write_file_list(test_save_path,file_list)

