import pickle
import os
import glob
import random

name_to_id = {"0":0,"1":1}

def __load_dir(root_dir,sub_dir,label):
    files = glob.glob(os.path.join(root_dir,sub_dir,"*.pkl"))
    datas = []
    for f in files:
        with open(f,"rb") as f:
            data = pickle.load(f)
        data['label'] = label
        data['frame_dir'] = os.path.join(sub_dir,data['frame_dir'])
        datas.append(data)
    return datas

def load_dir(root_dir,sub_dir,repeat={}):
    res = []
    for k,v in name_to_id.items():
        repeat_nr = repeat.get(k,1)
        cur_data = __load_dir(root_dir,os.path.join(sub_dir,k),v)
        cur_data = cur_data*repeat_nr
        res.extend(cur_data)
    return res

def write_file_list(save_path,datas,shuffle=False):
    if shuffle:
        random.shuffle(datas)
    with open(save_path,"wb") as f:
        pickle.dump(datas,f)

if __name__ == "__main__":
    repeat = {"1":10}
    repeat = {"1":1}
    root_dir = "/home/wj/ai/mldata/Le2i/FallDown"
    train_save_path = os.path.join(root_dir,"train_rawframes.pkl")
    test_save_path = os.path.join(root_dir,"test_rawframes.pkl")
    file_list = load_dir(root_dir,"training/images",repeat=repeat)
    write_file_list(train_save_path,file_list,True)
    file_list = load_dir(root_dir,"test/images")
    write_file_list(test_save_path,file_list)

