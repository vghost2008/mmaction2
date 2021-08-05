import os
import glob
import cv2

def remove_half_img(path):
    files = glob.glob(os.path.join(path,"*.jpg"))
    for f in files:
        img = cv2.imread(f)
        hw = img.shape[1]//2
        img = img[:,hw:]
        cv2.imwrite(f,img)

def remove_half_in_dirs(dir_path):
    for x in os.listdir(dir_path):
        path = os.path.join(dir_path,x)
        if os.path.isdir(path) and 'angerous' in x:
            print(f"Process {path}")
            remove_half_img(path)


if __name__ == "__main__":
    remove_half_in_dirs("/home/wj/ai/mldata/drive_and_act/phone/training/images/1")
