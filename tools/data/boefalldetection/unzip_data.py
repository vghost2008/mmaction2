import os
import glob
import sys
import wml_utils as wmlu
import shutil
import cv2

def unzip_dirs(dir_path,save_dir):
    zips = wmlu.recurse_get_filepath_in_dir(dir_path,suffix=".zip")
    tmp_dir = '/tmp/zip'
    for zf in zips:
        if zf.endswith('Camera1.zip') or zf.endswith('Camera2.zip'):
            base_name = wmlu.base_name(zf)
            cmd = f'unzip {zf} -d {tmp_dir}'
            wmlu.create_empty_dir(tmp_dir,True,True)
            os.system(cmd)
            files = wmlu.recurse_get_filepath_in_dir(tmp_dir,suffix='.jpg;;.png')
            t_save_dir = os.path.join(save_dir,base_name)
            wmlu.create_empty_dir(t_save_dir,True,True)
            files.sort()
            for i,f in enumerate(files):
                suffix = os.path.splitext(f)[-1]
                t_save_path = os.path.join(t_save_dir,f"img_{i+1:05d}.jpg")
                if suffix != ".jpg":
                    img = cv2.imread(f)
                    cv2.imwrite(t_save_path,img)
                else:
                    shutil.move(f,os.path.join(t_save_dir,f"img_{i+1:05d}.jpg"))


if __name__ == "__main__":
    unzip_dirs('/home/wj/ai/mldata/falldata1','/home/wj/ai/mldata/falldata1/rawframes')
