import os
import glob
import sys
import wml_utils as wmlu
import shutil
import cv2

def unzip_dirs(dir_path,save_dir):
    zips = wmlu.recurse_get_filepath_in_dir(dir_path,suffix=".zip")
    wmlu.create_empty_dir(save_dir,False,True)
    tmp_dir = '/tmp/zip'
    for zf in zips:
        base_name = wmlu.base_name(zf)
        cmd = f'unzip {zf} -d {tmp_dir}'
        wmlu.create_empty_dir(tmp_dir,True,True)
        os.system(cmd)
        files = wmlu.recurse_get_filepath_in_dir(tmp_dir,suffix='.avi')
        for i,f in enumerate(files):
            shutil.copy(f,os.path.join(save_dir,base_name+"_"+os.path.basename(f)))


if __name__ == "__main__":
    unzip_dirs('/home/wj/ai/mldata/MultiCameras/multi_cameras1','/home/wj/ai/mldata/MultiCameras/all_videos')
