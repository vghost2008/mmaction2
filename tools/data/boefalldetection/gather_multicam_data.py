import wml_utils as wmlu
import os

data_dir = "/home/wj/ai/mldata/FallDataset/MultiCameras/multi_cameras"
save_dir = "/home/wj/ai/mldata/FallDataset/MultiCameras/multi_cameras1"
wmlu.create_empty_dir(save_dir,False)
all_files = wmlu.recurse_get_filepath_in_dir(data_dir,suffix=".zip")
for f in all_files:
    os.link(f,os.path.join(save_dir,os.path.basename(f)))

