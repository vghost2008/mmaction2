import os
from posixpath import basename
import wml_utils as wmlu

ann_data = {
    1:[1075,1158],
    2:[367,450],#######
    3:[601,675],
    4:[340,706],
    5:[320,400],
    6:[700,775],
    7:[500,600],
    8:[260,402],
    9:[631,700],
    10:[501,600],
    11:[470,550],
    12:[621,700],
    13:[826,900],
    14:[980,1090],
    15:[751,845],
    16:[880,995],
    17:[741,845],
    18:[660,760],
    19:[501,645],
    20:[540,720],
    21:[850,955],
    22:[780,900],
    23:[0,0],
    24:[0,0]

}
save_dir = "/home/wj/ai/mldata/MultiCameras/Annotation_files"
if __name__ == "__main__":
    wmlu.create_empty_dir(save_dir,False,True)
    cam_nr = 8
    for k,v in ann_data.items():
        for i in range(cam_nr):
            base_name = f"chute{k:02d}_cam{i+1}.txt"
            save_path = os.path.join(save_dir,base_name)
            with open(save_path,"w") as f:
                f.write(f"{v[0]}\n")
                f.write(f"{-v[1]}\n")

