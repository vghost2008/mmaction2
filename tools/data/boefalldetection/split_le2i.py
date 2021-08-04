import os
import glob

MAX_FRAME_PER_SAMPLE = 250
MIN_FRAME_PER_POS_SAMPLE = 150
MIN_FRAME_PER_NEG_SAMPLE = 75

def base_name(file_path):
    return os.path.splitext(os.path.basename(file_path))[0]

def read_annotation(file_path):
    if not os.path.exists(file_path):
        print(f"ERROR: Find {file_path} faild.")
        return None,None

    with open(file_path,"r") as f:
        try:
            begin_frame = int(f.readline().strip())
            end_frame = int(f.readline().strip())
            return begin_frame,end_frame
        except Exception as e:
            print(f"Read {file_path} faild, {e}.")
            return None,None

def copy_files(src_dir,dst_dir,beg_frame,end_frame):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    for index,i in enumerate(range(beg_frame,end_frame+1)):
        src_file = os.path.join(src_dir,f"img_{i:05d}.jpg")
        dst_file = os.path.join(dst_dir,f"img_{index+1:05d}.jpg")
        os.link(src_file,dst_file)

def total_files(video_frames):
    return len(glob.glob(os.path.join(video_frames,"*.jpg")))

def save_neg_frames(video_frames,beg_frame,end_frame,save_dir):
    if end_frame<=0:
        end_frame = total_files(video_frames)
    if end_frame-beg_frame<MIN_FRAME_PER_NEG_SAMPLE:
        return
    if end_frame-beg_frame<MAX_FRAME_PER_SAMPLE:
        copy_files(video_frames,save_dir+f"_{beg_frame}_{end_frame}",beg_frame,end_frame)
    else:
        for i in range(beg_frame,end_frame+MAX_FRAME_PER_SAMPLE-1,MAX_FRAME_PER_SAMPLE):
            ef = min(i+MAX_FRAME_PER_SAMPLE,end_frame)
            if ef-i>MIN_FRAME_PER_NEG_SAMPLE:
                copy_files(video_frames, save_dir + f"_{i}_{ef}", i, ef)

def save_pos_frames(video_frames,beg_frame,end_frame,save_dir):
    total_frame = total_files(video_frames)
    beg_frame = beg_frame if beg_frame>=1 else 1
    end_frame = max(min(total_frame,beg_frame+MAX_FRAME_PER_SAMPLE),end_frame)
    save_dir = save_dir+f"_{beg_frame}_{end_frame}"
    copy_files(video_frames,save_dir,beg_frame,end_frame)


def split_one_video(video_frames,save_neg_dir,save_pos_dir):
    t_dir_name= os.path.dirname(video_frames)
    dir_name = os.path.dirname(t_dir_name)
    dir_namel0 = os.path.basename(dir_name)+"_"+os.path.basename(video_frames)
    dir_namel0 = dir_namel0.replace(" ","_")
    bn = base_name(video_frames)
    ann_path = os.path.join(dir_name,"Annotation_files",bn+".txt")
    if not os.path.exists(ann_path):
        ann_path = os.path.join(dir_name,"Annotations_files",bn+".txt")
    beg_frame,end_frame = read_annotation(ann_path)
    if beg_frame is None:
        return
    if end_frame<0:
        end_frame = -end_frame
        use_end_neg = False
    else:
        use_end_neg = True

    if end_frame<=beg_frame+1:
        save_neg_frames(video_frames,1,-1,os.path.join(save_neg_dir,dir_namel0))
    else:
        save_neg_frames(video_frames,1,beg_frame,os.path.join(save_neg_dir,dir_namel0))
        save_pos_frames(video_frames,beg_frame-25,end_frame,os.path.join(save_pos_dir,dir_namel0))
        if total_files(video_frames)-end_frame>MIN_FRAME_PER_NEG_SAMPLE and use_end_neg:
            print(f"Save off neg {video_frames} {end_frame+1}")
            save_neg_frames(video_frames, end_frame+1, -1, os.path.join(save_neg_dir, dir_namel0))

def split_vides(dir_path,save_neg_dir,save_pos_dir):
    for x in os.listdir(dir_path):
        dp = os.path.join(dir_path,x)
        if os.path.isdir(dp):
            split_one_video(dp,save_neg_dir,save_pos_dir)

def split_mvides(dir_path,save_neg_dir,save_pos_dir):
    for x in dir_path:
        split_vides(x,save_neg_dir,save_pos_dir)

if __name__ == "__main__":
    '''split_mvides([
        #"/home/wj/ai/mldata/Le2i/Coffee_room_01/raw_frames",
        #"/home/wj/ai/mldata/Le2i/Coffee_room_02/raw_frames",
        #"/home/wj/ai/mldata/Le2i/Home_01/raw_frames",
        #"/home/wj/ai/mldata/Le2i/Home_02/raw_frames",
        "/home/wj/ai/mldata/Le2i/Office/raw_frames",
        "/home/wj/ai/mldata/Le2i/Lecture_room/raw_frames",
        ],
        "/home/wj/ai/mldata/Le2i/FallDown/training/images/0_1",
                "/home/wj/ai/mldata/Le2i/FallDown/training/images/1_1",
                )'''
    '''split_mvides([
        #"/home/wj/ai/mldata/Le2i/Coffee_room_01/raw_frames",
        #"/home/wj/ai/mldata/Le2i/Coffee_room_02/raw_frames",
        #"/home/wj/ai/mldata/Le2i/Home_01/raw_frames",
        #"/home/wj/ai/mldata/Le2i/Home_02/raw_frames",
        "/home/wj/ai/mldata/Le2i/Office/raw_frames",
        "/home/wj/ai/mldata/Le2i/Lecture_room/raw_frames",
        ],
        "/home/wj/ai/mldata/Le2i/FallDown/training/images/0_1",
                "/home/wj/ai/mldata/Le2i/FallDown/training/images/1_1",
                )'''
    split_mvides([
        "/home/wj/ai/mldata/MultiCameras/raw_frames",
        ],
        "/home/wj/ai/mldata/Le2i/FallDown/training/images/0_2",
                "/home/wj/ai/mldata/Le2i/FallDown/training/images/1_2",
                )


