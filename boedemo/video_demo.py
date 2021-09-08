import sys
import os.path as osp
sys.path.append(osp.dirname(osp.dirname(__file__)))
from boedemo.get_keypoints import KPDetection
from demo_toolkit import *


class Model:
    def __init__(self):
        self.kpd = KPDetection()

    def __call__(self, img):
        ans = self.kpd(img)
        img = show_keypoints(img,ans)
        return img

if __name__ == "__main__":
    vd = VideoDemo(Model(),save_path="tmp.mp4")
    video_path = None
    if len(sys.argv)>1:
        video_path = sys.argv[1]
    vd.inference_loop(video_path)
    vd.close()