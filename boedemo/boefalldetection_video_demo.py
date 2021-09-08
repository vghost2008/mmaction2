import sys
import os.path as osp
sys.path.append(osp.dirname(osp.dirname(__file__)))
from demo_toolkit import *
import torch
import os
import numpy as np
import mmcv
from boedemo.get_keypoints import KPDetection
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mmaction.apis import init_recognizer
from mmaction.datasets.pipelines import Compose
pdir_path = osp.dirname(osp.dirname(__file__))

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
config_file = osp.join(pdir_path,"configs/skeleton/posec3d/slowonly_r50_u48_240e_boefalla_keypoint.py")
checkpoint_file =  osp.join(pdir_path,"boeweights/boefall.pth")

# assign the desired device.
device = 'cuda:0' # or 'cpu'
device = torch.device(device)

 # build the model from a config file and a checkpoint file
config = mmcv.Config.fromfile(config_file)
config.dataset_type = 'PoseDataset'
left_kp = [1, 3, 5, 7, 9, 11, 13, 15]
right_kp = [2, 4, 6, 8, 10, 12, 14, 16]
test_pipeline_cfg = [
    dict(
        type='UniformSampleFrames', clip_len=48, num_clips=1, test_mode=True,base_offset=-1),
    dict(type='MultiPersonProcess'),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(-1, 64)),
    dict(type='CenterCrop', crop_size=64),
    dict(
        type='GeneratePoseTarget',
        sigma=0.6,
        use_score=True,
        with_kp=True,
        with_limb=False,
        double=False,
        left_kp=left_kp,
        right_kp=right_kp),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]

test_pipeline = Compose(test_pipeline_cfg)

class Model(IntervalMode):
    def __init__(self):
        super().__init__(30)
        self.model = init_recognizer(config, checkpoint_file, device=device, use_frames=False)
        self.text_painter = BufferTextPainter(30)
        self.kpd = KPDetection()

    @staticmethod
    def label2text(labels):
        if labels == 1:
            return "Fall"
        return ""

    def __call__(self, img):
        raw_img = img
        if self.need_pred() and len(img)>100:
            print(f"Pred {self.idx}")
            with torch.no_grad():
                img = self.read_preprocess_data(img)
                scores = self.model.forward_test(img)[0]
                r_scores = scores
            labels = np.argsort(scores)[::-1][0]
            scores = scores[labels]
            print(labels,r_scores)
            if labels != 0 and scores<0.6:
                labels = 0
            text = self.label2text(labels)
        else:
            text = ""
        last_img = VideoDemo.get_last_img(raw_img)
        img = self.text_painter.putText(last_img,text)

        return img

    def preprocess(self,img):
        ans = self.kpd(img)
        #img = show_keypoints(img,ans)
        return {'image':img,"kp":ans}

    def read_preprocess_data(self,frames):

        total_frames = len(frames)
        keypoint = []
        keypoint_score = []
        img_shape = None
        for f in frames:
            if img_shape is None:
                img_shape = f['image'].shape[:2]
            ans = f['kp']
            keypoint.append(ans[..., :2])
            keypoint_score.append(ans[..., 2])

        data = dict(
            frame_dir=frames,
            total_frames=total_frames,
            label=-1,
            start_index=0,
            filename_tmpl='img_{:05}.jpg',
            modality="Pose",
            keypoint=keypoint,
            keypoint_score=keypoint_score,
            img_shape=img_shape,
        )

        data = test_pipeline(data)
        data = torch.unsqueeze(data['imgs'], 0)
        data = data.to(device)
        return data

if __name__ == "__main__":
    model = Model()
    vd = VideoDemo(model,save_path="tmp.mp4",buffer_size=200,show_video=True)
    vd.preprocess = model.preprocess
    video_path = None
    if len(sys.argv)>1:
        video_path = sys.argv[1]
    vd.inference_loop(video_path)
    vd.close()
