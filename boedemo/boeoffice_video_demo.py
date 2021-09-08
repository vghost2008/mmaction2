import sys
import os.path as osp
sys.path.append(osp.dirname(osp.dirname(__file__)))
from demo_toolkit import *
import torch
import os
import numpy as np
import mmcv
import glob
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mmaction.apis import init_recognizer
from mmaction.datasets.pipelines import Compose
pdir_path = osp.dirname(osp.dirname(__file__))

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
config_file = osp.join(pdir_path,"configs/recognition/tsm/tsm_k400_pretrained_r50_1x1x16_25e_boeoffice_rgb.py")
checkpoint_file =  osp.join(pdir_path,"boeweights/boeoffice.pth")

# assign the desired device.
device = 'cuda:0' # or 'cpu'
device = torch.device(device)

 # build the model from a config file and a checkpoint file
config = mmcv.Config.fromfile(config_file)
config.dataset_type = 'RawframeDataset'



img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
test_pipeline_cfg = [
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=16,
        test_mode=True),
    dict(type='GetImages'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = Compose(test_pipeline_cfg)

class Model(IntervalMode):
    def __init__(self):
        super().__init__(30)
        self.model = init_recognizer(config, checkpoint_file, device=device, use_frames=True)
        self.text_painter = BufferTextPainter(30)

    @staticmethod
    def label2text(labels):
        if labels == 1:
            return "Call"
        elif labels == 2:
            return "Sleep"
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
            if labels != 0 and scores<0.6:
                labels = 0
            text = self.label2text(labels)
        else:
            text = ""
        img = self.text_painter.putText(raw_img[-1],text)

        return img

    def read_preprocess_data(self,frames):
        frames = np.array(frames)
        total_frames = len(frames)
        data = dict(
            raw_images=frames,
            total_frames=total_frames,
            label=-1,
            start_index=1,
            filename_tmpl='img_{:05}.jpg',
            modality="RGB")

        data = test_pipeline(data)
        data = torch.unsqueeze(data['imgs'],0)
        data = data.to(device)
        return data

if __name__ == "__main__":
    vd = VideoDemo(Model(),save_path="tmp.mp4",buffer_size=300,show_video=True)
    vd.preprocess = lambda x:resize_height(x,224)
    video_path = None
    if len(sys.argv)>1:
        video_path = sys.argv[1]
    vd.inference_loop(video_path)
    vd.close()