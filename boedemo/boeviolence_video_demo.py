import sys
import os.path as osp
sys.path.append(osp.dirname(osp.dirname(__file__)))
from demo_toolkit import *
import torch
import os
import numpy as np
import mmcv
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mmaction.apis import init_recognizer
from mmaction.datasets.pipelines import Compose
pdir_path = osp.dirname(osp.dirname(__file__))

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
config_file = osp.join(pdir_path,"configs/recognition/timesformer/timesformer_divST_8x32x1_15e_boeviolence_rgb.py")
checkpoint_file =  osp.join(pdir_path,"boeweights/boeviolence.pth")

# assign the desired device.
device = 'cuda:0' # or 'cpu'
device = torch.device(device)

 # build the model from a config file and a checkpoint file
config = mmcv.Config.fromfile(config_file)
config.dataset_type = 'RawframeDataset'



img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_bgr=False)
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=8,
        frame_interval=32,
        num_clips=1,
        test_mode=True),
    dict(type='GetImages'),
    dict(type='ThreeCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
test_pipeline = Compose(test_pipeline)

class Model(IntervalMode):
    def __init__(self):
        super().__init__(30)
        self.model = init_recognizer(config, checkpoint_file, device=device, use_frames=True)
        self.text_painter = BufferTextPainter(30)

    @staticmethod
    def label2text(labels):
        if labels == 1:
            return "Violence"
        return ""

    def __call__(self, img):
        raw_img = img
        if self.need_pred() and len(img)>=100:
            print(f"Pred {self.idx}")
            with torch.no_grad():
                img = self.read_preprocess_data(img)
                scores = self.model.forward_test(img)[0]
                r_scores = scores
            labels = np.argsort(scores)[::-1][0]
            scores = scores[labels]
            print(labels,r_scores)
            if labels != 0 and scores<0.65:
                labels = 0
            text = self.label2text(labels)
        else:
            text = ""
        last_img = VideoDemo.get_last_img(raw_img)
        img = self.text_painter.putText(last_img,text)

        return img

    def read_preprocess_data(self,frames):
        if isinstance(frames[0],dict):
            frames = [x['image'] for x in frames]
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

    def preprocess(self,img):
        r_img = resize_height(img,224)
        return {'image':r_img,"raw_image":img}

if __name__ == "__main__":
    model = Model()
    vd = VideoDemo(model,save_path="tmp.mp4",buffer_size=100,show_video=True)
    vd.preprocess = model.preprocess
    video_path = None
    if len(sys.argv)>1:
        video_path = sys.argv[1]
    vd.inference_loop(video_path)
    vd.close()