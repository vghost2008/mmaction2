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
config_file = osp.join(pdir_path,"configs/recognition/tsm/tsm_k400_pretrained_r50_1x1x16_25e_ucf101_rgb.py")
checkpoint_file =  osp.join(pdir_path,"boeweights/boenormalaction.pth")

# assign the desired device.
device = 'cuda:0' # or 'cpu'
device = torch.device(device)

 # build the model from a config file and a checkpoint file
config = mmcv.Config.fromfile(config_file)
config.dataset_type = 'RawframeDataset'

id2name = {
1 :"ApplyEyeMakeup",
2 :"ApplyLipstick",
3 :"Archery",
4 :"BabyCrawling",
5 :"BalanceBeam",
6 :"BandMarching",
7 :"BaseballPitch",
8 :"Basketball",
9 :"BasketballDunk",
10 :"BenchPress",
11 :"Biking",
12 :"Billiards",
13 :"BlowDryHair",
14 :"BlowingCandles",
15 :"BodyWeightSquats",
16 :"Bowling",
17 :"BoxingPunchingBag",
18 :"BoxingSpeedBag",
19 :"BreastStroke",
20 :"BrushingTeeth",
21 :"CleanAndJerk",
22 :"CliffDiving",
23 :"CricketBowling",
24 :"CricketShot",
25 :"CuttingInKitchen",
26 :"Diving",
27 :"Drumming",
28 :"Fencing",
29 :"FieldHockeyPenalty",
30 :"FloorGymnastics",
31 :"FrisbeeCatch",
32 :"FrontCrawl",
33 :"GolfSwing",
34 :"Haircut",
35 :"Hammering",
36 :"HammerThrow",
37 :"HandstandPushups",
38 :"HandstandWalking",
39 :"HeadMassage",
40 :"HighJump",
41 :"HorseRace",
42 :"HorseRiding",
43 :"HulaHoop",
44 :"IceDancing",
45 :"JavelinThrow",
46 :"JugglingBalls",
47 :"JumpingJack",
48 :"JumpRope",
49 :"Kayaking",
50 :"Knitting",
51 :"LongJump",
52 :"Lunges",
53 :"MilitaryParade",
54 :"Mixing",
55 :"MoppingFloor",
56 :"Nunchucks",
57 :"ParallelBars",
58 :"PizzaTossing",
59 :"PlayingCello",
60 :"PlayingDaf",
61 :"PlayingDhol",
62 :"PlayingFlute",
63 :"PlayingGuitar",
64 :"PlayingPiano",
65 :"PlayingSitar",
66 :"PlayingTabla",
67 :"PlayingViolin",
68 :"PoleVault",
69 :"PommelHorse",
70 :"PullUps",
71 :"Punch",
72 :"PushUps",
73 :"Rafting",
74 :"RockClimbingIndoor",
75 :"RopeClimbing",
76 :"Rowing",
77 :"SalsaSpin",
78 :"ShavingBeard",
79 :"Shotput",
80 :"SkateBoarding",
81 :"Skiing",
82 :"Skijet",
83 :"SkyDiving",
84 :"SoccerJuggling",
85 :"SoccerPenalty",
86 :"StillRings",
87 :"SumoWrestling",
88 :"Surfing",
89 :"Swing",
90 :"TableTennisShot",
91 :"TaiChi",
92 :"TennisSwing",
93 :"ThrowDiscus",
94 :"TrampolineJumping",
95 :"Typing",
96 :"UnevenBars",
97 :"VolleyballSpiking",
98 :"WalkingWithDog",
99 :"WallPushups",
100 :"WritingOnBoard",
101 :"YoYo"}

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
        return id2name[labels+1]

    def __call__(self, img):
        raw_img = img
        if self.need_pred() and len(img)>99:
            print(f"Pred {self.idx}")
            with torch.no_grad():
                img = self.read_preprocess_data(img)
                scores = self.model.forward_test(img)[0]
                print(scores.shape)
                r_scores = scores
            labels = np.argsort(scores)[::-1][0]
            scores = scores[labels]
            if scores<0.4:
                text = ""
            else:
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

if __name__ == "__main__":
    vd = VideoDemo(Model(),save_path="tmp.mp4",buffer_size=100,show_video=True)
    vd.preprocess = lambda x:VideoDemo.resize_h_and_save_raw_image_preprocess(x,224)
    video_path = None
    if len(sys.argv)>1:
        video_path = sys.argv[1]
    vd.inference_loop(video_path)
    vd.close()
