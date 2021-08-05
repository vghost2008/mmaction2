from mmcv.runner.hooks import HOOKS,Hook
from torch.utils.tensorboard import SummaryWriter
import wtorch.summary as summary
import wtorch.utils as torchu
import numpy as np
import os

@HOOKS.register_module()
class TBSummary(Hook):
    def __init__(self,log_dir="tblog",interval=20):
        self.writer = None
        self.interval = interval
        self.log_dir = log_dir
    
    def before_run(self, runner):
        log_dir = os.path.join(runner.work_dir,self.log_dir)
        self.writer = SummaryWriter(log_dir)

    def after_iter(self, runner):
        global_step = runner.iter+runner.epoch*runner.max_iters
        #if global_step%self.interval != 1:
            #return
        mean=[123.675, 116.28, 103.53]
        std=[58.395, 57.12, 57.375]
        imgs = runner.model.module.input_imgs.cpu()
        self.writer.add_histogram("input_imgs",imgs,global_step=global_step)
        imgs = torchu.unnormalize(imgs,mean=mean,std=std)
        imgs = imgs.cpu().numpy()
        print(np.mean(imgs),np.min(imgs),np.max(imgs))
        imgs = np.clip(imgs,0,255).astype(np.uint8)
        labels = runner.model.module.input_labels.cpu().numpy()
        summary.add_video_with_label(self.writer,"input_videos",imgs,labels,global_step)
        summary.add_images_with_label(self.writer,"input_imgs",imgs[0],labels[0],global_step)
        #self.writer.add_scalar("lr",runner.current_lr(),global_step)
        self.writer.add_scalar("loss",runner.model.module.cur_losses,global_step)

    def __del__(self):
        if self.writer is not None:
            self.writer.close()
