##办公室行为视频Demo
###运行Demo

使用摄像头作为输入源

```
python boedemo/boeoffice_video_demo.py
```

指定摄像头编号

```
python boedemo/boeoffice_video_demo.py 1
```

使用文件作为输入源

```
python boedemo/boeoffice_video_demo.py ~/0day/test3p.webm
```

###测试环境

- cuda 11.1
- cudnn 11.4
- opencv 4.5.2
- OS: ubuntu 18.04/windows10
- GPU: 3090TI
- FPS: 35

##跌倒检测视频Demo

```
与办公室行为测试类似但使用脚本boedemo/boefalldetection_video_demo.py
```

##暴力检测Demo

```
与办公室行为测试类似但使用脚本 boedemo/boeviolence_video_demo.py
```

##日常行为Demo

```
与办公室行为测试类似但使用脚本 boedemo/boedailyaction_video_demo.py
```

##办公室行为在视频上进行预测

```
python tools/predict_on_frames.py
```

##输入图像格式
使用RGB顺序
视频图像需要经过以下处理

- 将视频图像均分为16份，然后从每份中随机取一桢图像
- 等比例将图像宽缩放为256
- 从图像中心裁剪出224x224的图像
- 减均值[123.675, 116.28, 103.53]
- 除以方差[58.395, 57.12, 57.375]
- 将TxHxWxC顺序转换为TxCxHxW
- 扩展为5维(增加batch_size维度)

可以参考 predict_on_frames.py

##输出格式

[B,3] 其中的值代表属于某一类的概率(B表示batch size)

##类别表

```
1:打电话
2:睡觉
0:其它情况
```

