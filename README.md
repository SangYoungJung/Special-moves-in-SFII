# Special moves SF2 recognition with YOLO + RNN
[![moves SF2 recognition](https://i.imgur.com/hOkH8Kz.png)](https://youtu.be/S657M5LdDDQ=0s "moves SF2 recognition")  
**Youtube Link :** [Special moves SF2 recognition with YOLO + RNN](https://youtu.be/S657M5LdDDQ=0s)  


This is a broadcasting model and tiny tool for Street Fighter2.  
First, YOLO detects characters of SF2 and feeds detected images to moves-RNN's. Second, moves-RNN's performs special moves SF2 recognition.  
1. Object Detection **YOLOv3** with Tiny weights
2. Recognition moves **RNNs** with deep many to many with ConvLSTM2D 

<img src="https://i.imgur.com/7QTjsLl.png" title="architecture"/>

### Special moves in SF2 for Ken only
* Hadoken
* Shoryuken
* Tatsumaki Senpuu Kyaku


### Dataset
* For YOLO : Labling dataset using LabelImg 
* For RNNs : Time serial dataset as detected images via YOLO


### Requirements
1. [YOLO (Windows and Linux version of Darknet )](https://github.com/AlexeyAB/darknet) 
2. [Tensorflow >= 2.x](https://www.tensorflow.org) 
3. [Keras](https://keras.io) 
4. [labelImg](https://github.com/heartexlabs/labelImg) 
5. [Python >= 3.x](https://www.python.org) 
6. [Opencv](https://opencv.org)


### Quick Start
###### Download and Preparing
* **moveSF2_download_data.py :** download images, videos, dataset and YOLO configuration from the repository.

###### Not supported YOLO
* **movesSF2_broadcast_image.py :** run broadingcasting moves SF2 recognition with prepared images.

###### Supported YOLO
* **movesSF2_broadcast_yolo.py :** run broadingcasting moves SF2 recognition with YOLOv3.

###### Build dataset
* **movesSF2_video_to_image.py :** video to image frames for YOLO training data(needs annotation) and etc.
* **movesSF2_yolo_data.py :** shuffle train and valid dataset
* **movesSF2_yolo_to_image.py :** detected object via YOLO to image sequence for RNN's data
* **movesSF2_time_series_data.py :**  make time series dataset for RNNs from detected image of YOLO

###### Training
* **movesSF2_model.ipynb :** recipe of moves recognition


### YOLO command line
###### Demo & Test
```
# Test using image
darknet.exe detector test ./yolocfg/moves-sf2.data ./yolocfg/yolov3-tiny-test-moves-sf2.cfg ./yolov3-tiny-final-sf2.weights -i 0 "./dataset/images/2022-07-14 21-05-17.mp4_1657807661_00000060.jpg" -ext_output

# Demo using video
darknet.exe detector demo ./yolocfg/moves-sf2.data ./yolocfg/yolov3-tiny-test-moves-sf2.cfg ./yolov3-tiny-final-sf2.weights -i 0 "./dataset/videos/ken_vs_zangief.mp4"
```

###### Training using pre-trained weights
```
# Training
darknet.exe detector train ./yolocfg/moves-sf2.data ./yolocfg/yolov3-tiny-train-moves-sf2.cfg ./yolov3-tiny.weights -clear
```


### To do 
- [ ] All characters and moves in SF2 is applied. 
- [ ] Supports PyTorch
- [ ] Unified model

