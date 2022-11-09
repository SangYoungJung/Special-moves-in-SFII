# Special moves SF2 recognition with YOLO + RNN
<img src="https://i.imgur.com/hOkH8Kz.png" title="Logo"/>

This is a broadcasting model and tiny tool for Street Fighter2.  
First, YOLO detects characters of SF2 and feeds detected images to moves-RNN's. Second, moves-RNN's performs special moves SF2 recognition.  
1. Object Detection **YOLOv3** with Tiny weights
2. Recognition moves **RNNs** with deep many to many with ConvLSTM2D 

<img src="https://i.imgur.com/7QTjsLl.png" title="architecture"/>

### Spcial moves in SF2 for Ken only
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


### To do 
- [ ] All characters and moves in SF2 is applied. 
- [ ] Supports PyTorch
- [ ] Unified Tool or App
