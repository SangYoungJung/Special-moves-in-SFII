import os
os.add_dll_directory('c:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.1/bin')
os.add_dll_directory(os.getcwd())
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import random
import time
import argparse
import numpy as np
import darknet
import darknet_images
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
import movesSF2_Util as util
import movesSF2_Util_Yolo as util_yolo


def main(args):
    # [[[ configuration for darknet
    random.seed(3)  # deterministic bbox colors
    network, class_names, class_colors = darknet.load_network( args.config, args.data, args.weights, batch_size=1 )
    # ]]]

    # [[[ load of pre-trained model as movesSF2
    model = tf.keras.models.load_model(args.model)
    model_config = model.get_config()
    input_time_steps = model_config["layers"][0]["config"]["batch_input_shape"][1]
    input_image_size = model_config["layers"][0]["config"]["batch_input_shape"][2:]
    print("Input shape : ", model_config["layers"][0]["config"]["batch_input_shape"])
    # ]]]
    
    cv2.namedWindow('yolo', cv2.WINDOW_NORMAL)
    cv2.namedWindow('time-steps', cv2.WINDOW_NORMAL)
    cv2.namedWindow('time-steps-result', cv2.WINDOW_NORMAL)
    queue_crops = {}
    for key in args.output_class: 
        queue_crops[key] = [ np.empty(input_image_size) for i in range(input_time_steps)]
        cv2.namedWindow(key, cv2.WINDOW_NORMAL)
    
    
    cap = cv2.VideoCapture(args.video)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret != True: continue
        
        # detect objects using yolo
        image, detections = util_yolo.video_detection(frame, network, class_names, class_colors, args.threshold, args.draw_boxes )
        cv2.imshow('yolo', image)

        # extract detections to images
        image_crop = util_yolo.video_crop(image, args.output_class, detections)
        
        # build time-steps images and show current image        
        for key in image_crop.keys():
            resize_img = cv2.resize(image_crop[key], dsize=input_image_size[:2])
            convert_img = cv2.cvtColor(resize_img, cv2.COLOR_BGR2RGB)
            queue_crops[key].pop(0)
            queue_crops[key].append(convert_img)
            cv2.imshow(key, resize_img)                   
        
        # carry out inference        
        for key in args.model_class:
            input = np.array(queue_crops[key]) / 255 
            result = model.predict(np.expand_dims(input, axis=0))
            time_step_images = np.array(np.hstack(queue_crops[key][:]), dtype = np.uint8)
    
        # show time-step
        if time_step_images is not None:     
            cv2.imshow('time-steps', cv2.cvtColor(time_step_images, cv2.COLOR_RGB2BGR))
        
        # show moves name
        if result is not None:
            frame_pos = '{:05} '.format(int(cap.get(cv2.CAP_PROP_POS_FRAMES)))
            pannel = np.array(np.full(time_step_images.shape, 255), dtype = np.uint8)
            pannel = cv2.putText(pannel, 
                                 frame_pos + str(np.argmax(result.squeeze(), axis=1)), 
                                 (0, int(time_step_images.shape[0]/2)), 
                                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.imshow('time-steps-result', cv2.cvtColor(pannel, cv2.COLOR_RGB2BGR),)
            cv2.setWindowTitle( 'time-steps-result', str(args.moves_name) ) 

        if cv2.waitKey(1) & 0xFF == ord('q'): break 

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # [[[ Input arguments
    parser.add_argument('--config',     dest='config',      type=str,   default='./yolocfg/yolov3-tiny-test.cfg')
    parser.add_argument('--data',       dest='data',        type=str,   default='./yolocfg/moves-sf2.data')
    parser.add_argument('--weights',    dest='weights',     type=str,   default='./yolocfg/result/yolov3-tiny_final.weights')
    parser.add_argument('--threshold',  dest='threshold',   type=float, default=0.25)
    parser.add_argument('--video',      dest='video',       type=str,   required=True)
    parser.add_argument('--model',      dest='model',       type=str,   required=True)
    # ]]]
    
    # [[[ Ouput arguments
    parser.add_argument('--print_detections',   dest='print_detections',   type=lambda s : s in ['True'], default=False)
    parser.add_argument('--print_coordinates',  dest='print_coordinates',  type=lambda s : s in ['True'], default=False)
    parser.add_argument('--print_fps',          dest='print_fps',          type=lambda s : s in ['True'], default=True)
    parser.add_argument('--draw_boxes',         dest='draw_boxes',         type=lambda s : s in ['True'], default=True)
    parser.add_argument('--output_class',       dest='output_class',       type=lambda s: s.split(','))
    parser.add_argument('--model_class',        dest='model_class',        type=lambda s: s.split(','))
    parser.add_argument('--moves_name',         dest='moves_name',         type=lambda s: s.split(','))
    # ]]]
    
    # args = parser.parse_args()
    
    #""" Example
    args = parser.parse_args(['--video', './dataset/video/2022-07-14-21-05-17.mp4', 
                              '--draw_boxes', 'False',
                              '--print_fps', 'True',
                              '--model', './movesSF2.h5',
                              '--output_class', 'ken_a,zangief_a',
                              '--model_class', 'ken_a',
                              '--moves_name', 'None,Hadoken,Shoryuken,Tatsumaki Senpuu Kyaku'])
    #"""
    main(args)
    