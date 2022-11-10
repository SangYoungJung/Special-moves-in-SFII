#===============================================================================
#
#   File name   : movesSF2_broadcast_yolo.py
#   Author      : lycobs@gmail.com
#   Created date: 2022-10
#   Description : Broadcsting with YOLO with video
#
#===============================================================================


import os
os.add_dll_directory(os.environ['CUDA_PATH'] + '/bin') # For darknet with CUDA
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

import util.movesSF2_util as util
import util.movesSF2_util_yolo as util_yolo
import util.movesSF2_util_model as util_model


#===============================================================================
# Functions
#===============================================================================


def main(args):
    # [[[ configuration for darknet
    random.seed(3)  # deterministic bbox colors
    network, class_names, class_colors = darknet.load_network( args.config, args.data, args.weights, batch_size=1 )
    darknet_width = darknet.network_width(network)
    darknet_height = darknet.network_height(network)
    # ]]]

    # [[[ load pre-trained model as movesSF2
    model, input_time_steps, input_image_size = util_model.load_model(args.model)
    # ]]]
    
    cv2.namedWindow('yolo', cv2.WINDOW_NORMAL)
    cv2.namedWindow('time-steps', cv2.WINDOW_NORMAL)
    queue_crops = {}
    for key in args.output_class: 
        queue_crops[key] = [ np.empty(input_image_size) for i in range(input_time_steps)]
        cv2.namedWindow(key, cv2.WINDOW_NORMAL)

    count = 0
    index = 0
    cap = cv2.VideoCapture(args.video)
    while(cap.isOpened()):
        # Read a frame
        ret, frame = cap.read()
        if ret != True: break
        
        # Horizontal Flip
        if args.video_flip: frame = cv2.flip(frame, 1) 
        
        # Detection using yolo
        image, detections = util_yolo.video_detection(frame, network, class_names, class_colors, 
                                                      args.threshold, False )
        # Yolo's detection to original frame
        detections_adjusted = [ ]
        if args.draw_boxes: 
            for label, confidence, bbox in detections:
                bbox_adjusted = util_yolo.convert_original_frame(frame, bbox, darknet_height, darknet_width)
                detections_adjusted.append((str(label), confidence, bbox_adjusted))
            frame = darknet.draw_boxes(detections_adjusted, frame, class_colors)
        cv2.imshow('yolo', frame)    
        
        # Inference per frame times
        count += 1
        if count % args.inference_per_frame != 0 : continue

        # extracting detections to image file
        # Note : RNN's input data is a detected and cropped image from resized yolo's input image. 
        image_crop = util_yolo.video_crop(image, args.output_class, detections)
        
        # Collect input time-step images as 10 frames
        for key in image_crop.keys():
            resize_img = cv2.resize(image_crop[key], dsize=input_image_size[:2])
            convert_img = cv2.cvtColor(resize_img, cv2.COLOR_BGR2RGB)
            queue_crops[key].pop(0)
            queue_crops[key].append(convert_img)
            cv2.imshow(key, resize_img)                   
        
        # Inference 
        argmax_result = util_model.inference(model, queue_crops[args.model_class])
        print("{:08d}".format(index), "predict :\t", argmax_result)
        index += 1
        
        # Show output image
        numpy_horizontal = util_model.inference_summary(queue_crops[args.model_class], 
                                                        argmax_result, 
                                                        args.judgement_time_step,
                                                        'left')
        cv2.imshow('time-steps', numpy_horizontal)
        
        # Show FPS
        if args.print_fps : util.fps_show()

        # Quit
        if cv2.waitKey(1) & 0xFF == ord('q'): break 

    cap.release()
    cv2.destroyAllWindows()


#===============================================================================
# Main
#===============================================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # [[[ Input arguments
    parser.add_argument('--config',     dest='config',      type=str,   default='./yolocfg/yolov3-tiny-test-moves-sf2.cfg')
    parser.add_argument('--data',       dest='data',        type=str,   default='./yolocfg/moves-sf2.data')
    parser.add_argument('--weights',    dest='weights',     type=str,   default='./yolov3-tiny-final-sf2.weights')
    parser.add_argument('--threshold',  dest='threshold',   type=float, default=0.25)
    parser.add_argument('--video',      dest='video',       type=str,   required=True)
    parser.add_argument('--model',      dest='model',       type=str,   required=True)
    parser.add_argument('--video_flip', dest='video_flip',  type=lambda s : s in ['True'], default=False)
    # ]]]
    
    # [[[ Ouput arguments for Yolo and movesSF2
    parser.add_argument('--print_detections',   dest='print_detections',   type=lambda s : s in ['True'], default=False)
    parser.add_argument('--print_coordinates',  dest='print_coordinates',  type=lambda s : s in ['True'], default=False)
    parser.add_argument('--print_fps',          dest='print_fps',          type=lambda s : s in ['True'], default=True)
    parser.add_argument('--draw_boxes',         dest='draw_boxes',         type=lambda s : s in ['True'], default=True)
    parser.add_argument('--output_class',       dest='output_class',       type=lambda s: s.split(','))
    parser.add_argument('--model_class',        dest='model_class',        type=str)
    parser.add_argument('--inference_per_frame',dest='inference_per_frame',type=int, default=3)
    parser.add_argument('--judgement_time_step',dest='judgement_time_step',type=int, default=3)
    # ]]]
    
    # args = parser.parse_args()
    
    #""" Example
    args = parser.parse_args(['--video', './dataset/videos/ken_vs_zangief.mp4', 
                              '--video_flip', 'False',
                              '--draw_boxes', 'True',
                              '--print_fps', 'True',
                              '--model', './movesSF2_pre_trained.h5',
                              '--output_class', 'ken_a,zangief_a',
                              '--model_class', 'ken_a',
                              '--inference_per_frame', '3',
                              '--judgement_time_step', '3'])
    #"""
    main(args)

