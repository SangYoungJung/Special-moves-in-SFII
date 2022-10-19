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

import movesSF2_util as util
import movesSF2_util_yolo as util_yolo


def moves_name(moves_num):
    return 'Hadoken' if moves_num == 1 else 'Shoryuken' if moves_num == 2 else 'Tatsumaki' if moves_num == 3 else ''

def moves_judgement(result, judgement_per_frame):
    continuous = 0
    prev_moves_num = result[-1]
    for moves_num in np.flip(result):
        if prev_moves_num == moves_num: continuous += 1
        if continuous >= judgement_per_frame: return moves_name(moves_num)
    return moves_name(0)

def main(args):
    # [[[ configuration for darknet
    random.seed(3)  # deterministic bbox colors
    network, class_names, class_colors = darknet.load_network( args.config, args.data, args.weights, batch_size=1 )
    # ]]]

    # [[[ load of pre-trained model as movesSF2
    model = tf.keras.models.load_model('movesSF2.h5')
    model_config = model.get_config()
    input_time_steps = model_config["layers"][0]["config"]["batch_input_shape"][1]
    input_image_size = model_config["layers"][0]["config"]["batch_input_shape"][2:]
    print("Input shape : ", model_config["layers"][0]["config"]["batch_input_shape"])
    # ]]]
    
    cv2.namedWindow('yolo', cv2.WINDOW_NORMAL)
    queue_crops = {}
    for key in args.output_class: 
        queue_crops[key] = [ np.empty(input_image_size) for i in range(input_time_steps)]
        cv2.namedWindow(key, cv2.WINDOW_NORMAL)
        
    cv2.namedWindow('time-steps', cv2.WINDOW_NORMAL)

    count = 0
    index = 0
    cap = cv2.VideoCapture(args.video)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret != True: break
        
        # detecting using yolo
        image, detections = util_yolo.video_detection(frame, network, class_names, class_colors, args.threshold, args.draw_boxes )
        cv2.imshow('yolo', image)
        
        # Frame Per Inference
        count += 1
        if count % args.inference_per_frame != 0 : continue

        # extracting detections to image file
        image_crop = util_yolo.video_crop(image, args.output_class, detections)
        
        ken_image = None
        for key in image_crop.keys():
            resize_img = cv2.resize(image_crop[key], dsize=input_image_size[:2])
            convert_img = cv2.cvtColor(resize_img, cv2.COLOR_BGR2RGB)
            queue_crops[key].pop(0)
            queue_crops[key].append(convert_img)
            cv2.imshow(key, resize_img)                   
        
        # Inference 
        argmax_result = np.zeros(input_time_steps)
        for key in args.model_class:
            input = np.array(queue_crops[key]) / 255              
            result = model.predict(np.expand_dims(input, axis=0))
            argmax_result = np.argmax(result.squeeze(), axis=1)
            print("{:08d}".format(index), "predict :\t", argmax_result)
            index += 1
        
        # showing output image after carrying out yolo
        numpy_horizontal = None
        for key in args.model_class: 
            numpy_horizontal = np.array(np.hstack(queue_crops[key][:]), dtype = np.uint8)
            numpy_horizontal = cv2.cvtColor(numpy_horizontal, cv2.COLOR_RGB2BGR)
        
        if numpy_horizontal is not None:
            judgement = moves_judgement(argmax_result, args.judgement_per_frame)
            textsize = cv2.getTextSize(judgement, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            textX = (numpy_horizontal.shape[1] - textsize[0]) / 2
            textY = (numpy_horizontal.shape[0] + textsize[1]) / 2
            numpy_horizontal = cv2.putText(numpy_horizontal, judgement, (int(textX), int(textY)), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA).copy()
            cv2.imshow('time-steps', numpy_horizontal)

        # Quit
        if cv2.waitKey(1) & 0xFF == ord('q'): break 

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # [[[ Input arguments
    parser.add_argument('--config',     dest='config',      type=str,   default='./yolocfg/yolov3-tiny-test.cfg')
    parser.add_argument('--data',       dest='data',        type=str,   default='./yolocfg/moves-sf2.data')
    parser.add_argument('--weights',    dest='weights',     type=str,   default='./yolocfg/result/yolov3-tiny_final_sf2.weights')
    parser.add_argument('--threshold',  dest='threshold',   type=float, default=0.25)
    parser.add_argument('--video',      dest='video',       type=str,   required=True)
    parser.add_argument('--model',      dest='model',       type=str,   required=True)
    # ]]]
    
    # [[[ Ouput arguments for Yolo and movesSF2
    parser.add_argument('--print_detections',   dest='print_detections',   type=lambda s : s in ['True'], default=False)
    parser.add_argument('--print_coordinates',  dest='print_coordinates',  type=lambda s : s in ['True'], default=False)
    parser.add_argument('--print_fps',          dest='print_fps',          type=lambda s : s in ['True'], default=True)
    parser.add_argument('--draw_boxes',         dest='draw_boxes',         type=lambda s : s in ['True'], default=True)
    parser.add_argument('--output_class',       dest='output_class',       type=lambda s: s.split(','))
    parser.add_argument('--model_class',        dest='model_class',        type=lambda s: s.split(','))
    parser.add_argument('--inference_per_frame',dest='inference_per_frame',type=int, default=1)
    parser.add_argument('--judgement_per_frame',dest='judgement_per_frame',type=int, default=5)
    # ]]]
    
    # args = parser.parse_args()
    
    #""" Example
    args = parser.parse_args(['--video', './dataset/videos/ken_vs_zangief.mp4', 
                              '--draw_boxes', 'False',
                              '--print_fps', 'True',
                              '--model', './movesSF2.h5',
                              '--output_class', 'ken_a,zangief_a',
                              '--model_class', 'ken_a',
                              '--inference_per_frame', '3',
                              '--judgement_per_frame', '3'])
    #"""
    main(args)

