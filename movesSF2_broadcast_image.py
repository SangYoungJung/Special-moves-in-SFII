#===============================================================================
#
#   File name   : movesSF2_broadcast_image.py
#   Author      : lycobs@gmail.com
#   Created date: 2022-10
#   Description : Broadcsting with only prepared images
#
#===============================================================================

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import random
import time
import argparse
import glob
import numpy as np
from datetime import datetime
import tensorflow as tf
from tensorflow import keras

import util.movesSF2_util as util
import util.movesSF2_util_model as util_model


#===============================================================================
# Functions
#===============================================================================


def main(args):
    # [[[ load pre-trained model as movesSF2
    model, input_time_steps, input_image_size = util_model.load_model(args.model)
    # ]]]
    
    queue_crops = {}
    queue_crops[args.output_class] = [ np.empty(input_image_size) for i in range(input_time_steps)]
    cv2.namedWindow(args.output_class, cv2.WINDOW_NORMAL)
    cv2.namedWindow('time-steps', cv2.WINDOW_NORMAL)
    
    count = 0
    index = 0
    frame_locations = sorted(glob.glob(args.image + '/*.jpg'))
    for loc in frame_locations:
        # Read Image
        frame = cv2.imread(loc)

        # Horizontal Flip
        if args.image_flip: frame = cv2.flip(frame, 1) 
        
        # Show Frame
        cv2.imshow(args.output_class, frame)
        
        # Inference per frame times            
        count += 1
        if count % args.inference_per_frame != 0 : continue
        
        # Collect input time-step images as 10 frames
        resize_img = cv2.resize(frame, dsize=input_image_size[:2])
        convert_img = cv2.cvtColor(resize_img, cv2.COLOR_BGR2RGB)
        queue_crops[args.model_class].pop(0)
        queue_crops[args.model_class].append(convert_img)
        
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
        
        # Quit
        if cv2.waitKey(1) & 0xFF == ord('q'): break 

    cv2.destroyAllWindows()


#===============================================================================
# Main
#===============================================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # [[[ Input arguments
    parser.add_argument('--image',      dest='image',       type=str,   required=True)
    parser.add_argument('--image_flip', dest='image_flip',  type=lambda s : s in ['True'], default=False)
    parser.add_argument('--model',      dest='model',       type=str,   required=True)
    # ]]]
    
    # [[[ Ouput arguments for Image and movesSF2
    parser.add_argument('--output_class',       dest='output_class',       type=str)
    parser.add_argument('--model_class',        dest='model_class',        type=str)
    parser.add_argument('--inference_per_frame',dest='inference_per_frame',type=int, default=1)
    parser.add_argument('--judgement_time_step',dest='judgement_time_step',type=int, default=3)
    # ]]]
    
    # args = parser.parse_args()
    
    #""" Example
    args = parser.parse_args(['--image', './dataset/images_moves', 
                              '--image_flip', 'True',
                              '--model', './movesSF2_pre_trained.h5',
                              '--output_class', 'ken_a',
                              '--model_class', 'ken_a',
                              '--inference_per_frame', '1',
                              '--judgement_time_step', '3'])
    #"""
    main(args)

