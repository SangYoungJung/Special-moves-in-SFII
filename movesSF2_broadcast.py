import os
os.add_dll_directory('c:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.1/bin')
os.add_dll_directory(os.getcwd())

import cv2
import random
import time
import argparse
import numpy as np
import darknet
import darknet_images
from datetime import datetime


def video_detection(frame, network, class_names, class_colors, thresh, draw_boxes=True):
    # Darknet doesn't accept numpy images.
    # Create one with image we reuse for each detect
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),
                               interpolation=cv2.INTER_LINEAR)

    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
    darknet.free_image(darknet_image)
    
    if draw_boxes:
        image = darknet.draw_boxes(detections, image_resized, class_colors)
    else:
        image = image_resized
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), detections


def video_crop(img, class_name, detections):
    crop = {}
    for d in detections:
        for name in class_name:
            if d[0] == name:
                x = int(d[2][0]) # center x
                y = int(d[2][1]) # center y
                w = int(d[2][2])
                h = int(d[2][3])            
                x = max(int(x - w/2), 0)
                y = max(int(y - h/2), 0)
                crop[name] = img[y: y + h, x: x + w]
    return crop


def main(args):
    # [[[ configuration
    random.seed(3)  # deterministic bbox colors
    network, class_names, class_colors = darknet.load_network( args.config, args.data, args.weights, batch_size=1 )
    # ]]]
       
    cv2.namedWindow('inference', cv2.WINDOW_NORMAL)
    queue_crops = {}
    for key in args.output_class: 
        queue_crops[key] = [ np.empty((56,56,3)) for i in range(10)]
        cv2.namedWindow(key, cv2.WINDOW_NORMAL)
        
    cap = cv2.VideoCapture(args.video)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret != True: continue
        
        # detecting using yolo
        image, detections = video_detection(frame, network, class_names, class_colors, args.threshold, args.draw_boxes )

        # extracting detections to image file
        image_crop = video_crop(image, args.output_class, detections)
        
        ken_image = None
        for key in image_crop.keys():
            print(len(queue_crops[key]))
            queue_crops[key].pop(0)
            queue_crops[key].append(cv2.resize(image_crop[key], dsize=(56, 56)))
            cv2.imshow(key, queue_crops[key][9])
            
               
        # showing detections information
        # if args.print_detections: darknet.print_detections(detections, args.print_coordinates)
            
        # showing fps(frame per seconds)
        # if args.print_fps: print("FPS: {}".format(cap.get(cv2.CAP_PROP_FPS)))
            
        # showing output image after carrying out yolo
        # numpy_horizontal = None
        # for key in args.output_class: 
        #     if len(queue_crops[key]) == 10:
        #         numpy_horizontal = np.hstack((queue_crops[key][0], queue_crops[key][1]))
        #         # numpy_vertical = np.vstack()
        
        # if numpy_horizontal is not None:    
        #     cv2.imshow('inference', numpy_horizontal)
        
        # cv2.imshow('inference', ken_image if ken_image != None else image )
        # for key in image_crop.keys():
        #     cv2.imshow('inference', image_crop[key])
            
        cv2.imshow('inference', image)
        if cv2.waitKey(33) & 0xFF == ord('q'): break 

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
    # ]]]
    
    # [[[ Ouput arguments
    parser.add_argument('--print_detections',   dest='print_detections',   type=lambda s : s in ['True'], default=False)
    parser.add_argument('--print_coordinates',  dest='print_coordinates',  type=lambda s : s in ['True'], default=False)
    parser.add_argument('--print_fps',          dest='print_fps',          type=lambda s : s in ['True'], default=True)
    parser.add_argument('--draw_boxes',         dest='draw_boxes',         type=lambda s : s in ['True'], default=True)
    parser.add_argument('--output_class',       dest='output_class',       type=lambda s: s.split(','))
    # ]]]
    
    # args = parser.parse_args()
    
    #""" Example
    args = parser.parse_args(['--video', './dataset/video/2022-07-14-21-05-17.mp4', 
                              '--draw_boxes', 'False',
                              '--output_class', 'ken_a,zangief_a'])
    #"""
    main(args)

