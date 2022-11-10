#===============================================================================
#
#   File name   : movesSF2_yolo_to_image.py
#   Author      : lycobs@gmail.com
#   Created date: 2022-10
#   Description : Detected object via YOLO to image sequence for RNN's data
#
#===============================================================================


import os
os.add_dll_directory(os.environ['CUDA_PATH'] + '/bin') # For darknet with CUDA
os.add_dll_directory(os.getcwd())

import cv2
import random
import time
import argparse
import darknet
import darknet_images
from datetime import datetime
import util.movesSF2_util_yolo as util_yolo


#===============================================================================
# Functions
#===============================================================================


def main(args):
    time_stamp = str(int(datetime.now().timestamp()))
    args.output_location = args.output_location + "_" + time_stamp
    try: os.makedirs(args.output_location)
    except FileExistsError: pass
    
    # [[[ configuration
    random.seed(3)  # deterministic bbox colors
    network, class_names, class_colors = darknet.load_network( args.config, args.data, args.weights, batch_size=1 )
    # ]]]
    
    count = {}
    for key in args.output_class: count[key] = 0
    
    cv2.namedWindow('inference', cv2.WINDOW_NORMAL)
    cap = cv2.VideoCapture(args.video)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret != True: continue
        
        # detecting using yolo
        image, detections = util_yolo.video_detection(frame, network, class_names, class_colors, args.threshold, True )

        # extracting detections to image file
        if args.output_detectons: 
            image_crop = util_yolo.video_crop(image, args.output_class, detections)
            for key in image_crop.keys():
                name = '{}/{}_{:08}.jpg'.format(args.output_location, key, int(count[key])%100000000)
                cv2.imwrite(name, image_crop[key])
                count[key] += 1
                
        # showing detections information
        if args.print_detections: darknet.print_detections(detections, args.print_coordinates)
            
        # showing fps(frame per seconds)
        if args.print_fps: print("FPS: {}".format(cap.get(cv2.CAP_PROP_FPS)))
            
        # showing output image after carrying out yolo
        if args.draw_show: cv2.imshow('inference', image)
            
        # keep going or quit 
        if cv2.waitKey(33) & 0xFF == ord('q'): break 
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
    # ]]]
    
    # [[[ Ouput arguments
    parser.add_argument('--print_detections',   dest='print_detections',   type=lambda s : s in ['True'], default=False)
    parser.add_argument('--print_coordinates',  dest='print_coordinates',  type=lambda s : s in ['True'], default=False)
    parser.add_argument('--print_fps',          dest='print_fps',          type=lambda s : s in ['True'], default=True)
    parser.add_argument('--draw_show',          dest='draw_show',          type=lambda s : s in ['True'], default=True)
    parser.add_argument('--draw_boxes',         dest='draw_boxes',         type=lambda s : s in ['True'], default=True)
    parser.add_argument('--output_detectons',   dest='output_detectons',   type=lambda s : s in ['True'], default=False)
    parser.add_argument('--output_class',       dest='output_class',       type=lambda s: s.split(','))
    parser.add_argument('--output_location',    dest='output_location',    type=str)
    # ]]]
    
    # args = parser.parse_args()
    
    #""" Example
    args = parser.parse_args(['--video', './dataset/videos/ken_vs_zangief.mp4', 
                              '--output_detectons', 'True',
                              '--output_class', 'ken_a,zangief_a',
                              '--output_location', './dataset/images_moves'])
    #"""
    main(args)

