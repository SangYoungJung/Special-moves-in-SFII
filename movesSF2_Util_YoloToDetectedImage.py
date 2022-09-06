import os
os.add_dll_directory('c:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.1/bin')
os.add_dll_directory(os.getcwd())

import cv2
import random
import time
import argparse
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
    
    count = {}
    for key in args.output_class: count[key] = 0
    
    cv2.namedWindow('inference', cv2.WINDOW_NORMAL)
    cap = cv2.VideoCapture(args.video)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret != True: continue
        
        # detecting using yolo
        image, detections = video_detection(frame, network, class_names, class_colors, args.threshold, True )

        # extracting detections to image file
        if args.output_detectons: 
            image_crop = video_crop(image, args.output_class, detections)
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
    parser.add_argument('--draw_show',          dest='draw_show',          type=lambda s : s in ['True'], default=True)
    parser.add_argument('--draw_boxes',         dest='draw_boxes',         type=lambda s : s in ['True'], default=True)
    parser.add_argument('--output_detectons',   dest='output_detectons',   type=lambda s : s in ['True'], default=False)
    parser.add_argument('--output_class',       dest='output_class',       type=lambda s: s.split(','))
    parser.add_argument('--output_location',    dest='output_location',    type=str)
    # ]]]
    
    args = parser.parse_args()
    
    """ Example
    args = parser.parse_args(['--video', './dataset/video/2022-07-14-21-05-17.mp4', 
                              '--output_detectons', 'True',
                              '--output_class', 'ken_a,zangief_a',
                              '--output_location', './dataset/test'])
    """
    main(args)

