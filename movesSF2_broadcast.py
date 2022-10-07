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


def PIL2OpenCV(pil_image):
    numpy_image= np.array(pil_image)
    opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    return opencv_image


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
    
    
    # time_steps_list   = util.make_image_list('./debugging/*.jpg')
    # time_steps_imgs,_ = util.load_images(time_steps_list)
    # time_steps_imgs   = util.make_time_series_data(time_steps_imgs, input_time_steps)
    # print(time_steps_imgs.shape)
    # print(time_steps_imgs[0][0])
    
       
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
        if ret != True: continue
        
        # detecting using yolo
        image, detections = util_yolo.video_detection(frame, network, class_names, class_colors, args.threshold, args.draw_boxes )
        cv2.imshow('yolo', image)

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
        for key in args.model_class:
            input = np.array(queue_crops[key]) / 255              
            # input = time_steps_imgs[index % time_steps_imgs.shape[0]]/255
            result = model.predict(np.expand_dims(input, axis=0))
            print("{:08d}".format(index), "predict :\t", np.argmax(result.squeeze(), axis=1))
            index += 1
        

        # if index == 155:
        #     count = 0
        #     for key in args.model_class:
        #         for i, step in enumerate(queue_crops[key]):
        #             cv2.imwrite('./debuging/ken' + str(count)+'.jpg', resize_img)
        #             count += 1

            
        # showing output image after carrying out yolo
        numpy_horizontal = None
        for key in args.model_class: 
            numpy_horizontal = np.array(np.hstack(queue_crops[key][:]), dtype = np.uint8)
            numpy_horizontal = cv2.cvtColor(numpy_horizontal, cv2.COLOR_RGB2BGR)
        #     numpy_horizontal = np.hstack(time_steps_imgs[index % time_steps_imgs.shape[0]][:])
            # print(numpy_horizontal.shape)
        
        # numpy_horizontal = PIL2OpenCV(time_steps_imgs[index % time_steps_imgs.shape[0]][0])
        if numpy_horizontal is not None:
            cv2.imshow('time-steps', numpy_horizontal)


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
    # ]]]
    
    # args = parser.parse_args()
    
    #""" Example
    args = parser.parse_args(['--video', './dataset/video/2022-07-14-21-05-17.mp4', 
                              '--draw_boxes', 'False',
                              '--print_fps', 'True',
                              '--model', './movesSF2.h5',
                              '--output_class', 'ken_a,zangief_a',
                              '--model_class', 'ken_a'])
    #"""
    main(args)

