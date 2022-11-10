
#===============================================================================
#
#   File name   : movesSF2_util_yolo.py
#   Author      : lycobs@gmail.com
#   Created date: 2022-10
#   Description : General helpful functions for YOLO
#
#===============================================================================


import darknet
import darknet_images
import cv2
import numpy as np


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


def convert_original_frame(image, bbox, darkent_height, darknet_width):
    x, y, w, h = bbox
    x, y, w, h = x/darknet_width, y/darkent_height, w/darknet_width, h/darkent_height

    image_h, image_w, __ = image.shape

    orig_x       = int(x * image_w)
    orig_y       = int(y * image_h)
    orig_width   = int(w * image_w)
    orig_height  = int(h * image_h)

    bbox_converted = (orig_x, orig_y, orig_width, orig_height)
    return bbox_converted

