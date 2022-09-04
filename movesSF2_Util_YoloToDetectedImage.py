import os
os.add_dll_directory('c:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.1/bin')
os.add_dll_directory('d:/MachineLearning/movesSF2')

import cv2
import random
import time
import darknet
import darknet_images
from datetime import datetime

def video_detection(capture, network, class_names, class_colors, thresh, draw_boxes=True):
    # Darknet doesn't accept numpy images.
    # Create one with image we reuse for each detect
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)

    ret, frame = capture.read()
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
    for d in detections:
        if d[0] == class_name:
            x = int(d[2][0]) # center x
            y = int(d[2][1]) # center y
            w = int(d[2][2])
            h = int(d[2][3])            
            x = max(int(x - w/2), 0)
            y = max(int(y - h/2), 0)
            return img[y: y + h, x: x + w]
    return None



config_file = './yolocfg/yolov3-tiny-test.cfg'
data_file = './yolocfg/moves-sf2.data'
weights = './yolocfg/result/yolov3-tiny_final.weights'
input = './dataset/video/2022-07-14-21-05-17.mp4'
thresh = 0.25

random.seed(3)  # deterministic bbox colors
network, class_names, class_colors = darknet.load_network(
    config_file,
    data_file,
    weights,
    batch_size=1
)

output_detection = False
output_coordinates = True
output_show = True
output_save = False
output_fps = True

index = 0
cap = cv2.VideoCapture(input)
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret != True: continue
    
    prev_time = time.time()
    image, detections = video_detection(cap, network, class_names, class_colors, thresh, True )
    
    # extracting detections to image file
    if output_save: 
        image_crop = video_crop(image, 'ken_a', detections)
        if image_crop is not None:
            name = '{}_{:08}.jpg'.format('./dataset_rnn/ken_a', 
                                index%10000)
            cv2.imwrite(name, image_crop)
            index += 1
    # showing detections information
    if output_detection: 
        darknet.print_detections(detections, output_coordinates)
     # showing fps(frame per seconds)
    if output_fps:
        fps = int(1/(time.time() - prev_time))
        print("FPS: {}".format(fps))
    # showing output image after carrying out yolo
    if output_show: 
        cv2.imshow('Inference', image)
    # keep going or quit
    if cv2.waitKey(33) & 0xFF == ord('q'):
        break   
        
cap.release()
cv2.destroyAllWindows()
