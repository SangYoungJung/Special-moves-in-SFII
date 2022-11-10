#===============================================================================
#
#   File name   : movesSF2_time_series_data.py
#   Author      : lycobs@gmail.com
#   Created date: 2022-10
#   Description : video to image frames for YOLO training data and etc.
#
#===============================================================================


import cv2
import os
import argparse
from datetime import datetime
import util.movesSF2_util as util


#===============================================================================
# Functions
#===============================================================================


def main(args):
    time_stamp = str(int(datetime.now().timestamp()))
    args.output_location = args.output_location + "_" + time_stamp
    try: os.makedirs(args.output_location)
    except FileExistsError: pass
    
    vidcap = cv2.VideoCapture(args.video)
    num_frames = util.video_information(vidcap)
    
    for index in range(num_frames):
        success,image = vidcap.read()
        if success and index % int(args.stride) == 0:
            name = '{}/{}_{}_{:08}.jpg'.format(args.output_location, 
                                os.path.basename(args.video), 
                                time_stamp,
                                index)
            ret = cv2.imwrite(name, image)
            if index % 100 == 0: print(ret,name)
            if ret != True: break
    vidcap.release()
    cv2.destroyAllWindows()


#===============================================================================
# Main
#===============================================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # [[[ Input arguments
    parser.add_argument('--stride', dest='stride', type=int, default=15)
    parser.add_argument('--video',  dest='video',  type=str, required=True)
    # ]]]
    
    # [[[ Ouput arguments
    parser.add_argument('--output_location', dest='output_location', type=str, required=True)
    # ]]]
    
    # args = parser.parse_args()
    
    #""" Example
    args = parser.parse_args(['--video', './dataset/videos/ken_vs_zangief.mp4', 
                              '--output_location', './dataset/images'])
    #"""
    main(args)
    
    