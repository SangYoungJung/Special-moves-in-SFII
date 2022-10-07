import cv2
import os
import argparse
from datetime import datetime


def video_information(vidcap):
    print('Frame width:', int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    print('Frame height:', int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print('Frame count:', int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)))
    print('FPS:',int(vidcap.get(cv2.CAP_PROP_FPS)))
    return int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))


def main(args):
    vidcap = cv2.VideoCapture(args.video)
    num_frames = video_information(vidcap)
    
    for index in range(num_frames):
        success,image = vidcap.read()
        if success and index % int(args.stride) == 0:
            name = '{}/{}_{}_{:08}.jpg'.format(args.output_location, 
                                os.path.basename(args.video), 
                                str(int(datetime.now().timestamp())),
                                index)
            ret = cv2.imwrite(name, image)
            if index % 100 == 0: print(ret,name)
            if ret != True: break
    vidcap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # [[[ Input arguments
    parser.add_argument('--stride', dest='stride', type=int, default=15)
    parser.add_argument('--video',  dest='video',  type=str, required=True)
    # ]]]
    
    # [[[ Ouput arguments
    parser.add_argument('--output_location', dest='output_location', type=str, required=True)
    # ]]]
    
    args = parser.parse_args()
    
    """ Example
    args = parser.parse_args(['--video', './dataset/videos/ken_vs_zangief.mp4', 
                              '--output_location', './dataset/temp'])
    """
    main(args)
    
    