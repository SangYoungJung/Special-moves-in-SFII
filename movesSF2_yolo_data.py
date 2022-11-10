#===============================================================================
#
#   File name   : movesSF2_yolo_data.py
#   Author      : lycobs@gmail.com
#   Created date: 2022-10
#   Description : shuffle train and valid dataset using seed and random of numpy
#
#===============================================================================


import glob
import numpy as np
import argparse


#===============================================================================
# Main
#===============================================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pathname', type=str, default='./dataset/images/*.jpg')
    parser.add_argument('--seed', type=int, default=1001)
    args = parser.parse_args()
    
    # 1. Load images
    files = sorted(glob.glob(args.pathname))
    
    # 2. Split dataset into train and valid
    train_save = './yolocfg/moves-sf2-train.txt'
    valid_save = './yolocfg/moves-sf2-valid.txt'
    
    np.random.seed(args.seed)
    index = np.arange(0, len(files))
    np.random.shuffle(index)

    train = index[0:int(len(files) * 0.8)]
    valid = index[len(train):]

    # 3. Create a train 
    with open(train_save, 'w') as f:
        for ii in train: f.write(files[ii] + "\n")

    # 4. Create a valid
    with open(valid_save, 'w') as f:
        for jj in valid: f.write(files[jj] + "\n")
    
