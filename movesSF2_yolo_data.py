import glob
import numpy as np
import argparse
import movesSF2_load_data as load_data


""" main fucntion
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pathname', type=str, default='./dataset/images/*.jpg')
    parser.add_argument('--seed', type=int, default=1001)
    args = parser.parse_args()
    
    # 1. Download a weight file of yolo
    load_data.download('https://drive.google.com/uc?id=' + '1xQsP0HJ61AZzdKQZkWpObEECGjfcXplw', 
                       './yolocfg/yolov3-tiny.weights')
    
    # 2. Download the moves SF2 weight file of yolo
    load_data.download('https://drive.google.com/uc?id=' + '1DvdjAOgwXv4gzDn-wIbHux4BHcARZo-P', 
                       './yolocfg/yolov3-tiny_final_sf2.weights')
    
    # 3. Load images
    files = sorted(glob.glob(args.pathname))
    
    # 4. Split dataset into train and valid
    train_save = './yolocfg/moves-sf2-train.txt'
    valid_save = './yolocfg/moves-sf2-valid.txt'
    
    np.random.seed(args.seed)
    index = np.arange(0, len(files))
    np.random.shuffle(index)

    train = index[0:int(len(files) * 0.8)]
    valid = index[len(train):]

    # 5. Create a train 
    with open(train_save, 'w') as f:
        for ii in train: f.write(files[ii] + "\n")

    # 6. Create a valid
    with open(valid_save, 'w') as f:
        for jj in valid: f.write(files[jj] + "\n")
    
    