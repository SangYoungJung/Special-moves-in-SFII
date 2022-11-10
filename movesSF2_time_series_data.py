#===============================================================================
#
#   File name   : movesSF2_time_series_data.py
#   Author      : lycobs@gmail.com
#   Created date: 2022-10
#   Description : Make time series dataset for RNNs 
#
#===============================================================================


import glob
import argparse
import numpy as np
import util.movesSF2_util as util


#===============================================================================
# Functions
#===============================================================================


def main(args):
    # Creation data array for moves info of each image
    images = sorted(glob.glob(args.input + '/*.jpg'))
    dataset_index = np.zeros((len(images),1))
    dataset_class_name = {"None":0}
    dataset_class_num = 0

    # Load classification(or moves) information from json which is defined.
    # Read moves list json and fill out dataset_index
    util.make_moves_list(args.input_moves, dataset_class_name, dataset_index)
    dataset_class_num = len(np.unique(dataset_index))
    print(dataset_index.shape)
    print(dataset_class_num, dataset_class_name)
        
    # Load images to dataset with resizing width x height
    re_images, re_images_aug = util.load_images(images, 
                                                args.output_width, 
                                                args.output_height, 
                                                args.output_aug_horizontal, 
                                                255) # Normalization
    
    # Make time series data as 10 frame and classes are paired 
    # train_x : (#, step, img_w, img_h, img_channel) (#, 10, 100, 100, 3)
    # train_y : (#, step, moves of index)            (#, 10, 1)
    dataset_x, dataset_y = util.make_time_series_data_xy(re_images, dataset_index, args.output_step)
    if args.output_aug_horizontal:
        x_aug, y_aug = util.make_time_series_data_xy(re_images_aug, dataset_index, args.output_step)
        dataset_x = np.concatenate((dataset_x, x_aug), axis=0)
        dataset_y = np.concatenate((dataset_y, y_aug), axis=0)
        print("Data Augmentation", dataset_x.shape)
        print("Data Augmentation", dataset_y.shape)
    
    # Create a pickle(or serialized) for dataset
    util.creation_pickle(args.output, dataset_x, dataset_y, dataset_class_name)


#===============================================================================
# Main
#===============================================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # [[[ Input arguments
    parser.add_argument('--input',  dest='input',  type=str, required=True)
    parser.add_argument('--input_moves',  dest='input_moves',  type=str, required=True)
    # ]]]
    
    # [[[ Ouput arguments
    parser.add_argument('--output_width', dest='output_width', type=int, default=56)
    parser.add_argument('--output_height', dest='output_height', type=int, default=56)
    parser.add_argument('--output_step', dest='output_step', type=int, default=10)
    parser.add_argument('--output_aug_horizontal', dest='output_aug_horizontal', type=lambda s : s in ['True'], default=False)
    parser.add_argument('--output', dest='output', type=str, required=True)
    # ]]]
    
    #args = parser.parse_args()
    
    #""" Example
    args = parser.parse_args(['--input', './dataset/images_moves', 
                              '--input_moves', './dataset/images_moves/ken_moves.json',
                              '--output_width', '56',
                              '--output_height', '56',
                              '--output_aug_horizontal', 'True', 
                              '--output', 'movesSF2.pickle'])
    #"""
    main(args)


""" Example : moves json ( creation by manual )
{
  "names": {
    "Hadoken" : 1,
    "Shoryuken" : 2,
    "Tatsumaki Senpuu Kyaku" : 3
  },
  "Hadoken" : [    
    63,75,
    200,212],
  "Shoryuken" : [    
    49,58,
    105,119],
  "Tatsumaki Senpuu Kyaku" : [   
    237,244,
    695,708]
}
"""