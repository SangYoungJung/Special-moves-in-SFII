import os
import glob
import json
import argparse
import numpy as np
import pickle
from PIL import Image, ImageOps


def creation_pickle(output, dataset_x, dataset_y, name_classes):
    dataset = dict()
    dataset['dataset_x'] = dataset_x
    dataset['dataset_y'] = dataset_y
    dataset['name_classes'] = name_classes
    with open(output, 'wb') as handle:
        pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)


def image_resize_and_aug_horizontal(images, w, h, aug):
    images_list = []
    images_list_aug = []
    for ix, im in enumerate(images):
        img = Image.open(im)
        img_resize = img.resize((w, h)) # Resize(of Fixed size) Image.LANCZOS
        img_arr = np.array(img_resize) # ImageOps.grayscale(img_resize)
        images_list.append(img_arr)
        if aug : images_list_aug.append(np.array(img_resize.transpose(Image.Transpose.FLIP_LEFT_RIGHT)))
    return np.array(images_list) / 255, np.array(images_list_aug) / 255 # Data Normalization
    
        
def make_moves_list(input_moves, dataset_class_name, dataset_index):
    with open(input_moves) as f:
        obj = json.load(f)
        for key in obj['names'].keys():
            class_name = key
            class_number = obj['names'][key]
            class_moves_list = obj[key]
            class_moves_list.sort()

            # fill dataset array out from json
            dataset_class_name[key] = class_number
            for even, odd in zip(class_moves_list[::2], class_moves_list[1::2]):
                dataset_index[even:odd+1] = class_number


def make_time_series_data(images_ndarry, dataset_index, step):
    dataset_x = np.array([images_ndarry[i:i+step] for i in range(images_ndarry.shape[0]-step)])
    dataset_x = dataset_x.reshape(dataset_x.shape[0], 
                                dataset_x.shape[1], 
                                dataset_x.shape[2], 
                                dataset_x.shape[3], 
                                dataset_x.shape[4]) 
    dataset_y = np.array([dataset_index[i:i+step] for i in range(dataset_index.shape[0]-step)])
    return dataset_x, dataset_y


def main(args):
    # Creation data array for moves info of each image
    images = sorted(glob.glob(args.input + '/*.jpg'))
    dataset_index = np.zeros((len(images),1))
    dataset_class_name = {"None":0}
    dataset_class_num = 0

    # Load classification(or moves) information from json which is defined.
    # Read moves list json and fill out dataset_index
    make_moves_list(args.input_moves, dataset_class_name, dataset_index)
    dataset_class_num = len(np.unique(dataset_index))
    print(dataset_index.shape)
    print(dataset_class_num, dataset_class_name)
        
    # Load images to dataset with resizing width x height
    re_images, re_images_aug = image_resize_and_aug_horizontal(images, 
                                                               args.output_width, 
                                                               args.output_height, 
                                                               args.output_aug_horizontal)
    
    # Make time series data as 10 frame and classes are paired 
    # train_x : (#, step, img_w, img_h, img_channel) (#, 10, 100, 100, 3)
    # train_y : (#, step, moves of index)            (#, 10, 1)
    dataset_x, dataset_y = make_time_series_data(re_images, dataset_index, args.output_step)
    if args.output_aug_horizontal:
        x_aug, y_aug = make_time_series_data(re_images_aug, dataset_index, args.output_step)
        dataset_x = np.concatenate((dataset_x, x_aug), axis=0)
        dataset_y = np.concatenate((dataset_y, y_aug), axis=0)
        print("Data Augmentation", dataset_x.shape)
        print("Data Augmentation", dataset_y.shape)
    
    # Create a pickle(or serialized) for dataset
    creation_pickle(args.output, dataset_x, dataset_y, dataset_class_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # [[[ Input arguments
    parser.add_argument('--input',  dest='input',  type=str, required=True)
    parser.add_argument('--input_moves',  dest='input_moves',  type=str, required=True)
    # ]]]
    
    # [[[ Ouput arguments
    parser.add_argument('--output_width', dest='output_width', type=int, default=100)
    parser.add_argument('--output_height', dest='output_height', type=int, default=100)
    parser.add_argument('--output_step', dest='output_step', type=int, default=10)
    parser.add_argument('--output_aug_horizontal', dest='output_aug_horizontal', type=lambda s : s in ['True'], default=False)
    parser.add_argument('--output', dest='output', type=str, required=True)
    # ]]]
    
    #args = parser.parse_args()
    
    #""" Example
    args = parser.parse_args(['--input', './dataset/image_moves', 
                              '--input_moves', './dataset/image_moves/ken_moves.json',
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