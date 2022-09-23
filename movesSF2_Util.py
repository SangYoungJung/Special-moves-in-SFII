import pickle
from PIL import Image, ImageOps
import numpy as np
import json
import glob

""" Creation Functions
"""
def creation_pickle(output, dataset_x, dataset_y, name_classes):
    dataset = dict()
    dataset['dataset_x'] = dataset_x
    dataset['dataset_y'] = dataset_y
    dataset['name_classes'] = name_classes
    with open(output, 'wb') as handle:
        pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)


""" Load Functions
"""
def load_images(images, w=0, h=0, aug_horizontal = False, normalization = 1, gray_scale = False):
    images_list = []
    images_list_aug = []
    for ix, im in enumerate(images):
        img = Image.open(im)
        if w != 0 and h != 0: img = img.resize((w, h))
        if gray_scale: img = ImageOps.grayscale(img)
        img_arr = np.array(img)
        images_list.append(img_arr)
        if aug_horizontal : images_list_aug.append(np.array(img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)))
    return np.array(images_list) / normalization, np.array(images_list_aug) / normalization # Data Normalization
      

""" Make Functions
"""        
def make_image_list(pathname):
    return sorted(glob.glob(pathname))


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
    dataset_x = make_time_series_data(images_ndarry, step)
    dataset_y = np.array([dataset_index[i:i+step] for i in range(dataset_index.shape[0]-step)])
    return dataset_x, dataset_y


def make_time_series_data(images_ndarry, step):
    dataset_x = np.array([images_ndarry[i:i+step] for i in range(images_ndarry.shape[0]-step)])
    dataset_x = dataset_x.reshape(dataset_x.shape[0], 
                                dataset_x.shape[1], 
                                dataset_x.shape[2], 
                                dataset_x.shape[3], 
                                dataset_x.shape[4]) 
    return dataset_x

