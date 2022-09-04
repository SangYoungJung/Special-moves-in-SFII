import os
import glob
import json
import numpy as np
import pickle
from PIL import Image, ImageOps

# Creation data array for moves info of each image
images = sorted(glob.glob('./dataset_rnn/*.jpg'))
dataset_index = np.zeros((len(images),1))
dataset_class_name = {"None":0}
dataset_class_num = 0


# Load classification(or moves) information from json which is defined.
with open('./dataset_rnn/ken_moves.json') as f:
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
dataset_class_num = len(np.unique(dataset_index))
print(dataset_index.shape)
print(dataset_class_num, dataset_class_name)


# Load images to dataset with resizing 100 by 100
images_list = []
for ix, im in enumerate(images):
    img = Image.open(im)
    img_resize = img.resize((100, 100)) # Resize(of Fixed size) Image.LANCZOS
    img_arr = np.array(img_resize) # ImageOps.grayscale(img_resize)
    images_list.append(img_arr)
images_ndarry = np.array(images_list) / 255 # Data Normalization


# Make time series data as 10 frame and classes are paired 
# train_x : (#, step, img_w, img_h, img_channel) (#, 10, 100, 100, 3)
# train_y : (#, step, moves of index)            (#, 10, 1)
step = 10
dataset_x = np.array([images_ndarry[i:i+step] for i in range(images_ndarry.shape[0]-step)])
dataset_x = dataset_x.reshape(dataset_x.shape[0], 
                              dataset_x.shape[1], 
                              dataset_x.shape[2], 
                              dataset_x.shape[3], 
                              dataset_x.shape[4]) 
dataset_y = np.array([dataset_index[i:i+step] for i in range(dataset_index.shape[0]-step)])


# Create a pickle(or serialized) for dataset
dataset = dict()
dataset['dataset_x'] = dataset_x
dataset['dataset_y'] = dataset_y
with open('movesKen.pickle', 'wb') as handle:
    pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)



"""
import pickle

your_data = {'foo': 'bar'}

# Store data (serialize)
with open('filename.pickle', 'wb') as handle:
    pickle.dump(your_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Load data (deserialize)
with open('filename.pickle', 'rb') as handle:
    unserialized_data = pickle.load(handle)

print(your_data == unserialized_data)
"""