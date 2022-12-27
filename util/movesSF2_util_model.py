#===============================================================================
#
#   File name   : movesSF2_util_model.py
#   Author      : lycobs@gmail.com
#   Created date: 2022-10
#   Description : General helpful functions for RNNs as recognition moves SF2
#
#===============================================================================


import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras


def gpu_support():
    print("physical_devices :", tf.config.list_physical_devices('GPU'))
    print("gpu support :", tf.test.is_built_with_gpu_support())


def name(moves_num):
    return 'Hadoken' if moves_num == 1 else 'Shoryuken' if moves_num == 2 else 'Tatsumaki' if moves_num == 3 else ''


def judgement(result, judgement_time_step):
    continuous = 0
    prev_moves_num = result[-1]
    for moves_num in np.flip(result):
        if prev_moves_num == moves_num: continuous += 1
        if continuous >= judgement_time_step: return name(moves_num)
    return name(0)


def load_model(name):
    # load of pre-trained model as movesSF2
    model = tf.keras.models.load_model(name)
    model_config = model.get_config()
    input_time_steps = model_config["layers"][0]["config"]["batch_input_shape"][1]
    input_image_size = model_config["layers"][0]["config"]["batch_input_shape"][2:]
    print("Input shape : ", model_config["layers"][0]["config"]["batch_input_shape"])
    return model, input_time_steps, input_image_size


def model_info(model):
    space_length = 25
    for idx, layer in enumerate(model.layers):
        print("".ljust(space_length+5, '-'))
        print("Layer:".ljust(space_length), idx)
        print("weights:".ljust(space_length), len(layer.weights))
        print("trainable_weights:".ljust(space_length), len(layer.trainable_weights))
        print("non_trainable_weights:".ljust(space_length), len(layer.non_trainable_weights))
        print("trainable:".ljust(space_length), layer.trainable)
    print("".ljust(space_length+5, '-'))


def inference(model, time_step_input):
    input = np.array(time_step_input) / 255              
    result = model.predict(np.expand_dims(input, axis=0))
    argmax_result = np.argmax(result.squeeze(), axis=1)
    return argmax_result


def inference_summary(time_step_input, argmax_result, judgement_time_step, text_align='left'):
    # showing output image after carrying out yolo
    numpy_horizontal = np.array(np.hstack(time_step_input[:]), dtype = np.uint8)
    numpy_horizontal = cv2.cvtColor(numpy_horizontal, cv2.COLOR_RGB2BGR)
    
    ret = judgement(argmax_result, judgement_time_step)
    textsize = cv2.getTextSize(ret, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    align_offset = 1 if text_align == 'left' else 2 if text_align == 'center' else 10e3
    textX = (numpy_horizontal.shape[1] - textsize[0]) / align_offset
    textY = (numpy_horizontal.shape[0] + textsize[1]) / 2
    numpy_horizontal = cv2.putText(numpy_horizontal, ret, (int(textX), int(textY)), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA).copy()
        
    return numpy_horizontal

