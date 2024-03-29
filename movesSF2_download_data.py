#===============================================================================
#
#   File name   : movesSF2_download_data.py
#   Author      : lycobs@gmail.com
#   Created date: 2022-10
#   Description : Downloading and Preparing data
#
#===============================================================================


import os
import gdown
import zipfile
import argparse


#===============================================================================
# Functions
#===============================================================================


def download(url, output, overwrite=False):
    if overwrite is False: 
        if os.path.isfile(output): return False
    
    try: gdown.download(url, output, quiet=False)
    except: return False
    return True


def unzip(input, output, overwrite=False):
    if overwrite is False: 
        if os.path.isdir(output): return False
    zipfile.ZipFile(input).extractall(output)
    return True
    

#===============================================================================
# Main
#===============================================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=str, default='./dataset')
    args = parser.parse_args()
    
    current_dir = './'
    dataset_dir = args.directory
    try: os.makedirs(dataset_dir)
    except FileExistsError: pass
    
    urls = []
    urls.append('https://drive.google.com/uc?id=' + '1FUz3hGOIBvuA-6nq3dJyv-GX8KKaVSOk')
    urls.append('https://drive.google.com/uc?id=' + '1on4TI3zEtngz9y93dfHXgPgWG80EIvJS')
    urls.append('https://drive.google.com/uc?id=' + '15V1WtWts4IHsUGSbykIGXpRRAszqv3EM')
    urls.append('https://drive.google.com/uc?id=' + '1rfJi0Wur9EryptYg8PeT89-Ma6J4-4iq')
    urls.append('https://drive.google.com/uc?id=' + '1xQsP0HJ61AZzdKQZkWpObEECGjfcXplw')
    urls.append('https://drive.google.com/uc?id=' + '1DvdjAOgwXv4gzDn-wIbHux4BHcARZo-P')
    
    downloads = []
    downloads.append(dataset_dir + '/yolo_image_dataset.zip')
    downloads.append(dataset_dir + '/time_series_image_dataset.zip')
    downloads.append(dataset_dir + '/yolo_video.zip')
    downloads.append(current_dir + '/movesSF2_pre_trained.h5')
    downloads.append(current_dir + '/yolov3-tiny.weights')
    downloads.append(current_dir + '/yolov3-tiny-final-sf2.weights')
    
    outputs = []
    outputs.append(dataset_dir + '/images')
    outputs.append(dataset_dir + '/images_moves')
    outputs.append(dataset_dir + '/videos')
    outputs.append(current_dir)
    outputs.append(current_dir)
    outputs.append(current_dir)
    
    for idx in range(len(urls)):
        success = download(urls[idx], downloads[idx])
        if downloads[idx].lower().endswith(('.zip')) and success:
            unzip(downloads[idx], outputs[idx])

