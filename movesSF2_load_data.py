
import os
import gdown
import zipfile
import argparse


""" download using gdown 
"""
def download(url, output, overwrite=False):
    if overwrite is False: 
        if os.path.isfile(output): return False
    
    try: gdown.download(url, output, quiet=False)
    except: return False
    return True

""" unzip
"""
def unzip(input, output, overwrite=False):
    if overwrite is False: 
        if os.path.isdir(output): return False
    zipfile.ZipFile(input).extractall(output)
    return True
    

""" main fucntion
"""
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
    
    downloads = []
    downloads.append(dataset_dir + '/yolo_image_dataset.zip')
    downloads.append(dataset_dir + '/time_series_image_dataset.zip')
    downloads.append(dataset_dir + '/yolo_video.zip')
    downloads.append(current_dir + '/movesSF2_pre_trained.h5')
    
    outputs = []
    outputs.append(dataset_dir + '/images')
    outputs.append(dataset_dir + '/images_moves')
    outputs.append(dataset_dir + '/videos')
    outputs.append(current_dir)
    
    for idx in range(len(urls)):
        success = download(urls[idx], downloads[idx])
        if downloads[idx].lower().endswith(('.zip')) and success:
            unzip(downloads[idx], outputs[idx])

