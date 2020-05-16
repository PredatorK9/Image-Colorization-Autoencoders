from skimage.color import rgb2lab
from skimage.io import imread
import cv2
import shutil
import numpy as np
import random
import os

data_dir = './Images/'
dataset_dir = './LabImages/train/flowers/'
valid_dir = './LabImages/valid/flowers/'

os.mkdir('./LabImages')
os.mkdir('./LabImages/train')
os.mkdir(dataset_dir)
os.mkdir('./LabImages/valid')
os.mkdir(valid_dir)

for i, image in enumerate(os.listdir(data_dir)):
    rgb_image = imread(data_dir + image)
    new_image = rgb_image/255
    lab_image = rgb2lab(rgb_image)
    
    # reshaping images to square images of size 224*224
    lab_image = cv2.resize(lab_image, (224, 224))
    
    # normalizing the image before hand
    lab_image[:, :, 0] = lab_image[:, :, 0]/100
    lab_image[:, :, 1] = lab_image[:, :, 1]/128
    lab_image[:, :, 2] = lab_image[:, :, 2]/128

    
    # making the image channels first
    lab_image = np.moveaxis(lab_image, 2, 0)

    # saving the images as .npy files
    np.save(dataset_dir + f'LabImage_{i+1:04d}.npy', lab_image)

print('Converted and Normalized...')

num_valid = int(len(os.listdir(dataset_dir)) * 0.1)

for i in range(num_valid):
    valid_image = random.choice(os.listdir(dataset_dir))
    shutil.move(dataset_dir + valid_image, valid_dir + valid_image)

print('Dataset splited...')