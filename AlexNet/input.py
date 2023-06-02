import cv2
import numpy as np
from PIL import Image
import torch
import torchvision



def downsample(img, min_size=256):
    h, w, _ = img.shape
    factor = min_size/min(h, w)
    new_dims = (int(h*factor),int(w*factor))
    return cv2.resize(img, new_dims)


def crop(img, size=256):
    h, w, _ = img.shape
    x = w//2 - size//2
    y = h//2 - size//2
    return img[y:y+size, x:x+size]


def substract_mean(img):
    # In the original paper they substract the mean across all images of the training set. 
    # Since here the idea is replicating the architecture only, we substract the mean for each image separately.
    img = img.astype('float64')
    img -= np.mean(img)
    return img 


def transform(img):
    img = downsample(img)
    img = crop(img)
    img = substract_mean(img)
    return torchvision.transforms.ToTensor()(img)


img = cv2.imread("nexttattoo.jpg")
print(transform(img))