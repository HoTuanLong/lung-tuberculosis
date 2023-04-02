import os, sys
from libs import *
from data import ImageDataset

dataset = ImageDataset(
    data_dir = "../../dataset/CXR-TB/train/", 
    augment = True, 
)

image, label = dataset[0]
print(image.shape, label)