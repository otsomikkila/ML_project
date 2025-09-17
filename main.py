import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv
import os  # import os module

directory = './archive/test'  # set directory path

cards = []

for entry in os.scandir(directory):
    if entry.is_dir():  # make sure it's a folder (e.g. 'five of diamonds')
        for image in os.scandir(entry.path):  # use .path instead of entry
            pic = image.path  # full path to image
            print(pic)
            card = np.load(pic)  # only if it's a .npy file
            cards.append(card)

print(cards)