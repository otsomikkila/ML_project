import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv
import os  # import os module

directory = './archive/test'  # set directory path

cards = []

for entry in os.scandir(directory):
    if entry.is_dir():  # make sure it's a folder (e.g. 'five of diamonds')
        for image in os.scandir(entry.path):  # use .path instead of entry
            if image.name.endswith(('.jpg', '.png', '.jpeg')):  # only images
                # Read image with OpenCV
                img = cv.imread(image.path)         # loads as BGR NumPy array
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # convert to RGB (optional)

                # Convert to NumPy array (though OpenCV already returns np.array)
                arr = np.array(img)

                cards.append(arr)
                print(f"Loaded {image.path} with shape {arr.shape}")

print(f"Total images loaded: {len(cards)}")