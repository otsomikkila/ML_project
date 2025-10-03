import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv
import os

def getData(directory):
  cards, labels = [], []
  for entry in os.scandir(directory):
      if entry.is_dir(): 
          label = entry.name
          for image in os.scandir(entry.path):
              if image.name.endswith(('.jpg', '.png', '.jpeg')): 
                  img = cv.imread(image.path)         # loads as BGR NumPy array

                  cards.append(img)
                  labels.append(label)
                  #print(f"Loaded {image.path} with shape {arr.shape}")
  return np.array(cards), np.array(labels)

X_test, y_test = getData('./archive/test')
X_val, y_val = getData('./archive/valid')
#X_test, y_test = getData('./archive/train')

