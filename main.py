import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv
import os

def getData(directory):
  cards = []
  for entry in os.scandir(directory):
      # make sure it's a folder (e.g. 'five of diamonds')
      if entry.is_dir():
          # use .path instead of entry
          for image in os.scandir(entry.path):
              # only images
              if image.name.endswith(('.jpg', '.png', '.jpeg')):  
                  # Read image with OpenCV
                  img = cv.imread(image.path)

                  cards.append(img)
                  print(f"Loaded {image.path} with shape {img.shape}")
  return cards

testData = getData('./archive/test')
validateData = getData('./archive/valid')
#trainData = getData('./archive/train')

