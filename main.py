import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from skimage.feature import hog

# attempt to make knn faster by only looking at the top left of each corner and not whole card
def extract_corner(image, corner_size=(150, 150)):
    ch, cw = corner_size

    # Crop top-left corner
    corner = image[0:ch, 0:cw]

    # gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)                # grayscale
    # edges = cv.Canny(gray, threshold1=50, threshold2=150)        # edge detection
    # features = hog(gray, pixels_per_cell=(8,8), cells_per_block=(2,2), orientations=9, visualize=False)
    # corner_rgb = cv.cvtColor(corner, cv.COLOR_BGR2RGB)
    # plt.imshow(edges)
    # plt.title("Cropped Corner")
    # plt.axis('off')
    # plt.show()

    return corner


def getData(directory):
  cards, labels = [], []
  for entry in os.scandir(directory):
      if entry.is_dir(): 
          label = entry.name
          for image in os.scandir(entry.path):
              if image.name.endswith(('.jpg', '.png', '.jpeg')): 
                  img = cv.imread(image.path)         # loads as BGR NumPy array

                  if img is not None:
                    # Crop corner
                    corner = extract_corner(img)
                    
                    cards.append(corner.flatten())
                    labels.append(label)
                  else:
                    print(f"Warning: Failed to load {image.path}")

                  # cards.append(img)
                  # labels.append(label)
                  #print(f"Loaded {image.path} with shape {arr.shape}")
  return np.array(cards), np.array(labels)

X_test, y_test = getData('./archive/test')
X_val, y_val = getData('./archive/valid')
X_train, y_train = getData('./archive/train')



knn = KNeighborsClassifier(n_neighbors=4, metric='cosine', weights='distance')

# X_tr = X_train.reshape(X_train.shape[0], -1)
knn.fit(X_train, y_train)

# X_te = X_test.reshape(X_test.shape[0], -1)
# X_va = X_val.reshape(X_val.shape[0], -1)
y_pred = knn.predict(X_val)

print("Accuracy:", accuracy_score(y_val, y_pred))

# Compute the confusion matrix
cm = confusion_matrix(y_val, y_pred)

# print("Confusion Matrix:\n", cm)

# Optional: Plot it
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=knn.classes_)
disp.plot(cmap='Blues', xticks_rotation=90)
plt.show()
