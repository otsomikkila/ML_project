import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv
import os
import json
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
#from skimage.feature import hog

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications.efficientnet import preprocess_input


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
                  #cards.append(img)
                  #labels.append(label)
                  #print(f"Loaded {image.path} with shape {arr.shape}")
  return np.array(cards), np.array(labels)


#EKA CNN MODEL
BATCH_SIZE = 32
IMG_SIZE = (128, 128)
EPOCHS = 30

train_ds = tf.keras.utils.image_dataset_from_directory(
    './archive/train',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    './archive/valid',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    './archive/test',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)


# Class names are inferred from subfolder names
class_names = train_ds.class_names
num_classes = len(class_names)
print(f"Found {num_classes} classes: {class_names}")

def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

AUTOTUNE = tf.data.AUTOTUNE

train_ds = (
    train_ds
    .map(preprocess, num_parallel_calls=AUTOTUNE)
    .cache()
    .shuffle(1000)
    .prefetch(buffer_size=AUTOTUNE)
)

val_ds = (
    val_ds
    .map(preprocess, num_parallel_calls=AUTOTUNE)
    .cache()
    .prefetch(buffer_size=AUTOTUNE)
)

test_ds = (
    test_ds
    .map(preprocess, num_parallel_calls=AUTOTUNE)
    .prefetch(buffer_size=AUTOTUNE)
)

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
], name="data_augmentation")

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(*IMG_SIZE, 3)),
    data_augmentation,
    
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2),
    
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2),
    
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2),

    tf.keras.layers.Conv2D(256, (3,3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(2),
  
  
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

callbacks_list = [
    tf.keras.callbacks.ModelCheckpoint(
        'best_card_model_3.h5',
        save_best_only=True,
        monitor='val_accuracy',
        mode='max',
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=6,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks_list
)

test_loss, test_acc = model.evaluate(test_ds)
print(f"\n Test Accuracy: {test_acc:.4f}")
print(f" Test Loss: {test_loss:.4f}")

plt.figure(figsize=(14, 5))

plt.subplot(1, 3, 1)
plt.plot(history.history['accuracy'], label='Train Acc', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Val Acc', linewidth=2)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Model Accuracy', fontsize=14)
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1, 3, 2)
plt.plot(history.history['loss'], label='Train Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Model Loss', fontsize=14)
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1, 3, 3)
train_val_gap = np.array(history.history['accuracy']) - np.array(history.history['val_accuracy'])
plt.plot(train_val_gap, linewidth=2, color='red')
plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Train - Val Accuracy', fontsize=12)
plt.title('Overfitting Monitor', fontsize=14)
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("training_metrics3.png", dpi=150)
plt.close()

#No autotune for confusion matrix
test_ds_seq = tf.keras.utils.image_dataset_from_directory(
    './archive/test',
    image_size=(128, 128),   
    batch_size=1, 
    shuffle=False,
    label_mode='int'
)
test_ds_seq = test_ds_seq.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))

y_true = np.concatenate([y.numpy() for x, y in test_ds_seq], axis=0)
y_pred_probs = model.predict(test_ds_seq, verbose = 1)
y_pred = np.argmax(y_pred_probs, axis=1)

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
fig, ax = plt.subplots(figsize=(10,10))
disp.plot(ax=ax, xticks_rotation=90)
plt.savefig("confusion_matrix_fixed.png")
plt.close()
print("Saved confusion matrix to confusion_matrix_fixed.png")

print("\n=== Classification Report ===")
report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
print(report)

# Save report to file
with open('classification_report.txt3', 'w') as f:
    f.write(report)
print("Saved classification report to classification_report3.txt")

per_class_acc = cm.diagonal() / cm.sum(axis=1)
print("\n=== Per-Class Accuracy ===")
for i, (name, acc) in enumerate(zip(class_names, per_class_acc)):
    print(f"{name:20s}: {acc:.4f}")

worst_indices = np.argsort(per_class_acc)[:5]
print("\n Worst Performing Classes:")
for idx in worst_indices:
    print(f"  {class_names[idx]:20s}: {per_class_acc[idx]:.4f}")

with open('label_map.json', 'w') as f:
    json.dump({i: name for i, name in enumerate(class_names)}, f)
print("Saved label map to label_map.json")

"""
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
"""








