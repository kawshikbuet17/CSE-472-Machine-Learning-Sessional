import numpy as np
import os
import cv2
import pandas as pd

def load_data(image_folder, label_path):
    images = []
    for filename in os.listdir(image_folder):
        img = cv2.imread(os.path.join(image_folder, filename))
        if img is not None:
            # convert to grayscale
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # resize to 28x28
            img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_CUBIC)
            # convert to float32
            img = img.astype(np.float32)
            images.append(img)

            if len(images) == 500:
                break
    df = pd.read_csv(label_path)
    labels = df['digit'].values
    labels = labels[:len(images)]

    return images, labels

def preprocess_data(images, labels):
    # rotate images iwth 90, 180, 270 degrees and coreesponding labels
    for i in range(len(images)):
        images.append(np.rot90(images[i], 1))
        labels = np.append(labels, labels[i])
        images.append(np.rot90(images[i], 1))
        labels = np.append(labels, labels[i])
        images.append(np.rot90(images[i], 1))
        labels = np.append(labels, labels[i])

    # convert to numpy array
    images = np.array(images)
    labels = np.array(labels)

    # reshape images to 28x28x1
    images = images.reshape(images.shape[0], 28, 28, 1)
    # normalize images with std and mean
    images = (images - np.mean(images)) / np.std(images)

    return images, labels
