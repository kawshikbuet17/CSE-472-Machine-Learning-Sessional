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

            # if len(images) == 1000:
            #     break
    df = pd.read_csv(label_path)
    labels = df['digit'].values
    labels = labels[:len(images)]

    return images, labels

def preprocess_data(images, labels):
    # detection friendly preprocessing
    for i in range(len(images)):
        images[i] = cv2.dilate(images[i], (3, 3))
        # images[i] = cv2.threshold(images[i], 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        # images[i] = cv2.GaussianBlur(images[i], (3, 3), 0)
        # images[i] = cv2.threshold(images[i], 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # convert to numpy array
    images = np.array(images)/255
    labels = np.array(labels)

    # reshape images to 28x28x1
    images = images.reshape(images.shape[0], 28, 28, 1)
    # normalize images with std and mean
    images = (images - np.mean(images)) / np.std(images)

    return images, labels
