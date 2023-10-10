# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 23:05:53 2023

@author: lenovo
"""
from PIL import Image
import cv2
import numpy as np
import os

path=r"C:\Users\lenovo\Desktop\DL folder"
os.chdir(path)

def euclidean_distance(x1, x2):
 return np.sqrt(np.sum((x1 - x2) ** 2))
def knn_predict(train_data, train_labels, test_data, k=1):
 predictions = []
 for test_sample in test_data:
  distances = [euclidean_distance(test_sample, train_sample) for train_sample in train_data]
  nearest_indices = np.argsort(distances)[:k]
  nearest_labels = [train_labels[i] for i in nearest_indices]
  prediction = np.argmax(np.bincount(nearest_labels))
  predictions.append(prediction)
 return predictions

duck_images = []
# Define the filenames of your images
image_filenames1 = ['image1.jpeg', 'image2.jpeg', 'image3.jpeg', 'image4.jpeg', 'image5.jpeg', 'image6.jpeg', 'image7.jpeg','image8.jpeg', 'image9.jpeg','image10.jpeg']

# Use a for loop to load and append each image to the list
for i in image_filenames1:
    img = Image.open(i)
    image_array = np.array(img)
    img = cv2.resize(image_array, (64, 64))
    img = img.flatten()
    duck_images.append((img, 1))
image_filenames2 = ['image11.jpeg', 'image12.jpeg', 'image13.jpeg', 'image14.jpeg', 'image15.jpeg', 'image16.jpeg', 'image17.jpeg','image18.jpeg', 'image19.jpeg','image20.jpeg']

empty_images = []
for j in image_filenames2:
    img = Image.open(j)
    image_array = np.array(img)
    img = cv2.resize(image_array, (64, 64))
    img = img.flatten()
    empty_images.append((img, 0))# Change label to 0 for "without object"

# Combining and shuffling the datasets
all_images = duck_images + empty_images
np.random.shuffle(all_images)
# Separating data and labels
data = np.array([item[0] for item in all_images])
labels = np.array([item[1] for item in all_images])
# Splitting the data into training 70% and testing 30%
split_ratio = 0.7
split_index = int(split_ratio * len(data))
train_data = data[:split_index]
train_labels = labels[:split_index]
test_data = data[split_index:]
test_labels = labels[split_index:]
# Using k-NN to predict labels for the testing data
k = 1
predictions = knn_predict(train_data, train_labels, test_data, k)
# Calculating accuracy
correct_predictions = np.sum(predictions == test_labels)
total_predictions = len(test_labels)
accuracy = correct_predictions / total_predictions * 100
print(f"Accuracy: {accuracy:.2f}%")

query_img = cv2.imread('query_image.jpeg')
testimage_array = np.array(query_img)
test_img = cv2.resize(testimage_array, (64, 64))
test_data = test_img.flatten()
# Using k-NN to predict if the image has candy or not
prediction = knn_predict(train_data, train_labels, [test_data], k)[0]
if prediction == 1:
 print("Duck(object) found.")
else:
 print("Duck(object) not found.")