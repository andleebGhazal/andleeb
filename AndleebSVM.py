# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 11:37:10 2023

@author: lenovo
"""
from PIL import Image
import cv2
import numpy as np
import os

path=r"C:\Users\lenovo\Desktop\DL folder"
os.chdir(path)


def svm_train(X, y, learning_rate=0.01, num_epochs=1000):
 num_samples, num_features = X.shape
 weights = np.zeros(num_features)
 bias = 0
 for epoch in range(num_epochs):
  for l in range(num_samples):
   condition = y[l] * (np.dot(X[l], weights) + bias)
   if epoch == 0:
    epoch_divisor = 1 # Avoid division by zero in the first iteration
   else:
    epoch_divisor = epoch
 
   if condition >= 1:
    weights -= learning_rate * (2 / epoch_divisor * weights)
   else:
    weights -= learning_rate * (2 / epoch_divisor * weights - np.dot(X[l], y[l]))
    bias -= learning_rate * y[l]
 return weights, bias
def svm_predict(X, weights, bias):
 prediction = np.dot(X, weights) + bias
 return np.sign(prediction)
#importing dataset
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
    empty_images.append((img, -1))# Change label to 0 for "without objec



# Combining and shuffling the datasets
all_images = duck_images + empty_images

np.random.shuffle(all_images)
# Separating data and labels
data = np.array([item[0] for item in all_images])
labels = np.array([item[1] for item in all_images])
# Spliting the data into training 70% and testing 30%
split_ratio = 0.6
split_index = int(split_ratio * len(data))
train_data = data[:split_index]
train_labels = labels[:split_index]
test_data = data[split_index:]
test_labels = labels[split_index:]
# Training the SVM model on the training data
learning_rate = 0.01
num_epochs = 1000
weights, bias = svm_train(train_data, train_labels, learning_rate, num_epochs)
predictions = svm_predict(test_data, weights, bias)
# Calculating accuracy
correct_predictions = np.sum(predictions == test_labels)
total_predictions = len(test_labels)
accuracy = correct_predictions / total_predictions * 100
print(f"Accuracy: {accuracy:.2f}%")
query_img = cv2.imread('query_image.jpeg')
testimage_array = np.array(query_img)
test_img = cv2.resize(testimage_array, (64, 64))
test_data = test_img.flatten()

# Predicting if the image has candy or not
prediction = svm_predict(test_data, weights, bias)
if prediction == 1:
 print("Duck(object) found.")
else:
 print("Duck(object) not found.")
