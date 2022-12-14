# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 12:08:41 2022

@author: Andleeb
"""

import numpy as np
import cv2
import os
path=r"C:\Users\Andleeb\Desktop"
os.chdir(path)
img = cv2.imread('mario.png')
#print(img.max())
#print(img.min())
cv2.imshow('image', img) 
cv2.waitKey(10)
image_data=np.array(img)
print(image_data)
print ("Dimensions of image are:", image_data.ndim)
print ("Shape of image is", image_data.shape)
print ("Size of image is", image_data.size)
reshape = image_data.reshape(50573,18,9)
print ("reshape array image is:",reshape)
print(type(img))
print ("transposed array image is:", image_data.T)#transposing array a using array.T

