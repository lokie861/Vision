'''
==========================================
 Title:  Image Comparison
 Author: Lokesh 
 Date:   July 2022
==========================================
'''


import cv2
import numpy as np
from matplotlib import pyplot as plt
img1 = cv2.imread('org.jpg')  #reading the images
img2 = cv2.imread('deft.jpg')
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)        #converting to gray scale
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
ret,thresh1 = cv2.threshold(gray1,200,255,cv2.THRESH_BINARY_INV)  #applying threshold (binary) 
ret,thresh2 = cv2.threshold(gray2,200,255,cv2.THRESH_BINARY_INV)
res =cv2.bitwise_xor(thresh1, thresh2, mask=None)      #comparing the images 

cv2.imshow('orginal',thresh1)
cv2.imshow('defect',thresh2)
cv2.imshow('Bitwise XOR', res)
cv2.waitKey(0)
cv2.destroyAllWindows()

