'''
==========================================
 Title:  Denoise Image
 Author: Lokesh 
 Date:   Jun 2022
==========================================
'''

import cv2
cap=cv2.VideoCapture(0)

while True:
    _,frame = cap.read()
    grey=cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)

    denoised=cv2.fastNlMeansDenoising(grey,None,10,10,7,21)
    cv2.imshow('frame',denoised)
    if cv2.waitKey(1) == ord('x'):
        break
