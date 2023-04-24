# TechVidvan Object detection of similar color

import cv2
import numpy as np
'''
==========================================
 Title:  Color Detection
 Author: Lokesh 
 Date:   Jun 2022
==========================================
'''

# Reading the image
cap =cv2.VideoCapture(0)


#define kernel size  
kernel = np.ones((27,27),np.uint8)

while True:
    _,img=cap.read()        

    denoised=cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
    hsv=cv2.cvtColor(denoised,cv2.COLOR_BGR2HSV)
    # convert to hsv colorspace 
    '''grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised=cv2.fastNlMeansDenoising(grey,None,10,10,7,21)
    rgb=cv2.cvtColor(denoised,cv2.COLOR_GRAY2BGR)
    hsv=cv2.cvtColor(rgb,cv2.COLOR_BGR2HSV)'''

    # lower bound and upper bound for Green color 
    # lower_bound = np.array([50, 20, 20])     
    # upper_bound = np.array([100, 255, 255])

    # lower bound and upper bound for Yellow color 
    lower_bound = np.array([20, 80, 80])     
    upper_bound = np.array([30, 255, 255])


    # find the colors within the boundaries
    mask = cv2.inRange(hsv, lower_bound, upper_bound)


    # Remove unnecessary noise from mask

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)



    # Segment only the detected region

    segmented_img = cv2.bitwise_and(img, img, mask=mask)

    # Find contours from the mask

    contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contour on segmented image
    
    # output = cv2.drawContours(segmented_img, contours, -1, (0, 0, 255), 3)


    # Draw contour on original image

    output = cv2.drawContours(img, contours, -1, (0, 0, 255), 3)

    # Showing the output

    # cv2.imshow("Image", img)
    cv2.imshow("Output", output)
    if cv2.waitKey(1) == ord('x'):
        break


cv2.destroyAllWindows()

