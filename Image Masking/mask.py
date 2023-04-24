import cv2
cap=cv2.VideoCapture(0)

while True:
    _,frame = cap.read()
    grey=cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
    
    denoised=cv2.fastNlMeansDenoising(grey,None,20,7,9)

    canny =cv2.Canny(denoised, 120, 120)
    cv2.imshow('frame',canny)
    if cv2.waitKey(1) == ord('x'):
        break
