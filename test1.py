import cv2


source=cv2.imread('org.jpg')
deft=cv2.imread('deft.jpg')

grey1=cv2.cvtColor(source,cv2.COLOR_RGB2GRAY)
grey2=cv2.cvtColor(deft,cv2.COLOR_RGB2GRAY)

ret1,th1 = cv2.threshold(grey1,127,255,cv2.THRESH_BINARY)
ret2,th2 = cv2.threshold(grey2,127,255,cv2.THRESH_BINARY)
res = cv2.bitwise_xor(th1, th2)   
sub = cv2.subtract(th1,th2)
new_img= None
contours, hierarchy = cv2.findContours(th1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for c in contours:
	x,y,w,h = cv2.boundingRect(c)
	if w>50 and h>50:
		new_img=source[y:y+h,x:x+w]

cv2.imshow('Threshold',th1)
cv2.imshow('Thershold',th2)
cv2.imshow('Sub',sub)
cv2.imshow('Result', res)
cv2.imshow('rr',new_img)

cv2.waitKey(0)
