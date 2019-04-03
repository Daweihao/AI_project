import cv2
import numpy as np
image = cv2.imread("earImageDataset/001_dt.jpg")
image = cv2.resize(image,(756,1008))
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
eq = cv2.equalizeHist(gray)         #灰度图片直方图均衡化
left_ear_cascade = cv2.CascadeClassifier(r'haarcascade_mcs_rightear.xml')
if left_ear_cascade.empty():
    raise IOError('Unable to load the left ear cascade classifier xml file')
xx = 1008
yy = 1008
ww = 1008
hh = 1008
for i in range(5):
    for j in range(1,100):
        left_ear = left_ear_cascade.detectMultiScale(eq,scaleFactor = (1+j/100),minNeighbors = i,minSize = (200,300))
        if(len(left_ear) == 1):
            for (x,y,w,h) in left_ear:
                if(x<xx and y<yy):
                    xx = x
                    yy = y
                    ww = w
                    hh = h
cv2.rectangle(image, (xx,yy), (xx+ww,yy+hh), (0,255,0), 3)
cv2.imshow('Ear Detector', cv2.resize(image,(378,504)))
cv2.imshow('Ear Only', image[yy:yy+hh,xx:xx+ww])
cv2.waitKey(0)
cv2.destroyAllWindows()
