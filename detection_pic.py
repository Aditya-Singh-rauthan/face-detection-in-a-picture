# -*- coding: utf-8 -*-
"""
Created on Thu May 30 02:29:37 2019

@author: Aditya Rauthan
"""

import cv2
import matplotlib.pyplot as plt

face_cascade = cv2.CascadeClassifier('haar_face.xml')
eye_cascade = cv2.CascadeClassifier('frontalEyes35x16.xml')

pic = cv2.imread('DSC_0103.jpg')
#pic=cv2.cvtColor(pic,cv2.COLOR_RGB2BGR)

gray=cv2.cvtColor(pic,cv2.COLOR_RGB2GRAY)
faces=face_cascade.detectMultiScale(gray,1.3,5)
for(x,y,w,h) in faces:
    cv2.rectangle(pic, (x,y), (x+w,y+h), (255, 0, 0), thickness=5)

roi_color=pic[y:y+h, x:x+w ]
roi_gray=gray[y:y+h, x:x+w ]
eyes=eye_cascade.detectMultiScale(roi_gray,1.1, 3)
for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color, (ex,ey), (ex+eh,ey+ew), (255, 0, 0), thickness=5)
cv2.namedWindow('window',cv2.WINDOW_NORMAL)
while True:
    cv2.imshow('window',pic)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

    if cv2.waitKey(1) & 0xFF == ord('a'):
        cv2.imwrite('output1.jpg',pic)
    
        
    
