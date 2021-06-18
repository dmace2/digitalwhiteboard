#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 10:23:41 2021

@author: allisonai
"""

import cv2

vid = cv2.VideoCapture(0)

while True:
    ret,frame = vid.read()
    cv2.imshow('frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
vid.release()
cv2.destroyAllWindows()