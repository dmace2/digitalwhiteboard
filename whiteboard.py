#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 10:20:33 2021

@author: xavier
"""

import cv2
import numpy as np
from unified_detector import Fingertips
from hand_detector.detector import SOLO, YOLO


class whiteboard:
    
    def __init__(self, hand_detection_method = 'yolo'):
        
        if hand_detection_method is 'solo':
            self.hand = SOLO(weights='weights/solo.h5', threshold=0.8)
        elif hand_detection_method is 'yolo':
            self.hand = YOLO(weights='weights/yolo.h5', threshold=0.8)
        else:
            assert False, "'" + hand_detection_method + \
                          "' hand detection does not exist. use either 'solo' or 'yolo' as hand detection method"
        
        self.fingertips = Fingertips(weights='weights/fingertip.h5')
        
        self.cam = cv2.VideoCapture(0)
        
        self.frame_shape = self.cam.read()[-1].shape
        
        self.whiteboard = np.zeros(self.frame_shape[:2])
        
        print('Unified Gesture & Fingertips Detection')
    
    def run(self):
        while True:
            ret, image = self.cam.read()
            if ret is False:
                break
        
            # hand detection
            tl, br = self.hand.detect(image=image)
        
            if tl and br is not None:
                cropped_image = image[tl[1]:br[1], tl[0]: br[0]]
                height, width, _ = cropped_image.shape
        
                # gesture classification and fingertips regression
                prob, pos = self.fingertips.classify(image=cropped_image)
                pos = np.mean(pos, 0)
                spos = pos.reshape((-1,2))
        
                # post-processing
                prob = np.asarray([(p >= 0.5) * 1.0 for p in prob])
#                for i in range(0, len(pos), 2):
#                    pos[i] = pos[i] * width + tl[0]
#                    pos[i + 1] = pos[i + 1] * height + tl[1]
                for i in range(len(spos)):
                    spos[i][0] = spos[i][0] * width + tl[0]
                    spos[i][1] = spos[i][1] * height + tl[1]

                # drawing
                index = 0
                color = [(15, 15, 240), (15, 240, 155), (240, 155, 15), (240, 15, 155), (240, 15, 240)]
                image = cv2.rectangle(image, (tl[0], tl[1]), (br[0], br[1]), (235, 26, 158), 2)
                for c, p in enumerate(prob):
                    if p > 0.5:
                        x,y = int(spos[index][0]), int(spos[index][1])
                        image = cv2.circle(image, (x, y), radius=12,
                                           color=color[c], thickness=-2)
                        self.whiteboard[y:y + 10,x:x + 10] = 255
                    index += 1
        
            if cv2.waitKey(1) & 0xff == 27:
                break
        
            # display image
            cv2.imshow('Unified Gesture & Fingertips Detection', image)
            cv2.imshow('Whiteboard', self.whiteboard)
    
        self.cam.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    white = whiteboard()
    white.run()