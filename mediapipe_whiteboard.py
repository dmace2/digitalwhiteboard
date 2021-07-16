# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 10:20:33 2021

@author: dylanmace
"""
import cv2
import copy
import numpy as np
import mediapipe as mp
from mediapipe_finger_methods import finger_angles, pointer_position

class Whiteboard:
    
    def __init__(self):
        self.hand = None

        self.cam = cv2.VideoCapture(2)
        self.frame_shape = self.cam.read()[-1].shape[:2]

        self.whiteboard = np.zeros(self.frame_shape)
        self.overlay = copy.deepcopy(self.whiteboard)

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands

    def draw(self, landmarks):
        angles = finger_angles(landmarks)
        prob = (angles < 0.2) * 1.0
        n_fingers = np.count_nonzero(prob)


        pos = pointer_position(landmarks)        
        x = int(pos[0] * self.frame_shape[1])
        y = int(pos[1] * self.frame_shape[0])

        
        #index finger open = draw on whiteboard
        if n_fingers == 1 and prob[0] == 1:
            cv2.circle(self.whiteboard, (x,y), radius=7, color=(255,0,0), thickness=-1)
            
            self.overlay = copy.deepcopy(self.whiteboard)
        
        # two fingers detected: INDEX + MIDDLE | action: show pointer
        elif n_fingers == 2 and prob[0] == 1.0 and prob[1] == 1.0:
            self.overlay = copy.deepcopy(self.whiteboard)
            cv2.circle(self.overlay, (x,y), radius=5, color=(255,0,0), thickness=2)

        # five fingers detected | action:  erase 
        elif n_fingers == 4 :
            cv2.circle(self.whiteboard, (x,y), radius=30, color=(0,0,0), thickness=-1)
            self.overlay = copy.deepcopy(self.whiteboard)
            cv2.circle(self.overlay, (x,y), radius=30, color=(255,0,0), thickness=2)
#
        # two fingers detected: INDEX + PINKY | action: clean whiteboard
        elif n_fingers == 2 and prob[0] == 1.0 and prob[3] == 1.0:
            self.whiteboard = np.zeros(self.frame_shape, np.uint8)
            self.overlay = copy.deepcopy(self.whiteboard)
#        
        # three fingers detected: INDEX + MIDDLE + RING | action: save whiteboard
        elif n_fingers == 3 and prob[0] == 1.0 and prob[1] == 1.0 and prob[2] == 1.0:
            cv2.imwrite('saved/whiteboard.jpg', self.whiteboard)
            print('-- whiteboard.jpg saved! ')
            self.info_whiteboard = copy.deepcopy(self.whiteboard)



    def run(self):
        with self.mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
            while self.cam.isOpened():
                success, image = self.cam.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    # If loading a video, use 'break' instead of 'continue'.
                    continue
                # Flip the image horizontally for a later selfie-view display, and convert
                # the BGR image to RGB.
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                results = hands.process(image)

                # Draw the hand annotations on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.draw(hand_landmarks)
                        self.mp_drawing.draw_landmarks(
                            image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                cv2.imshow('MediaPipe Hands', image)
                cv2.imshow('Whiteboard', self.overlay)

                if cv2.waitKey(5) & 0xFF == 27: #if need to break
                    break
            self.cam.release()
            cv2.destroyAllWindows()



if __name__ == '__main__':
    white = Whiteboard()
    white.run()