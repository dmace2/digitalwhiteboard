# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 10:20:33 2021

@author: dylanmace
"""
import cv2
import copy
import numpy as np
import mediapipe as mp
from spaghetti import finger_angles

class Whiteboard:
    
    def __init__(self):
        self.hand = None

        self.cam = cv2.VideoCapture(2)
        self.frame_shape = self.cam.read()[-1].shape[:2]

        self.whiteboard = np.zeros(self.frame_shape)
        self.overlay = copy.deepcopy(self.whiteboard)

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands

    def draw(self):
        pass

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
                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                results = hands.process(image)

                # Draw the hand annotations on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        print(finger_angles(hand_landmarks))
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