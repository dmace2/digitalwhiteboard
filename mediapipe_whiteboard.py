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
from LRUCache import LRUCache
from tkinter import *
from PIL import ImageTk, Image
import threading
import _thread


class Whiteboard:
    
    def __init__(self, max_frame_buffer_len=7, video_capture=2):
        self.tk = Tk()
        self.tk.resizable(False, False)
        self.canvas = Canvas(self.tk, width=1500, height=750)
        self.add_color_buttons()
        self.canvas.pack()

        self.hand = None

        self.cam = cv2.VideoCapture(video_capture)
        self.frame_shape = self.cam.read()[-1].shape

        self.whiteboard = np.zeros(self.frame_shape)
        self.overlay = copy.deepcopy(self.whiteboard)

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands

        self.cache = LRUCache(max_frame_buffer_len)

        # _thread.start_new_thread(self.update_gui, (False,))

        # self.tk.mainloop()
        while True:
            self.update_gui(False)

    def add_color_buttons(self):
        buttonCanvas = Canvas(self.tk)
        
        button = Button(buttonCanvas, text="Red", fg='red')
        button.pack(side=LEFT)

        button = Button(buttonCanvas, text="Orange", fg='orange')
        button.pack(side=LEFT)

        button = Button(buttonCanvas, text="Yellow", fg='yellow')
        button.pack(side=LEFT)

        button = Button(buttonCanvas, text="Gren", fg='green')
        button.pack(side=LEFT)

        button = Button(buttonCanvas, text="Blue", fg='blue')
        button.pack(side=LEFT)

        button = Button(buttonCanvas, text="Purple", fg='purple')
        button.pack(side=LEFT)

        button = Button(buttonCanvas, text="White", fg='white')
        button.pack(side=LEFT)

        buttonCanvas.pack()



    def draw(self, landmarks):
        angles = finger_angles(landmarks)
        prob = (angles < 0.2) * 1.0
        n_fingers = np.count_nonzero(prob)

        pos = pointer_position(landmarks)        
        x = int(pos[0] * self.frame_shape[1])
        y = int(pos[1] * self.frame_shape[0])

        print(self.cache.cache)        
        # index finger open = draw on whiteboard
        if n_fingers == 1 and prob[0] == 1:
            if self.cache.cache_equal("draw"):
                cv2.circle(self.whiteboard, (x,y), radius=7, color=(255, 255, 255), thickness=-1)
                self.overlay = copy.deepcopy(self.whiteboard)
            self.cache.add("draw")
        
        # two fingers detected: INDEX + MIDDLE | action: show pointer
        elif n_fingers == 2 and prob[0] == 1.0 and prob[1] == 1.0:
            if self.cache.cache_equal("move"):
                self.overlay = copy.deepcopy(self.whiteboard)
                cv2.circle(self.overlay, (x, y), radius=5, color=(255, 255, 255), thickness=2)
            self.cache.add("move")

        # five fingers detected | action:  erase 
        elif n_fingers == 4 :
            if self.cache.cache_equal("erase"):
                cv2.circle(self.whiteboard, (x, y), radius=30, color=(0, 0, 0), thickness=-1)
                self.overlay = copy.deepcopy(self.whiteboard)
                cv2.circle(self.overlay, (x, y), radius=30, color=(255, 255, 255), thickness=2)
            self.cache.add("erase")
#
        # two fingers detected: INDEX + PINKY | action: clean whiteboard
        elif n_fingers == 2 and prob[0] == 1.0 and prob[3] == 1.0:
            if self.cache.cache_equal("clear"):
                self.whiteboard = np.zeros(self.frame_shape, np.uint8)
                self.overlay = copy.deepcopy(self.whiteboard)
            self.cache.add("clear")
#        
        # three fingers detected: INDEX + MIDDLE + RING | action: save whiteboard
        elif n_fingers == 3 and prob[0] == 1.0 and prob[1] == 1.0 and prob[2] == 1.0:
            if self.cache.cache_equal("save"):
                cv2.imwrite('saved/whiteboard.jpg', self.whiteboard)
                print('-- whiteboard.jpg saved! ')
                self.overlay = copy.deepcopy(self.whiteboard)
            self.cache.add("save")

    def update_gui(self, flip=False):
        with self.mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands = 1) as hands:
            while self.cam.isOpened():
                success, image = self.cam.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    # If loading a video, use 'break' instead of 'continue'.
                    continue
                if flip:
                    # Flip the image horizontally for a later selfie-view display, and convert
                    # the BGR image to RGB.
                    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                else:
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

                # cv2.imshow('MediaPipe Hands', image)
                video = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                videotk = ImageTk.PhotoImage(video)
                self.canvas.create_image((25, 250), image=videotk, anchor=NW)

                # cv2.imshow('Whiteboard', self.overlay)
                whiteboard = Image.fromarray(self.overlay.astype(np.uint8))
                wbtk = ImageTk.PhotoImage(whiteboard)
                self.canvas.create_image((825, 250), image=wbtk, anchor = NW)

                if cv2.waitKey(5) & 0xFF == 27: #if need to break
                    break

                self.tk.update_idletasks()
                self.tk.update()
            self.cam.release()
            cv2.destroyAllWindows()




if __name__ == '__main__':
    white = Whiteboard(5)

