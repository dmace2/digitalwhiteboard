# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 10:20:33 2021

@author: dylanmace
"""
import cv2
import copy
import os
import numpy as np
import mediapipe as mp
from mediapipe_finger_methods import finger_angles, pointer_position
from mediapipe_finger_methods import *
from LRUCache import LRUCache
from tkinter import *

from PIL import ImageTk, Image
import sys


class Whiteboard:
    
    def __init__(self, max_frame_buffer_len=7, video_capture=0, background=None):
        self.tk = Tk()
        self.tk.resizable(False, False)

        self.color_dict = {
            'Red': (255, 0, 0),
            'Green': (0, 255, 0),
            'Blue': (0, 0, 255),
            'Orange': (255, 127, 0),
            'Yellow': (255, 255, 0),
            'Purple': (148, 0, 211),
            'White': (255, 255, 255),
            'Black': (0, 0, 0)
        }

        self.cam = cv2.VideoCapture(video_capture)
        self.frame_shape = self.cam.read()[-1].shape
        print(self.frame_shape)

        self.bkgd_color = 'White'

        if background:
            if os.path.isfile(background):
                self.bkgd = cv2.imread(background)
                self.bkgd = cv2.resize(self.bkgd, (self.frame_shape[1], self.frame_shape[0]))
                self.bkgd = cv2.cvtColor(self.bkgd, cv2.COLOR_BGR2RGB)
            elif background in self.color_dict.keys():
                self.bkgd = cv2.imread(f'backgrounds/{background}.jpg')
                self.bkgd = cv2.resize(self.bkgd, (self.frame_shape[1], self.frame_shape[0]))
                self.bkgd = cv2.cvtColor(self.bkgd, cv2.COLOR_BGR2RGB)
                self.bkgd_color = background
        else:
            self.bkgd = np.ones(self.frame_shape).astype(np.uint8) * 255

        title = Label(self.tk, text=f"AI {self.bkgd_color}board!", font=("Arial", 25))
        title.pack()

        self.canvas = Canvas(self.tk, width=1500, height=500)
        self.canvas.pack(side=TOP, pady=20)

        self.add_ui_elements()

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands

        self.draw_color = (255, 255, 255)

        if background:
            if os.path.isfile(background):
                self.bkgd = cv2.imread(background)
                self.bkgd = cv2.resize(self.bkgd, (self.frame_shape[1], self.frame_shape[0]))
                self.bkgd = cv2.cvtColor(self.bkgd, cv2.COLOR_BGR2RGB)
            elif background in self.color_dict.keys():
                self.bkgd = cv2.imread(f'backgrounds/{background}.jpg')
                self.bkgd = cv2.resize(self.bkgd, (self.frame_shape[1], self.frame_shape[0]))
                self.bkgd = cv2.cvtColor(self.bkgd, cv2.COLOR_BGR2RGB)
        else:
            self.bkgd = np.zeros(self.frame_shape).astype(np.uint8)

        self.whiteboard = copy.deepcopy(self.bkgd)
        self.overlay = copy.deepcopy(self.whiteboard)

        self.action_cache = LRUCache(max_frame_buffer_len)

        self.last_point = [0, 0]
        self.draw_radius = 5

        self.update_gui(False)

    def add_ui_elements(self):
        buttonCanvas = Canvas(self.tk)
        
        button = Button(buttonCanvas, text="Red", bg='red', command=lambda:self.update_draw_color('Red'))
        button.pack(side=LEFT)

        button = Button(buttonCanvas, text="Orange", bg='orange', command=lambda:self.update_draw_color('Orange'))
        button.pack(side=LEFT, padx=2)

        button = Button(buttonCanvas, text="Yellow", bg='yellow', command=lambda:self.update_draw_color('Yellow'))
        button.pack(side=LEFT, padx=2)

        button = Button(buttonCanvas, text="Green", bg='green', command=lambda:self.update_draw_color('Green'))
        button.pack(side=LEFT, padx=2)

        button = Button(buttonCanvas, text="Blue", bg='blue', command=lambda:self.update_draw_color('Blue'))
        button.pack(side=LEFT, padx=2)

        button = Button(buttonCanvas, text="Purple", bg='purple', command=lambda:self.update_draw_color('Purple'))
        button.pack(side=LEFT, padx=2)

        button = Button(buttonCanvas, text="White", bg='white', command=lambda:self.update_draw_color('White'))
        button.pack(side=LEFT, padx=2)

        button = Button(buttonCanvas, text="Black", bg='black', fg='white',
                        command=lambda: self.update_draw_color('Black'))
        button.pack(side=LEFT, padx=2)

        buttonCanvas.pack(side=TOP, pady=20)

        drawCanvas = Canvas(self.tk)
        label = Label(drawCanvas, text="Draw Size")
        size_up = Button(drawCanvas, text="+", command=lambda: self.update_draw_radius(True))
        size_down = Button(drawCanvas, text="-", command=lambda: self.update_draw_radius(False))
        
        size_up.pack(side=LEFT, padx=2)
        label.pack(side=LEFT, padx=2)
        size_down.pack(side=LEFT, padx=2)
        drawCanvas.pack(side=TOP, pady=20)

    def update_draw_radius(self, increase):
        if increase and self.draw_radius < 20:
            self.draw_radius += 2
        elif not increase and self.draw_radius - 2 > 0:
            self.draw_radius -= 2

    def update_draw_color(self, color):
        self.draw_color = self.color_dict[color]

    def convert_float_to_absolute_pos(self, pos):
        x = int(pos[0] * self.frame_shape[1])
        y = int(pos[1] * self.frame_shape[0])
        return x, y

    def draw(self, landmarks):
        angles = finger_angles(landmarks)
        prob = (angles < 0.15) * 1.0
        n_fingers = np.count_nonzero(prob)

        pos = pointer_position(landmarks)        
        # x = int(pos[0] * self.frame_shape[1])
        # y = int(pos[1] * self.frame_shape[0])
        x, y = self.convert_float_to_absolute_pos(pos)

        # index finger open = draw on whiteboard
        if n_fingers == 1 and prob[0] == 1:
            if self.action_cache.cache_equal("draw"):
                diff = np.array([x, y])-np.array(self.last_point)
                point_dist = np.linalg.norm(diff)
                print(point_dist)
                if point_dist < 20:
                    slope = diff[1]/diff[0]
                    incremental_steps = 100
                    incremental_dist = diff/incremental_steps
                    for i in range(incremental_steps):
                        incremental_point = (np.array(self.last_point) + (incremental_dist * (i + 1))).astype(np.uint16)
                        cv2.circle(self.whiteboard, incremental_point, radius=self.draw_radius, color=self.draw_color, thickness=-1)
                self.overlay = copy.deepcopy(self.whiteboard)
            self.action_cache.add("draw")
        
        # two fingers detected: INDEX + MIDDLE | action: show pointer
        elif n_fingers == 2 and prob[0] == 1.0 and prob[1] == 1.0:
            if self.action_cache.cache_equal("move"):
                self.overlay = copy.deepcopy(self.whiteboard)
                cv2.circle(self.overlay, (x, y), radius=self.draw_radius, color=self.draw_color, thickness=2)
            self.action_cache.add("move")

        # five fingers detected | action:  erase 
        elif n_fingers == 4:
            if self.action_cache.cache_equal("erase"):
                eraser = np.ones(self.frame_shape, dtype=np.uint8) * 255

                pinky_knuckle = get_pinky_joint(landmarks)
                pinky_knuckle = self.convert_float_to_absolute_pos(pinky_knuckle)
                index_knuckle = get_index_joint(landmarks)
                index_knuckle = self.convert_float_to_absolute_pos(index_knuckle)
                wrist = get_wrist(landmarks)
                wrist = self.convert_float_to_absolute_pos(wrist)

                palmx = int((pinky_knuckle[0] + index_knuckle[0] + wrist[0]) / 3)
                palmy = int((pinky_knuckle[1] + index_knuckle[1] + wrist[1]) / 3)

                cv2.circle(eraser, (palmx, palmy), radius=30, color=(0, 0, 0), thickness=-1)
                and_whiteboard = cv2.bitwise_and(self.whiteboard, eraser)
                and_bkgd = cv2.bitwise_and(self.bkgd, cv2.bitwise_not(eraser))
                self.whiteboard = cv2.add(and_whiteboard, and_bkgd)
                self.overlay = copy.deepcopy(self.whiteboard)
                cv2.circle(self.overlay, (palmx, palmy), radius=30, color=self.draw_color, thickness=2)
            self.action_cache.add("erase")
#
        # two fingers detected: INDEX + PINKY | action: clean whiteboard
        elif n_fingers == 2 and prob[0] == 1.0 and prob[3] == 1.0:
            if self.action_cache.cache_equal("clear"):
                self.whiteboard = copy.deepcopy(self.bkgd)
                self.overlay = copy.deepcopy(self.whiteboard)
            self.action_cache.add("clear")
#        
        # three fingers detected: INDEX + MIDDLE + RING | action: save whiteboard
        elif n_fingers == 3 and prob[0] == 1.0 and prob[1] == 1.0 and prob[2] == 1.0:
            if self.action_cache.cache_equal("save"):
                cv2.imwrite('saved/whiteboard.jpg', cv2.cvtColor(self.whiteboard, cv2.COLOR_RGB2BGR))
                print('-- whiteboard.jpg saved! ')
                self.overlay = copy.deepcopy(self.whiteboard)
            self.action_cache.add("save")

        else:
            self.overlay = copy.deepcopy(self.whiteboard)
        
        self.last_point = [x, y]

    def update_gui(self, flip=False):
        with self.mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands = 1) as hands:
            while self.cam.isOpened():
                success, image = self.cam.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    # If loading a video, use 'break' instead of 'continue'.
                    break
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
                            image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                            landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=tuple(reversed(self.draw_color))),
                            connection_drawing_spec=self.mp_drawing.DrawingSpec(color=tuple(reversed(self.draw_color))))

                # cv2.imshow('MediaPipe Hands', image)
                video = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                videotk = ImageTk.PhotoImage(video)
                self.canvas.create_image((25, 0), image=videotk, anchor=NW)

                # cv2.imshow('Whiteboard', self.overlay)
                whiteboard = Image.fromarray(self.overlay.astype(np.uint8))
                wbtk = ImageTk.PhotoImage(whiteboard)
                self.canvas.create_image((825, 0), image=wbtk, anchor = NW)

                if cv2.waitKey(5) & 0xFF == 27: #if need to break
                    break

                self.tk.update_idletasks()
                self.tk.update()
            self.cam.release()
            cv2.destroyAllWindows()


if __name__ == '__main__':
    back = None
    if len(sys.argv) > 1:
        back = sys.argv[1]
    white = Whiteboard(video_capture=2, max_frame_buffer_len=5, background=back)
