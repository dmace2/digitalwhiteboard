import tkinter as tk
from mediapipe_whiteboard import Whiteboard

class TKInterView:
    def __init__(self):
        self.root = tk.Tk()
        self.view = tk.Canvas(self.root, width=1000, height=1000)
        self.view.pack(side="top", fill="both", expand=True)
        self.root.mainloop()


class WhiteboardView:
    def __init__(self):
        self.whiteboard = Whiteboard(max_frame_buffer_len=5)






if __name__ == "__main__":
    view = TKInterView()
    