# Digital AI Whiteboard
Implemented using Google's MediaPipe hand detection, we have created a digital whiteboard! Just draw, erase, save, and clear using hand gestures. And for larger-scale changes, such as background and color change, use our command args and simple UI.

## Gesture Support
- index finger: draw
- index and middle fingers: move cursor
- index, middle, and ring fingers: save whiteboard to jpeg
- full hand: erase (uses palm center as erase point)
- index and ring fingers: clear whiteboard

## Command Line Args

#### Custom Backgrounds
This whiteboard was designed to allow for any type of background image. Just pass in your image filepath as the first command line argument when running the script.

#### Default Colors
If you just want a simpler background, pass in one of the default colors (Red, Orange, Yellow, Green, Blue, Purple, White, Black) as the first argument and your colorful background will be created.

## Requirements
Build your environment from the the included [env.yml](https://github.com/dmace2/digitalwhiteboard/blob/tk_branch/env.yml) Anaconda file. If you cannot install the environment, just work one by one through the pip requirements list until the file runs.

**NOTE:** If you are running this on an nvidia jetson device, you cannot use the YAML file. You must be running Ubuntu 20.04 and using Python 3.8. Additionally, you need to install MediaPipe using the most recent prebuilt wheel located [here](https://github.com/jiuqiant/mediapipe_python_aarch64).
