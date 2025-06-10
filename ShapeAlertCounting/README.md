# Car Parking Space Detection Project

This project demonstrates a simple car parking space detection system using OpenCV and cvzone. It allows you to mark parking spaces of various shapes (rectangles and polygons) on a static image, and then it detects the occupancy of these spaces in a video feed from either a local file or a live RTSP camera stream. 



## Overview

The main goal of this project is to detect and monitor car parking spaces. It consists of two main parts:

1. **ParkingSpacePickerMultiShape.py**: This Python script allows you to manually select parking spaces on a static image (`carParkImg.png`). You can define:
    - **Rectangular spaces**: Click to place, then click and drag corners/edges to resize. Toggle between 'horizontal' (H key) and 'vertical' (V key) default orientations.
    - **Polygonal spaces**: Click to place points, then click near the starting point to close the polygon.
    Right-click to remove an existing space or cancel polygon creation.
    The coordinates and shape information are saved in a file named `CarParkPos` using pickle. Press 'R' for rectangle mode and 'P' for polygon mode.

2. **mainVideo.py**: This Python script reads the saved parking space coordinates. It can process:
    - A local video feed (e.g., `carPark.mp4`).
    - A live RTSP camera stream.
    The script will prompt you to choose the source at startup. It then detects the occupancy of parking spaces, displays the video with marked spaces, and updates the count of free spaces in real-time.

## Installation

2. Install the required dependencies, including OpenCV and cvzone:

   ```bash
   pip install opencv-python-headless
   pip install numpy
   pip install cvzone
   ```

3. Ensure you have the following files in your project folder:

   - `carPark.mp4` (sample video file, if using video file input)
   - `carParkImg.png` (static image of the parking lot for the picker)
   - `ParkingSpacePickerMultiShape.py`
   - `mainVideo.py`

## Usage

1. Run `ParkingSpacePickerMultiShape.py` to mark the parking spaces on the static image (`carParkImg.png`).
   - Use 'R' key to switch to Rectangle mode, 'P' key for Polygon mode.
   - For Rectangles: Use 'H' for horizontal, 'V' for vertical default. Left-click to place, then click and drag handles to resize.
   - For Polygons: Left-click to place points. Click near the first point to close.
   - Right-click to remove a space or cancel current polygon drawing.
   - Press 'Q' to quit and save. The coordinates will be saved in `CarParkPos`.

2. Run `mainVideo.py`.
   - The script will first prompt you to choose the video source:
     - **Option 1**: Video File (e.g., `carPark.mp4`).
     - **Option 2**: RTSP Camera Stream. If selected, you will be prompted to choose a pre-configured camera.
       (Note: RTSP camera IPs and credentials need to be configured within `mainVideo.py` in the `CAMERA_MAP` dictionary).
   - The script will then start processing the selected feed and detecting parking space occupancy. The video will display with marked parking spaces and a count of free spaces. Press 'Q' to quit.

## Files and Folders

- `carPark.mp4`: Sample input video file containing parking lot footage.
- `carParkImg.png`: Static image of the parking lot used by `ParkingSpacePickerMultiShape.py`.
- `ParkingSpacePickerMultiShape.py`: Script to select and save parking space coordinates (supports rectangles and polygons).
- `mainVideo.py`: Script for processing the video/RTSP stream and detecting parking space occupancy.
- `CarParkPos`: Binary file containing saved parking space coordinates and shape information (created by `ParkingSpacePickerMultiShape.py`).
