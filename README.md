CCTV Analytics Dashboard DEMO
Overview
This system provides real-time object detection, tracking, and analytics for CCTV camera feeds or video files.
Architecture Flow
utils.py & sort/sort.py
         ↓
    main.py (Core Processing)
         ↓
    display.py (Streamlit Dashboard)
         ↓
    run_dashboard.py (Launcher)
Components
1. Core Modules

utils.py: Utility functions for license plate reading, formatting, and CSV writing
sort/sort.py: SORT (Simple Online and Realtime Tracking) algorithm implementation
main.py: Core processing pipeline that:

Connects to RTSP cameras or loads video files
Performs object detection using YOLOv8
Tracks objects across frames
Detects and reads license plates (if model available)
Counts objects crossing IN/OUT lines



2. User Interface

display.py: Streamlit dashboard that:

Provides user interface for camera/video selection
Calls main.py functions for processing
Displays live video feed with annotations
Shows real-time analytics and counts
Allows drawing IN/OUT counting lines
Exports results and reports



3. Additional Components

line_drawer.py: Helper functions for interactive line drawing
run_dashboard.py: Launcher script that:

Checks and installs dependencies
Verifies required files
Starts the Streamlit dashboard



Installation

Ensure you have Python 3.8+ installed
Run the launcher script:
bashpython run_dashboard.py
This will automatically install required packages and start the dashboard.

Usage
Camera Mode

Select "Camera" as source type
Enter camera credentials (default: admin/d@t@rium2023)
Click "Connect to Camera"
Draw IN/OUT lines if needed
Click "Start Processing"

Video File Mode

Select "Video File" as source type
Upload a video file (MP4, AVI, MOV, MKV)
Draw IN/OUT lines if needed
Click "Start Processing"

Features

Real-time object detection and tracking
IN/OUT counting with custom lines
License plate detection (requires model at ./models/best.pt)
Live analytics and visualization
Export results to CSV
Generate reports

File Structure
project/
├── main.py              # Core processing logic
├── display.py           # Streamlit dashboard
├── utils.py            # Utility functions
├── run_dashboard.py    # Launcher script
├── line_drawer.py      # Line drawing utilities
├── sort/
│   └── sort.py        # SORT tracking algorithm
└── models/
    └── best.pt        # License plate model (optional)
Camera Configuration
Default camera IPs:

Camera 1: 10.20.1.7
Camera 2: 10.20.1.8
Camera 3: 10.20.1.18
Camera 4: 10.20.1.19

Model Requirements

YOLOv8n: Downloaded automatically for object detection
License Plate Model: Place your trained model at ./models/best.pt for license plate detection

Notes

The system uses YOLOv8 for object detection
SORT algorithm is used for object tracking
EasyOCR is used for license plate text recognition
Results are saved with timestamps for traceability