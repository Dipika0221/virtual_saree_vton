Female Detection and Skeleton Tracking (VTON Preprocessing)
This module is part of a Virtual Try-On (VTON) project. It focuses on detecting female users in real time using a webcam, extracting body keypoints, and capturing snapshots that will later be used for saree overlay.
Overview
The system combines object detection, gender classification, and pose estimation to identify female subjects and track their body structure. This forms the base for aligning clothing in the next stage of the project.
Features
Detects full human body using YOLOv8
Filters only female subjects using a gender classification model
Draws a bounding box around detected females
Extracts body keypoints using MediaPipe Pose
Captures snapshots automatically at fixed intervals
Tech Stack
Python
OpenCV
Ultralytics YOLOv8
MediaPipe
NumPy
Installation
Install the required libraries:
pip install ultralytics opencv-python mediapipe numpy

Required Files
Ensure the following files are available:
yolov8n.pt (downloaded automatically)
gender_deploy.prototxt
Gender_net.caffemodel

How to Run
python saree_tryon.py
Press ESC to close the application.
Working
The webcam captures live video input
YOLO detects persons in the frame
The upper body region is used for gender prediction
Only female detections are considered
A bounding box is drawn around the detected subject
MediaPipe extracts full-body keypoints
Snapshots are saved periodically
Output
Captured images are saved in the following format:
snapshot_<timestamp>.jpg

Limitations
Gender detection depends on visible facial features
Performance may vary under poor lighting
Accuracy may drop for side or back-facing subjects

Next Steps
The next stage is to use the detected keypoints to align and overlay a saree on the user for the virtual try-on system.
