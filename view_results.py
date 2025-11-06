#!/usr/bin/env python3
"""Quick script to view already-processed results."""
import sys
import cv2
sys.path.insert(0, 'src/python')
from viewer import InteractiveViewer

video_path = 'output/analysis_225bench.mp4'
print(f"Loading {video_path}...")

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frames = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)

cap.release()

print(f"Loaded {len(frames)} frames at {fps} fps")

viewer = InteractiveViewer("Bench Press Analysis Results")
viewer.load_frames(frames, fps)
viewer.run()
