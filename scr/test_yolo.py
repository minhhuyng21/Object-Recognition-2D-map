import cv2
import numpy as np
import torch
from ultralytics import YOLO
import math

video_path = "D:\\AI\\Object-Recognition-2D-map\\asset\\Leeds.mp4"

# model = YOLO('yolov5n.pt')  # load an official model
model = YOLO('yolo11n.pt')  # load an official model
results = model.track(source=video_path, show=True, tracker="bytetrack.yaml")  # with ByteTrack
