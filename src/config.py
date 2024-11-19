import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_FILE = os.path.join(BASE_DIR, "logs", "app.log")
YOLO_MODEL_PATH = os.path.join(BASE_DIR, "../models", "yolov8n.pt")
CAFFE_PROTOTXT_PATH = os.path.join(BASE_DIR, "../models", "deploy.prototxt")
CAFFE_MODEL_PATH = os.path.join(BASE_DIR, "../models", "res10_300x300_ssd_iter_140000.caffemodel")

# General settings
CONFIDENCE_THRESHOLD = 0.5  # Confidence threshold for face detection
IMAGE_WIDTH = 640           # Width for resizing the image

# Model selection
DETECTION_MODEL = "YOLO"  # Options: "YOLO", "CAFFE"
