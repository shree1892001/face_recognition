import cv2
import numpy as np
from src.config import CONFIDENCE_THRESHOLD
from src.logger import logger


class CaffeFaceDetector:
    def __init__(self, prototxt_path, model_path):
        """
        Initialize the Caffe-based face detector.
        :param prototxt_path: Path to the prototxt file.
        :param model_path: Path to the Caffe model file.
        """
        logger.info("Loading Caffe model...")
        self.net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

    def detect_faces(self, image):
        """
        Detect faces in the given image using Caffe.
        :param image: Input image as a NumPy array.
        :return: Image with bounding boxes drawn around detected faces.
        """
        logger.info("Running face detection with Caffe...")
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()
        logger.info(f"Detected {detections.shape[2]} potential faces.")

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > CONFIDENCE_THRESHOLD:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                logger.info(f"Face detected with confidence {confidence:.2f}: Bounding box {startX, startY, endX, endY}.")
                cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
                text = f"{confidence * 100:.2f}%"
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

        return image
