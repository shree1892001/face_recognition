import cv2
from ultralytics import YOLO
from src.config import CONFIDENCE_THRESHOLD
from src.logger import logger


class YOLOv8FaceDetector:
    def __init__(self, model_path):
        """
        Initialize the YOLOv8 face detector.
        :param model_path: Path to the YOLOv8 model file.
        """
        logger.info("Loading YOLOv8 model...")
        self.model = YOLO(model_path)

    def detect_faces(self, image):
        """
        Detect faces in the given image using YOLOv8.
        :param image: Input image as a NumPy array.
        :return: Image with bounding boxes drawn around detected faces.
        """
        logger.info("Running face detection with YOLOv8...")
        results = self.model(image)
        detections = results[0].boxes.data.cpu().numpy()
        logger.info(f"Detected {len(detections)} faces.")

        for box in detections:
            x1, y1, x2, y2, conf, cls = box
            if conf > CONFIDENCE_THRESHOLD:
                logger.info(f"Face detected with confidence {conf:.2f}: Bounding box {x1, y1, x2, y2}.")
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                text = f"{conf * 100:.2f}%"
                y = int(y1) - 10 if int(y1) - 10 > 10 else int(y1) + 10
                cv2.putText(image, text, (int(x1), y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

        return image
