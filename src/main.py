import cv2
from handlers.CameraHandler  import take_photo
from handlers.filehandler import ensure_directory_exists
from services.Yolov8_Detector import YOLOv8FaceDetector
from services.Caffe_Detector import CaffeFaceDetector
from src.config import (
    YOLO_MODEL_PATH,
    CAFFE_PROTOTXT_PATH,
    CAFFE_MODEL_PATH,
    DETECTION_MODEL,
    IMAGE_WIDTH,
)
from src.logger import logger


def main():
    """
    Main function to capture an image, detect faces, and display results.
    """
    # Ensure necessary directories exist
    ensure_directory_exists("logs")
    ensure_directory_exists("models")

    # Select face detection model
    if DETECTION_MODEL == "YOLO":
        logger.info("Using YOLOv8 for face detection.")
        face_detector = YOLOv8FaceDetector(YOLO_MODEL_PATH)
    elif DETECTION_MODEL == "CAFFE":
        logger.info("Using Caffe for face detection.")
        face_detector = CaffeFaceDetector(CAFFE_PROTOTXT_PATH, CAFFE_MODEL_PATH)
    else:
        logger.error("Invalid detection model specified. Exiting...")
        return

    # Capture photo
    logger.info("Capturing photo...")
    photo_path = take_photo("captured_photo.jpg")
    if not photo_path:
        logger.error("No photo captured. Exiting...")
        return

    # Load the captured image
    image = cv2.imread(photo_path)
    if image is None:
        logger.error("Could not read the captured photo. Exiting...")
        return

    # Resize image for consistent detection
    image = cv2.resize(image, (IMAGE_WIDTH, int(image.shape[0] * IMAGE_WIDTH / image.shape[1])))

    # Perform face detection
    result_image = face_detector.detect_faces(image)

    # Display results
    logger.info("Displaying results. Press any key to close.")
    cv2.imshow("Face Detection Results", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
