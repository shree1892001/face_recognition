import cv2
import numpy as np
import imutils
import os

# Function to capture a photo using the webcam
def take_photo(filename='photo.jpg'):
    print("[INFO] Starting webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Unable to access the webcam.")
        return None

    print("[INFO] Press 'SPACE' to capture a photo or 'ESC' to exit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to capture frame.")
            break

        cv2.imshow("Webcam", frame)
        key = cv2.waitKey(1) & 0xFF

        # Capture the photo when SPACE is pressed
        if key == 32:  # SPACE key
            cv2.imwrite(filename, frame)
            print(f"[INFO] Photo saved as {filename}")
            break
        elif key == 27:  # ESC key
            print("[INFO] Exiting without capturing a photo.")
            frame = None
            break

    cap.release()
    cv2.destroyAllWindows()
    return filename if frame is not None else None

# Function to download the face detection model files
def download_model_files():
    prototxt_file = "D:\\facerecogniton\\models\\deploy.prototxt"
    model_file = "D:\\facerecogniton\\models\\res10_300x300_ssd_iter_140000.caffemodel"

    # if not os.path.exists(prototxt_file):
    #     print("[INFO] Downloading prototxt file...")
    #     os.system(f"wget -O {prototxt_file} https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt")
    # if not os.path.exists(model_file):
    #     print("[INFO] Downloading caffemodel file...")
    #     os.system(f"wget -O {model_file} https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel")

    return prototxt_file, model_file

# Function to load the face detection model
def load_face_detector(prototxt_file, model_file):
    print("[INFO] Loading face detection model...")
    net = cv2.dnn.readNetFromCaffe(prototxt_file, model_file)
    return net

# Function to perform face detection on an image
def detect_faces(image, net, confidence_threshold=0.5):
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    # Process the detections
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_threshold:
            # Compute the bounding box coordinates
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Draw the bounding box and confidence score
            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
            cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    return image

# Main function to capture, detect faces, and display the results
def main():
    print("[INFO] Checking and downloading model files if necessary...")
    prototxt_file, model_file = download_model_files()

    print("[INFO] Taking a photo...")
    image_file = take_photo()
    if image_file is None:
        print("[ERROR] No photo captured.")
        return

    image = cv2.imread(image_file)
    if image is None:
        print("[ERROR] Could not read the image.")
        return

    net = load_face_detector(prototxt_file, model_file)

    print("[INFO] Detecting faces...")
    image = imutils.resize(image, width=400)
    result_image = detect_faces(image, net)

    print("[INFO] Displaying results. Press any key to close.")
    cv2.imshow("Face Detection", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
