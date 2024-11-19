import cv2

def take_photo(output_path='photo.jpg'):
    """
    Capture a photo using the webcam.
    :param output_path: File path to save the captured image.
    :return: Path to the saved photo, or None if no photo was captured.
    """
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

        if key == 32:  # SPACE key
            cv2.imwrite(output_path, frame)
            print(f"[INFO] Photo saved at {output_path}")
            break
        elif key == 27:  # ESC key
            print("[INFO] Exiting without capturing a photo.")
            frame = None
            break

    cap.release()
    cv2.destroyAllWindows()
    return output_path if frame is not None else None
