# webcam_detection.py

import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')  # Or yolov8s.pt, etc.

def detect_from_webcam():
    cap = cv2.VideoCapture(0)  # 0 = default camera

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = model(frame)
        annotated_frame = results[0].plot()

        cv2.imshow("YOLOv8 Webcam Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    detect_from_webcam()
