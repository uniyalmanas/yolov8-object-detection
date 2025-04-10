import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')  # Or yolov8s.pt, etc.

# Function to detect objects in webcam feed
def detect_from_webcam(confidence_threshold=0.4):
    cap = cv2.VideoCapture(0)  # 0 = default camera

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Run YOLO detection with specified confidence threshold
        results = model(frame, conf=confidence_threshold)
        annotated_frame = results[0].plot()

        # Display results on the frame
        cv2.imshow("YOLOv8 Webcam Detection", annotated_frame)

        # Print detection info
        print(f"Detected {len(results[0].boxes)} objects:")
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])
            coords = box.xyxy[0].tolist()
            print(f"- {class_name} ({confidence:.2f}) at {coords}")

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    detect_from_webcam(confidence_threshold=0.4)
