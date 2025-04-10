import cv2
from ultralytics import YOLO

# Load the larger YOLOv8 model for better accuracy
model = YOLO('yolov8s.pt')  # Try yolov8s.pt or larger models like yolov8m.pt

# Function to detect objects
def detect_objects(image_path):
    img = cv2.imread(image_path)

    # Run object detection with a lower confidence threshold
    results = model(img, conf=0.4)  # You can adjust the confidence value (e.g., 0.4)

    # Annotate and show result
    for result in results:
        annotated = result.plot()
        cv2.imshow("YOLOv8 Detection", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        print(f"Detected {len(result.boxes)} objects:")
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])
            coords = box.xyxy[0].tolist()
            print(f"- {class_name} ({confidence:.2f}) at {coords}")

# Replace this with your image path
detect_objects("image1.jpg")

def detect_from_webcam():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.4)  # Adjust the confidence threshold here as well
        annotated_frame = results[0].plot()
        cv2.imshow("Webcam Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# Call webcam detection function
detect_from_webcam()
