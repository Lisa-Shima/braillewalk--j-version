from ultralytics import YOLO
import cv2

# Load the YOLOv8 pre-trained model
model = YOLO("yolov8n.pt")  # You can replace with yolov8m.pt or yolov8l.pt for better accuracy.

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam!")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame!")
        break

    # Run YOLOv8 inference on the frame
    results = model(frame)

    # Visualize the results
    frame_with_boxes = results[0].plot()  # This will draw boxes on the frame

    # Display the resulting frame
    cv2.imshow("YOLOv8 Detection", frame_with_boxes)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture object and close windows
cap.release()
cv2.destroyAllWindows()
