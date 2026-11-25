
import cv2
import mss
import numpy as np
from ultralytics import YOLO

# Load your trained model
model = YOLO(r"runs\detect\runs\detect\target_detection-8\weights\best.pt")

# Screen capture
sct = mss.MSS()
monitor = sct.monitors[1]

while True:
    # Capture screen
    screenshot = np.array(sct.grab(monitor))

    # Convert BGRA to BGR
    frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)

    # Detect circular targets
    results = model.predict(
        frame,
        conf=0.25,
        imgsz=640,
        verbose=False
    )

    # Draw bounding boxes
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            confidence = float(box.conf[0])

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Target {confidence:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    cv2.imshow("Live Target Detection", frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

sct.close()
cv2.destroyAllWindows()
