import time
import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp

# Load YOLO model
model = YOLO("yolov8n.pt")

# Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

# Gender detection model (OpenCV DNN)
gender_net = cv2.dnn.readNetFromCaffe(
    "gender_deploy.prototxt",
    "gender_net.caffemodel"
)

GENDER_LIST = ['Male', 'Female']

# Webcam
cap = cv2.VideoCapture(0)

last_capture_time = 0
capture_interval = 5  # seconds

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO detection
    results = model(frame)

    for r in results:
        boxes = r.boxes

        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            # Class 0 = person
            if cls == 0 and conf > 0.5:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                person_crop = frame[y1:y2, x1:x2]

                if person_crop.size == 0:
                    continue

                # Try face region (upper part)
                face = person_crop[0:int((y2-y1)/3), :]

                if face.size == 0:
                    continue

                blob = cv2.dnn.blobFromImage(
                    face, 1.0, (227, 227),
                    (78.4263377603, 87.7689143744, 114.895847746),
                    swapRB=False
                )

                gender_net.setInput(blob)
                gender_preds = gender_net.forward()
                gender = GENDER_LIST[gender_preds[0].argmax()]

                # 👉 ONLY FEMALE
                if gender == "Female":
                    current_time = time.time()

                if current_time - last_capture_time > capture_interval:
                  filename = f"snapshot_{int(current_time)}.jpg"
                  cv2.imwrite(filename, frame)
                  print(f"Snapshot saved: {filename}")
                  last_capture_time = current_time
                    
                    
                    

                    # BLUE BOX (BGR: 255,0,0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                cv2.putText(frame, f"Female",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (255, 0, 0), 2)

                    # Skeleton on full frame
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result_pose = pose.process(rgb)

                if result_pose.pose_landmarks:
                        mp_draw.draw_landmarks(
                            frame,
                            result_pose.pose_landmarks,
                            mp_pose.POSE_CONNECTIONS
                        )

    cv2.imshow("Female Detection + Skeleton", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()