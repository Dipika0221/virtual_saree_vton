import cv2
import mediapipe as mp
from ultralytics import YOLO

# 1. Setup Models
yolo_model = YOLO('yolov8n.pt')
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
# Using 'Lite' model (complexity 0) to prevent lag/conflicts
pose = mp_pose.Pose(model_complexity=0, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    # STEP 1: YOLO detects the person
    # verbose=False keeps your terminal clean so you can see errors
    results = yolo_model(frame, classes=[0], verbose=False) 

    for r in results:
        for box in r.boxes:
            # Get person coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Draw YOLO Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # STEP 2: CROP the frame to just the person
            # This is the "Magic Fix" - MediaPipe works better on crops
            person_rgb = frame[y1:y2, x1:x2]
            if person_rgb.size > 0:
                person_rgb = cv2.cvtColor(person_rgb, cv2.COLOR_BGR2RGB)
                pose_results = pose.process(person_rgb)

                # STEP 3: Draw landmarks BACK onto the original frame
                if pose_results.pose_landmarks:
                    mp_draw.draw_landmarks(
                        frame[y1:y2, x1:x2], # Draw directly on the slice of the frame
                        pose_results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        mp_draw.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2),
                        mp_draw.DrawingSpec(color=(255,255,255), thickness=2)
                    )

    cv2.imshow('YOLO + MediaPipe (Cropped Mode)', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()