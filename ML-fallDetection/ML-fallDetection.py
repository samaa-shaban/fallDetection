import cv2
import mediapipe as mp
import numpy as np
import time
from playsound import playsound
import threading  # لتشغيل الصوت بدون ما يوقف الكاميرا

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)

alarm_played = False  # علشان منعيدش تشغيل الإنذار كل مرة

def play_alarm():
    playsound("alarm.mp3")  # لازم الملف يكون في نفس المجلد

with mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6) as pose:
    fall_detected = False
    fall_start_time = None
    fall_duration_threshold = 2  # ثواني لتأكيد السقوط

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            shoulder_y = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y +
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y) / 2
            hip_y = (landmarks[mp_pose.PoseLandmark.LEFT_HIP].y +
                     landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y) / 2
            head_y = landmarks[mp_pose.PoseLandmark.NOSE].y

            shoulder_hip_diff = abs(shoulder_y - hip_y)
            head_hip_diff = abs(head_y - hip_y)

            if head_y > hip_y and shoulder_hip_diff < 0.15:
                if not fall_detected:
                    fall_detected = True
                    fall_start_time = time.time()
                    alarm_played = False  # نسمح بتشغيل الإنذار من جديد
                    cv2.putText(image, 'Possible Fall Detected...', (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                else:
                    elapsed = time.time() - fall_start_time
                    if elapsed >= fall_duration_threshold:
                        cv2.putText(image, 'Confirmed Fall!', (50, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                        if not alarm_played:
                            threading.Thread(target=play_alarm).start()
                            alarm_played = True
            else:
                fall_detected = False
                fall_start_time = None
                alarm_played = False

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow('Fall Detection', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
