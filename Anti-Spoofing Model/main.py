import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist

# Load the Haar Cascade Classifier for face detection
face_cascade = cv2.CascadeClassifier('C:/Users/Asma Laaribi/Anti-Spoofing Model/haarcascade_frontalface_default.xml')

# Load the facial landmarks predictor
predictor_path = "shape_predictor_68_face_landmarks.dat"  
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def detect_blinks(eye_landmarks):
    left_eye = eye_landmarks[36:42]
    right_eye = eye_landmarks[42:48]
    left_ear = eye_aspect_ratio(left_eye)
    right_ear = eye_aspect_ratio(right_eye)
    return left_ear, right_ear

blink_threshold = 0.25  # You can adjust this threshold

cap = cv2.VideoCapture(0)  # For Webcam
cap.set(3, 640)
cap.set(4, 480)

while True:
    success, img = cap.read()
    
    # Convert the frame to grayscale for Haar Cascade Classifier
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces using Haar Cascade Classifier
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        # Crop the detected face region
        face_roi = img[y:y+h, x:x+w]
        
        # Detect facial landmarks
        dlib_rect = dlib.rectangle(x, y, x + w, y + h)
        shape = predictor(gray, dlib_rect)
        shape = np.array([[p.x, p.y] for p in shape.parts()])
        
        left_ear, right_ear = detect_blinks(shape)
        
        # Determine if the face is real based on blink detection
        is_real_face = left_ear > blink_threshold and right_ear > blink_threshold
        
        # Draw a rectangle around the detected face and label it as real or fake
        color = (0, 255, 0) if is_real_face else (0, 0, 255)
        label = "Real" if is_real_face else "Fake"
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# import math
# import time

# import cv2
# import cvzone
# from ultralytics import YOLO

# confidence = 0.6

# cap = cv2.VideoCapture(0)  # For Webcam
# cap.set(3, 640)
# cap.set(4, 480)
# # cap = cv2.VideoCapture("../Videos/motorbikes.mp4")  # For Video


# model = YOLO("../models/l_version_1_300.pt")

# classNames = ["fake", "real"]

# prev_frame_time = 0
# new_frame_time = 0

# while True:
#     new_frame_time = time.time()
#     success, img = cap.read()
#     results = model(img, stream=True, verbose=False)
#     for r in results:
#         boxes = r.boxes
#         for box in boxes:
#             # Bounding Box
#             x1, y1, x2, y2 = box.xyxy[0]
#             x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#             # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
#             w, h = x2 - x1, y2 - y1
#             # Confidence
#             conf = math.ceil((box.conf[0] * 100)) / 100
#             # Class Name
#             cls = int(box.cls[0])
#             if conf > confidence:

#                 if classNames[cls] == 'real':
#                     color = (0, 255, 0)
#                 else:
#                     color = (0, 0, 255)

#                 cvzone.cornerRect(img, (x1, y1, w, h),colorC=color,colorR=color)
#                 cvzone.putTextRect(img, f'{classNames[cls].upper()} {int(conf*100)}%',
#                                    (max(0, x1), max(35, y1)), scale=2, thickness=4,colorR=color,
#                                    colorB=color)


#     fps = 1 / (new_frame_time - prev_frame_time)
#     prev_frame_time = new_frame_time
#     print(fps)

#     cv2.imshow("Image", img)
#     cv2.waitKey(1)