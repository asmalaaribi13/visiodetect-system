import cv2

def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
    coords = []
    for (x, y , w , h) in features:
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        
        # Calculate text size and position
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 1)[0]
        text_x = x
        text_y = y - 5 if y >= text_size[1] + 5 else y + h + text_size[1] + 5

        cv2.putText(img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        coords = [x, y, w, h]

    return coords




def detect(img, faceCascade, eyesCascade, noseCascade, mouthCascade):
    color = {"blue": (255,0,0), "red": (0,0,255), "green": (0,255,0), "white": (255,255,255)}
    coords = draw_boundary(img, faceCascade, 1.1, 10, color['blue'], "Face")
    
    if len(coords) == 4:
        x, y, w, h = coords
        roi_img = img[y:y + h, x:x + w]
        
        # Adjust the ROI for eyes
        eyes_roi = roi_img[h // 5 : h // 2, :]
        coords = draw_boundary(eyes_roi, eyesCascade, 1.1, 5, color['red'], "Eye")
        
        # Adjust the ROI for nose
        nose_roi = roi_img[h // 2 : h * 3 // 4, :]
        coords = draw_boundary(nose_roi, noseCascade, 1.1, 5, color['green'], "Nose")
        
        # Adjust the ROI for mouth
        mouth_roi = roi_img[h * 3 // 4 :, :]
        coords = draw_boundary(mouth_roi, mouthCascade, 1.1, 10, color['white'], "Mouth")
        
    return img



faceCascacde = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eyesCascade = cv2.CascadeClassifier("haarcascade_eye.xml")
noseCascade = cv2.CascadeClassifier("Nariz.xml")
mouthCascade = cv2.CascadeClassifier("Mouth.xml")


video_capture = cv2.VideoCapture(0)

while True:
    _,img = video_capture.read()
    img = detect(img, faceCascacde, eyesCascade,noseCascade, mouthCascade)
    cv2.imshow("Face Detection", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
       break

video_capture.release()
cv2.destroyAllWindows() 
