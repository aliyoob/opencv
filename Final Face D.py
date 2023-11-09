import cv2

# ساخت یک شیء CascadeClassifier برای تشخیص چهره
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# شروع دوربین ویدیو
cap = cv2.VideoCapture(0)  

while True:
    ret, frame = cap.read()

   
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        face_image = frame[y:y + h, x:x + w]
        cv2.imwrite("detected_face.jpg", face_image)

    cv2.imshow('Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# بستن دوربین و پنجره
cap.release()
cv2.destroyAllWindows()
