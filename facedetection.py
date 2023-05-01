import cv2
import numpy as np
import time

classifier  = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

livecam = cv2.VideoCapture(0)

if not livecam.isOpened():
    print("Camera is not opened")
    exit()

Status = False
while (Status == False):
    ret, frame = livecam.read()

    if ret == True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        classified = classifier.detectMultiScale(gray, scaleFactor = 1.3, minNeighbors = 2)

        for (x, y, w, h) in classified:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        text = "Number of face = " + str(len(frame))

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, text, (0, 30), font, 1, (255, 0, 0), 1)

        cv2.imshow("Face Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            Status = True
            break


livecam.release()
cv2.destroyAllWindows()