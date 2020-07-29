import cv2 as cv

capture = cv.VideoCapture(0)
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    ret, frame = capture.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv.circle(frame, (x + int(w * 0.5), y + int(h * 0.5)), 4, (0, 255, 0), -1)
    cv.imshow('Frame', frame)
    if cv.waitKey(1) & 0xFF == 27:
        break
capture.release()
cv.destroyAllWindows()