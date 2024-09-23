import cv2

cap = cv2.VideoCapture("ch02_20240901031703.mp4")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imwrite("tempt.jpg",frame)