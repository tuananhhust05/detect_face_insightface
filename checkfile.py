import cv2

cap = cv2.VideoCapture("/home/ubuntu4080/detect/datacenter/ch02_20240901000000.mp4")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imwrite("tempt.jpg",frame)