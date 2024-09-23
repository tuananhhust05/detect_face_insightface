import cv2

frame_count = 0 
cap = cv2.VideoCapture("/home/ubuntu4080/detect/datacenter/ch02_20240901000000.mp4")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count = frame_count + 1
    print(frame_count)
    cv2.imwrite("tempt.jpg",frame)