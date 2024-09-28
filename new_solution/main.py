from Detector import * 
import time 
detector = Detector()

start = time.time()
check = detector.processVideo("/home/poc4a5000/facesx/ch02_20240904040117.mp4")
end = time.time()
print("ecution", end - start)
print(detector.result)