from Detector import * 
import cv2
detector = Detector()
img = cv2.imread("jackma.png")
check = detector.processImage(img)
print(detector.result)