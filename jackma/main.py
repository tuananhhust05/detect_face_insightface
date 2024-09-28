from Detector import * 
import cv2
detector = Detector()
img = cv2.imread("10740_0_output.jpg")
check = detector.processImage(img)
print(detector.result)