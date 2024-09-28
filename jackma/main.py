from Detector import * 

detector = Detector()

check = detector.processImage("jackma.png")
print(detector.result)