from Detector import * 

detector = Detector()

check = detector.processImage("jackma.png")
print(check)
print(detector.result)