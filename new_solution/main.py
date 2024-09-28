from Detector import * 

detector = Detector()

checkface = detector.processImage("tempt.jpg")

print(checkface)