import numpy as np 
import cv2 
from imutils.video import FPS 

class Detector:
    def __init__(self):
        self.faceModel = cv2.dnn.readNetFromCaffe("res10_300x300_ssd_iter_140000.prototxt",
            caffeModel="res10_300x300_ssd_iter_140000.caffemodel")
        self.result = False

    def processImage(self, imgName):
        self.img = cv2.imread(imgName)
        (self.height, self.width) = self.img.shape[:2]
        self.processFrame()
        

        
    def processFrame(self):
        self.result = False
        blob = cv2.dnn.blobFromImage(self.img, 1.0, (300,300), (104.0, 177.0, 123.0), swapRB = False, crop = False)

        self.faceModel.setInput(blob)
        predictions = self.faceModel.forward()
        for i in range(0, predictions.shape[2]):
            if predictions[0, 0, i, 2] > 0.5 :
                self.result =  True
                return
          
            