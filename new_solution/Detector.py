import numpy as np 
import cv2 
from imutils.video import FPS 

class Detector:
    def __init__(self):
        self.faceModel = cv2.dnn.readNetFromCaffe("res10_300x300_ssd_iter_140000.prototxt",
            caffeModel="res10_300x300_ssd_iter_140000.caffemodel")
    
    def processImage(self, imgName):
        self.img = cv2.imread(imgName)
        (self.height, self.width) = self.img.shape[:2]

        self.processFrame()

        
    def processFrame(self):
        blob = cv2.dnn.blobFromImage(self.img, 1.0, (300,300), (104.0, 177.0, 123.0), swapRB = False, crop = False)

        self.faceModel.setInput(blob)
        predictions = self.faceModel.forward()
        print("predictions", predictions)
        for i in range(0, predictions.shape[2]):
            if predictions[0, 0, i, 2] > 0.5 :
                bbox = predictions[0,0,i,3:7]   * np.array([self.width, self.height,self.width, self.height] )
                (xmin, ymin, xmax, ymax) = bbox.astype("int")
                print(xmin, ymin, xmax, ymax)
                cv2.rectangle(self.img, (xmin, ymin), (xmax,ymax), (0,0,255), 2 )
                cv2.imwrite("face.png", self.img[bbox[1]:bbox[3], bbox[0]:bbox[2], ::-1])

                # top_left = (bbox[0], bbox[1])
                # bottom_right = (bbox[2], bbox[3])
                # color = (255, 0, 0)
                # thickness = 2
                # cv2.rectangle(frame, top_left, bottom_right, color, thickness)
                # cv2.rectangle(self.img, (xmin, ymin), (xmax,ymax), (0,0,255), 2 )
                # cv2.imwrite("face.png", self.img[xmax:x, bbox[0]:bbox[2]])
                print("done rectangle")