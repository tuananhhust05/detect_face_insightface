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
        return
        

        
    def processFrame(self):
        self.result = False
        
        # Convert the image to a CUDA GpuMat
        gpu_img = cv2.cuda_GpuMat()
        gpu_img.upload(self.img)

        # Prepare the blob from the GPU image
        blob = cv2.cuda.dnn.blobFromImage(gpu_img, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)

        # Set the input blob for the model
        self.faceModel.setInput(blob)
        predictions = self.faceModel.forward()
        
        for i in range(predictions.shape[2]):
            if predictions[0, 0, i, 2] > 0.5:
                bbox = predictions[0, 0, i, 3:7] * np.array([self.width, self.height, self.width, self.height])
                (xmin, ymin, xmax, ymax) = bbox.astype("int")
                # Here you can draw the bounding box or perform other actions
                cv2.rectangle(self.img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                self.result = True
            
    def processVideo(self, videoName):
        count = 0 
        cap = cv2.VideoCapture(videoName)
        if(cap.isOpened() == False):
            print("error")
        
        (sucess, self.img) = cap.read()
        (self.height, self.width)  = self.img.shape[:2]

        fps = FPS().start()
        while sucess:
            if(count % 60 == 0):
                self.processFrame()
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
              break
            (sucess,self.img) = cap.read()
            count = count + 1 
            print(count)

        fps.stop()

        cap.release()
          
            