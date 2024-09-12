import numpy as np 
import os 
import glob 
import cv2
import matplotlib.pyplot as plt 

import insightface 
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image 
print('insightface',insightface.__version__ )

app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640,640))
img = ins_get_image('test')

faces = app.get(img)

print("face",faces)

# img = cv2.imread("test.jpg")
# # DISPLAY
# cv2.imshow('Lena Soderberg', img)