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
img = ins_get_image('t1')

faces = app.get(img)

fig, axs = plt.subplots(1,6,figsize=(12,5))
for i, face in enumerate(faces):
    bbox = face['bbox']
    bbox = [int(b) for b in bbox]
    axs[i].imshow(img[bbox[1]:bbox[3],bbox[0]:bbox[2],::-1])
    axs[i].axis('off')
print("face",len(faces))

# img = cv2.imread("test.jpg")
# # DISPLAY
# cv2.imshow('Lena Soderberg', img)