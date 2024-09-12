import datetime
import numpy as np
import os
import os.path as osp
import glob
import cv2
import insightface
import matplotlib.plot as plt 
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

app = FaceAnalysis('buffalo_l')
app.prepare(ctx_id = 0, det_size=(640,640))
img = cv2.imread("test.jpg")
fig,axs = plt.subplots(1,6,figsize=(12,5))
faces = app.get(img)
for i,face in enumerate(faces):
    bbox = face['box']
    bbox = [int(b) for b in bbox]
    filename = f"1test.jpg"
    cv2.imwrite('./outputs/%s'%filename,img[bbox[1] : bbox[3], bbox[0]: bbox[2], ::-1])
print("faces",len(faces))
print(faces[0])