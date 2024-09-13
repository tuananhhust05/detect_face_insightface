import datetime
import numpy as np
import os
import os.path as osp
import glob
import cv2
import insightface
import matplotlib.pyplot as plt 
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
from numpy.linalg import norm

def cosin(question,answer):
    cosine = np.dot(question,answer)/(norm(question)*norm(answer))
    return cosine

array_cosin = []
array_em = []
app = FaceAnalysis('buffalo_l')
app.prepare(ctx_id = 0, det_size=(640,640))
img = cv2.imread("./videotest_frames/frame_135.jpg") 
fig,axs = plt.subplots(1,6,figsize=(12,5))
faces = app.get(img)

for i,face in enumerate(faces):
    bbox = face['bbox']
    bbox = [int(b) for b in bbox]
    filename = f"{i}test.jpg"
    cv2.imwrite('./outputs/%s'%filename,img[bbox[1] : bbox[3], bbox[0]: bbox[2], ::-1])
    array_em.append(face['embedding'])

img2 = cv2.imread("./videotest_frames/frame_150.jpg")
faces2 = app.get(img2)
for i,face in enumerate(faces):
    bbox = face['bbox']
    bbox = [int(b) for b in bbox]
    filename = f"{i}test.jpg"
    cv2.imwrite('./outputs/%s'%filename,img[bbox[1] : bbox[3], bbox[0]: bbox[2], ::-1])
    for em in array_em:
      cosin_value = cosin(em,face['embedding'])
      array_cosin.append(cosin_value)
    #   array_em.append(face['embedding'])

print("array_em",array_em)
print("array_cosin",array_cosin)
print("first",faces[0])


