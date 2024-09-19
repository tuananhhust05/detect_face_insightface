
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
import time 
from mutagen.mp4 import MP4
from deblur import apply_blur,wiener_filter,gaussian_kernel
from pinecone import Pinecone, ServerlessSpec
import uuid
import json 
import numba
from numba import jit
pc = Pinecone(api_key="be4036dc-2d41-4621-870d-f9c4e8958412")
index = pc.Index("facejackma")
app = FaceAnalysis('buffalo_l')
app.prepare(ctx_id = 0, det_size=(640,640))


def extract_frames(video_file):
    # resetPincone()
    frame_count = 0
    frame_rate = 60  # default 1s with 30 frame)   
    cap = cv2.VideoCapture(video_file)
    
    while True :
        ret, frame = cap.read()
        frame_count = frame_count + 1 
        print(frame_count)
        if not ret:
            break
        faces = app.get(frame)
        for i,face in enumerate(faces):
            bbox = face['bbox']
            bbox = [int(b) for b in bbox]
            filename=f"{frame_count}_{i}.jpg"
            cv2.imwrite('./imgs/%s'%filename,frame[bbox[1] : bbox[3], bbox[0]: bbox[2], ::-1])
    print("end video")

extract_frames('ali2.mp4')


