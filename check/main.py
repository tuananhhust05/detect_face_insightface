import datetime
import numpy as np
import os
import cv2
import insightface
import torch
from mutagen.mp4 import MP4
import json
import time
from numpy.linalg import norm
from insightface.app import FaceAnalysis
from insightface.model_zoo import model_zoo
from pinecone import Pinecone
from numba import jit, cuda
import subprocess
import threading
import matplotlib.pyplot as plt 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set ctx_id based on device
ctx_id = 0 if device.type == 'cuda' else -1

# Initialize FaceAnalysis
app = FaceAnalysis('buffalo_l',providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=ctx_id, det_size=(640, 640))

dir_path = r'output'
list_file = []
for path in os.listdir(dir_path):
    # check if current path is a file
    if os.path.isfile(os.path.join(dir_path, path)):
        full_path = f"{dir_path}/{path}"
        list_file.append(full_path)
print(list_file)

start = time.time()

count = 0 
for file in list_file:
   count = count + 1 
   print(file)
   img = cv2.imread(f"{file}")
   faces = app.get(img)
   print("so mat", len(faces))

end = time.time()

print("excution time", end - start)