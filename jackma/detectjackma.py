
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
from pinecone import Pinecone, ServerlessSpec
import uuid
import json 
import numba
from numba import jit
pc = Pinecone(api_key="be4036dc-2d41-4621-870d-f9c4e8958412")
index = pc.Index("detectface2")

weight_point = 0.5

def cosin(question,answer):
    cosine = np.dot(question,answer)/(norm(question)*norm(answer))
    return cosine

array_cosin = []
array_em = []
app = FaceAnalysis('buffalo_l')
app.prepare(ctx_id = 0, det_size=(640,640))




list_result = []

def extract_frames(video_file):
    frame_count = 0
    frame_rate = 60  # default 1s with 30 frame
    duration = 0 
    audio = MP4("../video.mp4")
    duration = audio.info.length
    print("duration",duration)
    cap = cv2.VideoCapture(video_file)
    
    while True :
        # time.sleep(0.1)
        ret, frame = cap.read()
        
        if not ret:
            break

        frame_count = frame_count +  1
        print("frame_count",frame_count)
     
        if(frame_count % frame_rate == 0):
   
            
            # Deblur the image
            sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpen = cv2.filter2D(frame, 0 , sharpen_kernel)
            frame = cv2.fastNlMeansDenoisingColored(sharpen,None,10,10,7,21)
            faces = app.get(frame)
            for i,face in enumerate(faces):
              if(face["det_score"] > 0.6):
                search_result = index.query(
                                vector=face['embedding'].tolist(),
                                top_k=1,
                                include_metadata=True,
                                include_values=True,
                                filter={
                                    "face":0
                                },
                            )
                matches = search_result["matches"]
                
#                print("matches", matches)
                if(len(matches) > 0):
                    if(matches[0]['score'] > weight_point ):
                        if(len(array_em) == 0):
                            array_em.append({
                                    "speaker":0,
                                    "frames":[frame_count],
                                }
                            )
                        else:
                            array_em[0]["frames"].append(frame_count)
                        
                        filename = f"0_face.jpg"
                        try:
                            bbox = face['bbox']
                            bbox = [int(b) for b in bbox]
                            filename = f"{frame_count}_{filename}"
                            cv2.imwrite('./faces/%s'%filename,frame[bbox[1] : bbox[3], bbox[0]: bbox[2], ::-1])
                            cv2.imwrite('./outputs/%s'%filename,frame)
                        except:
                            print("Saving error")
    
    for ele in array_em:
        ele["frame_count"] = frame_count
        ele["duration"] = duration
        ele["frame_rate"] = frame_rate
    cap.release()
    print("End video")


start = time.time() 
extract_frames('../video.mp4')



    
print("array_em",array_em)
print("array_em",len(array_em))

with open('data.json', 'w') as f:
    json.dump(array_em, f, indent=4)

# Open and read the JSON file
with open('data.json', 'r') as file:
    data = json.load(file)
    for em in data:
        frame_rate = em["frame_rate"] 
        time_per_frame = em["duration"] / em["frame_count"]
        list_time_exist = []
        duration_exist = []
        list_frame = em["frames"]
        # print(list_frame)
        print("so frame", len(list_frame))
        for i in range(len(list_frame)-1):
           if(list_frame[i] == frame_rate):
              duration_exist.append(0)
           duration_exist.append(list_frame[i])
           if( (list_frame[i + 1] - list_frame[i]) > frame_rate):
               list_time_exist.append([duration_exist[0]*time_per_frame,duration_exist[len(duration_exist) - 1] * time_per_frame])
               duration_exist = []
           else:
                if( i == len(list_frame)-2):
                    duration_exist.append(list_frame[i+1])
                    list_time_exist.append([duration_exist[0]*time_per_frame,duration_exist[len(duration_exist) - 1] * time_per_frame])
                    duration_exist = []
        list_result.append({
            'face':em['speaker'],
            'duration_exist':list_time_exist
        })

with open('result.json', 'w') as f:
    json.dump(list_result, f, indent=4)

end = time.time() 
print("excution time", end - start)
