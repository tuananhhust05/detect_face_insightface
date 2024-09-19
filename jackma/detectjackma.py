
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
import asyncio
pc = Pinecone(api_key="be4036dc-2d41-4621-870d-f9c4e8958412")
index = pc.Index("detectface2")

weight_point = 0.4

def cosin(question,answer):
    cosine = np.dot(question,answer)/(norm(question)*norm(answer))
    return cosine

array_cosin = []
array_em = []
app = FaceAnalysis('buffalo_l')
app.prepare(ctx_id = 0, det_size=(640,640))
list_result = []

def handleFrame(frames,start):
    print("so luong frame", frames)

def multiple_extract_frame(video_file):
    frame_count = 0
    frame_rate = 60  # default 1s with 30 frame
    list_frame = []
    audio = MP4(video_file)
    duration = audio.info.length

    cap = cv2.VideoCapture(video_file)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print( length )
    print("duration",duration)
    print("frame per second", length/duration)
  

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
              if(face["det_score"] > 0.5):
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
                
                print("matches", matches)
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

@jit(target_backend='cuda')
def extract_per_file(video_file,index_local):
    array_em_result = []
    list_result_ele = []
    frame_count = 0
    frame_rate = 60  # default 1s with 30 frame
    duration = 0 
    audio = MP4(video_file)
    duration = audio.info.length
    print("duration",duration)
    cap = cv2.VideoCapture(video_file)
    
    while True :
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
              if(face["det_score"] > 0.5):
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
                
                if(len(matches) > 0):
                    if(matches[0]['score'] > weight_point ):
                        if(len(array_em_result) == 0):
                            array_em_result.append({
                                    "speaker":0,
                                    "frames":[frame_count],
                                }
                            )
                        else:
                            array_em_result[0]["frames"].append(frame_count)
                        
                        filename = f"0_face.jpg"
                        try:
                            bbox = face['bbox']
                            bbox = [int(b) for b in bbox]
                            filename = f"{frame_count}_{filename}"
                            cv2.imwrite('./faces/%s'%filename,frame[bbox[1] : bbox[3], bbox[0]: bbox[2], ::-1])
                            cv2.imwrite('./outputs/%s'%filename,frame)
                        except:
                            print("Saving error")
    
    for ele in array_em_result:
        ele["frame_count"] = frame_count
        ele["duration"] = duration
        ele["frame_rate"] = frame_rate
    cap.release()

    with open(f"datas/{index_local}.json", 'w') as f:
       json.dump(array_em_result, f, indent=4)

    with open(f"datas/{index_local}.json", 'r') as file:
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
            list_result_ele.append({
                'face':em['speaker'],
                'duration_exist':list_time_exist
            })

    with open(f"results/{index_local}.json", 'w') as f:
        json.dump(list_result_ele, f, indent=4)
        print("End video")

import threading

async def main():
    start = time.time() 
    t1 = threading.Thread(target=extract_per_file, args=("videos/1.mp4",1))
    t2 = threading.Thread(target=extract_per_file, args=("videos/2.mp4",2))
    t3 = threading.Thread(target=extract_per_file, args=("videos/3.mp4",3))
    t4 = threading.Thread(target=extract_per_file, args=("videos/4.mp4",4))
    t5 = threading.Thread(target=extract_per_file, args=("videos/5.mp4",5))
    t6 = threading.Thread(target=extract_per_file, args=("videos/6.mp4",6))
    t7 = threading.Thread(target=extract_per_file, args=("videos/7.mp4",7))
    t8 = threading.Thread(target=extract_per_file, args=("videos/8.mp4",8))

    t1.start()
    t2.start()
    t3.start()
    t4.start()
    t5.start()
    t6.start()
    t7.start()
    t8.start()

    t1.join()
    t2.join()
    t3.join()
    t4.join()
    t5.join()
    t6.join()
    t7.join()
    t8.join()

    print("Done!")
    end = time.time() 
    print("excution time", end - start)
asyncio.run(main())

