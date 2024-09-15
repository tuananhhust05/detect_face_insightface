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
weight_point = 0.5

def cosin(question,answer):
    cosine = np.dot(question,answer)/(norm(question)*norm(answer))
    return cosine

array_cosin = []
array_em = []
app = FaceAnalysis('buffalo_l')
app.prepare(ctx_id = 0, det_size=(640,640))
img = cv2.imread("./videotest_frames/frame_11.jpg") 
fig,axs = plt.subplots(1,6,figsize=(12,5))
# faces = app.get(img)

# for i,face in enumerate(faces):
#     bbox = face['bbox']
#     bbox = [int(b) for b in bbox]
#     filename = f"{i}test.jpg"
#     cv2.imwrite('./outputs/%s'%filename,img[bbox[1] : bbox[3], bbox[0]: bbox[2], ::-1])
#     array_em.append(face['embedding'])

# img2 = cv2.imread("./videotest_frames/frame_22.jpg")
# faces2 = app.get(img2)
# for i,face in enumerate(faces2):
#     bbox = face['bbox']
#     bbox = [int(b) for b in bbox]
#     filename = f"{i}test.jpg"
#     cv2.imwrite('./outputs/%s'%filename,img[bbox[1] : bbox[3], bbox[0]: bbox[2], ::-1])
#     for em in array_em:
#       cosin_value = cosin(em,face['embedding'])
#       array_cosin.append(cosin_value)
     #   array_em.append(face['embedding'])

def extract_frames(video_file):
    cap = cv2.VideoCapture(video_file)
    count = 0
    frame_rate = 2  # Desired frame rate (1 frame every 0.5 seconds)
    frame_count = 0
    
    # Get the video file's name without extension
    video_name = os.path.splitext(os.path.basename(video_file))[0]
    
    # Create an output folder with a name corresponding to the video
    output_directory = f"{video_name}_frames"
    os.makedirs(output_directory, exist_ok=True)
    
   
    while True and (frame_count < 3000) :
        time.sleep(0.1)
        ret, frame = cap.read()
        
        if not ret:
            break
        
        frame_count += 1
        print("frame_count",frame_count)
        # Only extract frames at the desired frame rate
        # if frame_count % int(cap.get(5) / frame_rate) == 0:
            # output_file = f"{output_directory}/frame_{frame_count}.jpg"
            # cv2.imwrite(output_file, frame)
        faces = app.get(frame)
        for i,face in enumerate(faces):
            if(len(array_em) == 0):
                bbox = face['bbox']
                bbox = [int(b) for b in bbox]
                filename=f"1.jpg"
                cv2.imwrite('./faces/%s'%filename,frame[bbox[1] : bbox[3], bbox[0]: bbox[2], ::-1])
                array_em.append({
                    "speaker":1,
                    "frames":[frame_count],
                    "embeddings":[face['embedding']]
                })
                filename=f"{frame_count}_1.jpg"
                cv2.imwrite('./outputs/%s'%filename,frame)
            else:
                flag  = False 
                print("soluong  nguoi hien tai", len(array_em))
                for em in array_em:
                    # print("em",em)
                    print("so phan tu con",len(em["embeddings"]))
                    print("phan tu dau tien", em["embeddings"][0])
                    for embed in em["embeddings"]:
                        # print("phan tu con")
                        cosin_value = cosin(embed,face['embedding'])
                        count = count + 1 
                        # print("so lan tinh", count)
                        # print("cosin_value",cosin_value)
                        # print("count speaker", len(array_em))
                        if(cosin_value >  weight_point):
                           flag = True
                           em["embeddings"].append(face['embedding'])
                        em["frames"].append(frame_count)

                        filename = f"{len(array_em)}_face.jpg"
                        bbox = face['bbox']
                        bbox = [int(b) for b in bbox]
                        try:
                            filename = f"{frame_count}_{filename}"
                            cv2.imwrite('./outputs/%s'%filename,frame)
                        except:
                            print("Error saving") 
                if (flag == False): 
                    array_em.append({
                            "speaker":len(array_em),
                            "frames":[],
                            "embeddings":[face['embedding']]
                        }
                    )
                    filename = f"{len(array_em)}_face.jpg"
                    bbox = face['bbox']
                    bbox = [int(b) for b in bbox]
                    try:
                        cv2.imwrite('./faces/%s'%filename,frame[bbox[1] : bbox[3], bbox[0]: bbox[2], ::-1])
                        filename = f"{frame_count}_{filename}"
                        cv2.imwrite('./outputs/%s'%filename,frame)
                    except:
                        print("End video") 
        # print(f"Frame {frame_count} has been extracted and saved as {output_file}")
    for ele in array_em:
        del(ele['embedding'])
    cap.release()
    print("End video")
import json 
extract_frames('video.mp4')
print("array_cosin",array_cosin)
with open('data.json', 'w') as f:
    json.dump(array_em, f, indent=4)
print("array_em",array_em)
print("array_em",len(array_em))
