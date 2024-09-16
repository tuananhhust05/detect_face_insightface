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
weight_point = 0.4

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
    audio = MP4("video.mp4")
    duration = audio.info.length
    print("duration",duration)

    cap = cv2.VideoCapture(video_file)
    frame_rate = 2  # Desired frame rate (1 frame every 0.5 seconds)
    frame_count = 0
    
    # Get the video file's name without extension
    video_name = os.path.splitext(os.path.basename(video_file))[0]
    
    # Create an output folder with a name corresponding to the video
    output_directory = f"{video_name}_frames"
    os.makedirs(output_directory, exist_ok=True)
    
   
    while True :
        # time.sleep(0.1)
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # if(frame_count > 5):
        #     break
        frame_count += 1
        print("frame_count",frame_count)
     
        if(frame_count % 6 == 0):
   

            # Deblur the image
            sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpen = cv2.filter2D(frame, 0 , sharpen_kernel)

            frame = cv2.fastNlMeansDenoisingColored(sharpen,None,10,10,7,21)
            cv2.imwrite('tempt.jpg',frame)
            # frame = cv2.fastNlMeansDenoisingColored(sharpen,None,10,10,7,21)
            faces = app.get(frame)
            for i,face in enumerate(faces):
                if(len(array_em) == 0):
                    bbox = face['bbox']
                    bbox = [int(b) for b in bbox]
                    filename=f"0.0.jpg"
                    cv2.imwrite('./faces/%s'%filename,frame[bbox[1] : bbox[3], bbox[0]: bbox[2], ::-1])
                    array_em.append({
                        "speaker":0,
                        "frames":[frame_count],
                        "embeddings":[face['embedding']]
                    })
                    filename=f"{frame_count}_0.jpg"
                    cv2.imwrite('./outputs/%s'%filename,frame)
                else:
                    flag  = False 
                    print("so luong  nguoi hien tai", len(array_em))
                    stt = 0  # số thứ tự của face khớp với face mới detect được 
                    for j in range(len(array_em)):
                        em = array_em[j]
                        # print("em",em)
                        print("so mat", len(faces))
                        print("so phan tu con",len(em["embeddings"]), frame_count)
                        # print("phan tu dau tien", em["embeddings"][0])
                        count = 0
                        for x in range(len(em["embeddings"])):
                            # if(  ( ( len(em["embeddings"]) > 6 ) and (x >  (len(em["embeddings"]) - 6 )) ) or (len(em["embeddings"]) <= 6) ):
                                # time.sleep(0.1)
                                # print("phan tu con",x)
                                cosin_value = cosin(em["embeddings"][x],face['embedding'])
                                # print("cosin_value",cosin_value)
                                # print("count speaker", len(array_em))
                                if(cosin_value >  weight_point):
                                    stt = j
                                    flag = True
                                count = count + 1 
                            
                        print("so lan tinh ....... ", count)
                    if (flag == False): 
                        
                        array_em.append({
                                "speaker":len(array_em),
                                "frames":[],
                                "embeddings":[face['embedding']]
                            }
                        )
                        filename = f"{len(array_em) -1 }_face.jpg"
                        bbox = face['bbox']
                        bbox = [int(b) for b in bbox]
                        try:
                            filename = f"{frame_count}_{filename}"
                            cv2.imwrite('./faces/%s'%filename,frame[bbox[1] : bbox[3], bbox[0]: bbox[2], ::-1])
                            cv2.imwrite('./outputs/%s'%filename,frame)
                        except:
                            print("End video") 
                            return
                    if(flag == True):
                        array_em[stt]["embeddings"].append(face['embedding'])
                        array_em[stt]["frames"].append(frame_count)

                        filename = f"{stt}_face.jpg"
                        bbox = face['bbox']
                        bbox = [int(b) for b in bbox]
                        try:
                            filename = f"{frame_count}_{filename}"
                            cv2.imwrite('./faces/%s'%filename,frame[bbox[1] : bbox[3], bbox[0]: bbox[2], ::-1])
                            cv2.imwrite('./outputs/%s'%filename,frame)
                        except:
                            print("Error saving") 

        # print(f"Frame {frame_count} has been extracted and saved as {output_file}")
    for ele in array_em:
        del(ele['embeddings'])
        ele["frame_count"] = frame_count
        ele["duration"] = duration
        ele["frame_rate"] = 6
    cap.release()
    print("End video")
import json 

start = time.time() 
extract_frames('video.mp4')
print("array_cosin",array_cosin)
with open('data.json', 'w') as f:
    json.dump(array_em, f, indent=4)
# print("array_em",array_em)
print("array_em",len(array_em))

list_result = []
# Open and read the JSON file
with open('data.json', 'r') as file:
    data = json.load(file)
    for em in data:
        frame_rate = em["frame_rate"] 
        time_per_frame = em["duration"] / em["frame_count"]
        list_time_exist = []
        duration_exist = []
        list_frame = em["frames"]
        print(list_frame)
        print("so frame", len(list_frame))
        for i in range(len(list_frame)-1):
           if(list_frame[i] == frame_rate):
              duration_exist.append(0)
           duration_exist.append(list_frame[i])
           if( (list_frame[i + 1] - list_frame[i]) > frame_rate):
               list_time_exist.append(duration_exist[0]*time_per_frame)
               list_time_exist.append(duration_exist[len(duration_exist) - 1] * time_per_frame)
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
    json.dump(array_em, f, indent=4)

end = time.time() 
print("excution time", end - start)