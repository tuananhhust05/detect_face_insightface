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
from pinecone import Pinecone
import subprocess
import threading
# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

pc = Pinecone(api_key="be4036dc-2d41-4621-870d-f9c4e8958412")
index = pc.Index("facejackma")

weight_point = 0.4

# torch for handling vector 
def cosin(question, answer):
    question = torch.tensor(question).to(device)
    answer = torch.tensor(answer).to(device)
    cosine = torch.dot(question, answer) / (torch.norm(question) * torch.norm(answer))
    return cosine.item()  # Return as scalar

array_em = []
app = FaceAnalysis('buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))  # Ensure InsightFace uses GPU
list_result = []

def extract_frames(video_file,index_local):
    array_em_result = []
    list_result_ele = []
    frame_count = 0
    frame_rate = 60  #  default 1s with 30 frames
    audio = MP4(video_file)
    duration = audio.info.length
    print("duration", duration)
    cap = cv2.VideoCapture(video_file)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        print("frame_count", frame_count)

        if frame_count % frame_rate == 0:
            # Sharpen and denoise the image
            sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            sharpen = cv2.filter2D(frame, 0, sharpen_kernel)
            frame = cv2.fastNlMeansDenoisingColored(sharpen, None, 10, 10, 7, 21)

            faces = app.get(frame)
            for face in faces:
                if face["det_score"] > 0.5:
                    embedding = torch.tensor(face['embedding']).to(device)  # Move embedding to GPU
                    search_result = index.query(
                        vector=embedding.tolist(),
                        top_k=1,
                        include_metadata=True,
                        include_values=True,
                        filter={"face": 0},
                    )
                    matches = search_result["matches"]

                    if len(matches) > 0 and matches[0]['score'] > weight_point:
                        if len(array_em_result) == 0:
                            array_em_result.append({
                                "speaker": 0,
                                "frames": [frame_count],
                            })
                        else:
                            array_em_result[0]["frames"].append(frame_count)

                        try:
                            bbox = [int(b) for b in face['bbox']]
                            filename = f"{frame_count}_0_face.jpg"
                            if not os.path.exists(f"./faces/{index_local}"):
                                os.makedirs(f"./faces/{index_local}")
                            if not os.path.exists(f"./outputs/{index_local}"):
                                os.makedirs(f"./outputs/{index_local}")

                            cv2.imwrite(f'./faces/{index_local}/{filename}', frame[bbox[1]:bbox[3], bbox[0]:bbox[2], ::-1])
                            cv2.imwrite(f'./outputs/{index_local}/{filename}', frame)
                        except Exception as e:
                            print(f"Error saving frame: {e}")
    
    for ele in array_em_result:
        ele["frame_count"] = frame_count
        ele["duration"] = duration
        ele["frame_rate"] = frame_rate
    
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

    cap.release()
    print("End video")

def groupJson(video_file,count_thread):
    final_result = []
    audio = MP4(video_file)
    duration = audio.info.length
    time_per_segment = duration / count_thread
    print("duration",time_per_segment, duration)
    list_stt = []
    for path in os.listdir("results"):
        if os.path.isfile(os.path.join("results", path)):
            stt = int(path.split(".")[0])
            list_stt.append(stt)
           
    list_stt=sorted(list_stt)
    for stt in list_stt:
        with open(f"results/{stt}.json", 'r') as file:
           data = json.load(file)
           print(data)
           if(len(data) > 0):
                data = data[0]
                for duration in data["duration_exist"]:
                    final_result.append([duration[0] + stt * time_per_segment,duration[1] + stt * time_per_segment])
           print(f"Result after file {stt}",final_result )
    with open(f"final_result.json", 'w') as f:
        json.dump(final_result, f, indent=4)
        print("End video") 

def trimvideo(videofile,count_thread):
    audio = MP4(videofile)
    duration = audio.info.length
    time_per_segment = duration / count_thread
    for i in range(count_thread):
        if(i == count_thread - 1):
            command = f"ffmpeg -i ../video.mp4 -ss {time_per_segment*i} -c:v copy -c:a copy  videos/{i}.mp4 -y"
            subprocess.run(command, shell=True, check=True)
        else:
            command = f"ffmpeg -i ../video.mp4 -ss {time_per_segment*i} -t {time_per_segment} -c:v copy -c:a copy  videos/{i}.mp4 -y"
            subprocess.run(command, shell=True, check=True)

def process_videos(video_file_origin,count_thread):
    
    trimvideo(video_file_origin,count_thread)
    video_files = [f"videos/{i}.mp4" for i in range(count_thread)]  # Example video file list
    threads = []
    for i, video_file in enumerate(video_files):
        t = threading.Thread(target=extract_frames, args=(video_file,i))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    groupJson(video_file_origin,count_thread)
    print("Processing complete")

# Run with  GPU
start_time = time.time()
process_videos("../video.mp4",40)
end_time = time.time()
print(f"Total execution time: {end_time - start_time}")
