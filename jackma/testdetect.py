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
import matplotlib.pyplot as plt 
# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

pc = Pinecone(api_key="be4036dc-2d41-4621-870d-f9c4e8958412")
index = pc.Index("facejackma")

weight_point = 0.4

def getduration(file):
    data = cv2.VideoCapture(file) 
    # count the number of frames 
    frames = data.get(cv2.CAP_PROP_FRAME_COUNT) 
    fps = data.get(cv2.CAP_PROP_FPS) 
    data.release()
    # calculate duration of the video 
    if fps > 0 :
      seconds = round(frames / fps) 
    else:
      seconds = 0
    return seconds

# torch for handling vector 
def cosin(question, answer):
    question = torch.tensor(question).to(device)
    answer = torch.tensor(answer).to(device)
    cosine = torch.dot(question, answer) / (torch.norm(question) * torch.norm(answer))
    return cosine.item()  # Return as scalar

ctx_id = 0 if device.type == 'cuda' else -1
array_em = []
app = FaceAnalysis('buffalo_l')
app.prepare(ctx_id=ctx_id, det_size=(640, 640))
print(f"FaceAnalysis is using {'GPU' if ctx_id >=0 else 'CPU'}")
list_result = []

def extract_frames(folder,video_file,index_local,time_per_segment):
    array_em_result = []
    list_result_ele = []
    frame_count = 0
    frame_rate = 60  #  default 1s with 30 frames
    # audio = MP4(video_file)
    # duration = audio.info.length
    duration = getduration(video_file)
    print("duration", duration)
    cap = cv2.VideoCapture(video_file)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        print("frame_count", frame_count)

        if frame_count % frame_rate == 0:
#             gpu_image = cv2.cuda_GpuMat()
#             gpu_image.upload(frame)

# # Define a sharpening kernel
#             sharpen_kernel = np.array([[-1, -1, -1], 
#                            [-1,  9, -1], 
#                            [-1, -1, -1]], dtype=np.float32)

# # Create a filter 2D operation on the GPU
#             gpu_sharpen_filter = cv2.cuda.createFilter2D(gpu_image.type(), -1, sharpen_kernel)

# # Apply the filter to sharpen the image
#             gpu_sharp = gpu_sharpen_filter.apply(gpu_image)

# # A workaround for using denoising, Color often need CPU fall-back
# # You could use another GPU-based noise reduction technique or manage sections in CPU
#             gpu_result = gpu_sharp.download()

#             # This will denoise but is on CPU (not optimal here for GPU workflow):
#             frame_denoised = cv2.fastNlMeansDenoisingColored(gpu_result, None, 10, 10, 7, 21)


            faces = app.get(frame)
            for face in faces:
                print(face)
                if face["det_score"] > 0.5:
                    # embedding = torch.tensor(face['embedding']).to(device)  # Move embedding to GPU
                    # search_result = index.query(
                    #     vector=embedding.tolist(),
                    #     top_k=1,
                    #     include_metadata=True,
                    #     include_values=True,
                    #     filter={"face": 0},
                    # )
                    # matches = search_result["matches"]

                    # if len(matches) > 0 and matches[0]['score'] > weight_point:
                    if True:
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
                            if not os.path.exists(f"./faces/{folder}/{index_local}"):
                                os.makedirs(f"./faces/{folder}/{index_local}")
                            if not os.path.exists(f"./outputs/{folder}/{index_local}"):
                                os.makedirs(f"./outputs/{folder}/{index_local}")

                            cv2.imwrite(f'./faces/{folder}/{index_local}/{filename}', frame[bbox[1]:bbox[3], bbox[0]:bbox[2], ::-1])

                            top_left = (bbox[0], bbox[1])
                            bottom_right = (bbox[2], bbox[3])
                            color = (255, 0, 0)
                            thickness = 2
                            cv2.rectangle(frame, top_left, bottom_right, color, thickness)
                            time_per_frame = duration / total_frames
                            text = frame_count * time_per_frame + time_per_segment*index_local
                            text = str(text)
                            position = (bbox[0], bbox[1])
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            font_scale = 1

                            cv2.putText(frame, text, position, font, font_scale, color, thickness)
                            cv2.imwrite(f'./outputs/{folder}/{index_local}/{filename}', frame)
                        except Exception as e:
                            print(f"Error saving frame: {e}")
    
    for ele in array_em_result:
        ele["frame_count"] = frame_count
        ele["duration"] = duration
        ele["frame_rate"] = frame_rate
    
    with open(f"datas/{folder}/{index_local}.json", 'w') as f:
       json.dump(array_em_result, f, indent=4)
    
    with open(f"datas/{folder}/{index_local}.json", 'r') as file:
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


    with open(f"results/{folder}/{index_local}.json", 'w') as f:
        json.dump(list_result_ele, f, indent=4)
        print("End video")       

    cap.release()
    print("End video")

def groupJson(folder,video_file,count_thread):
    final_result = []
    # audio = MP4(video_file)
    # duration = audio.info.length
    duration = getduration(video_file)
    time_per_segment = duration / count_thread
    print("duration",time_per_segment, duration)
    list_stt = []
    for path in os.listdir(f"results/{folder}"):
        if os.path.isfile(os.path.join(f"results/{folder}", path)):
            stt = int(path.split(".")[0])
            list_stt.append(stt)
           
    list_stt=sorted(list_stt)
    for stt in list_stt:
        with open(f"results/{folder}/{stt}.json", 'r') as file:
           data = json.load(file)
           print(data)
           if(len(data) > 0):
                data = data[0]
                for duration in data["duration_exist"]:
                    final_result.append([duration[0] + stt * time_per_segment,duration[1] + stt * time_per_segment])
           print(f"Result after file {stt}",final_result )
    with open(f"final_result/{folder}/final_result.json", 'w') as f:
        json.dump(final_result, f, indent=4)
        print("End video") 

def trimvideo(folder,videofile,count_thread):
    # audio = MP4(videofile)
    # duration = audio.info.length
    duration = getduration(videofile)
    time_per_segment = duration / count_thread
    for i in range(count_thread):
        if(i == count_thread - 1):
            command = f"ffmpeg -i {videofile} -ss {time_per_segment*i} -c:v copy -c:a copy  videos/{folder}/{i}.mp4 -y"
            subprocess.run(command, shell=True, check=True)
        else:
            command = f"ffmpeg -i {videofile} -ss {time_per_segment*i} -t {time_per_segment} -c:v copy -c:a copy  videos/{folder}/{i}.mp4 -y"
            subprocess.run(command, shell=True, check=True)


    
def process_videos(folder,video_file_origin,count_thread):
    print("process_videos", folder,video_file_origin,count_thread)
    # audio = MP4(video_file_origin)
    # duration = audio.info.length
    duration = getduration(video_file_origin)
    time_per_segment = duration / count_thread

    trimvideo(folder,video_file_origin,count_thread)
    video_files = [f"videos/{folder}/{i}.mp4" for i in range(count_thread)]  # Example video file list
    threads = []
    for i, video_file in enumerate(video_files):
        t = threading.Thread(target=extract_frames, args=(folder,video_file,i,time_per_segment))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    groupJson(folder,video_file_origin,count_thread)
    print("Processing complete")

def handle_multiplefile(listfile,thread):
    for file in listfile:
        file_name = file.split(".")[0]
        if "/" in file_name: 
            file_name = file_name.split("/")[len(file_name.split("/")) - 1]
        if not os.path.exists(f"./faces/{file_name}"):
            os.makedirs(f"./faces/{file_name}")
        

        if not os.path.exists(f"./outputs/{file_name}"):
            os.makedirs(f"./outputs/{file_name}")
        

        if not os.path.exists(f"./videos/{file_name}"):
            os.makedirs(f"./videos/{file_name}")
        

        if not os.path.exists(f"./datas/{file_name}"):
            os.makedirs(f"./datas/{file_name}")
       

        if not os.path.exists(f"./results/{file_name}"):
            os.makedirs(f"./results/{file_name}")
       

        if not os.path.exists(f"./final_result/{file_name}"):
            os.makedirs(f"./final_result/{file_name}")
    
        folder = file_name
        process_videos(folder,file,thread)
        subprocess.run("rm -rf videos/{file_name}", shell=True, check=True)


      
# # Run with  GPU
dir_path = r'/mnt/casehdd16tb/DataVideoHTC'
list_file = []
for path in os.listdir(dir_path):
    # check if current path is a file
    if os.path.isfile(os.path.join(dir_path, path)):
        full_path = f"{dir_path}/{path}"
        list_file.append(full_path)
#print(list_file)


# start_time = time.time()
# print("Start ......",str(start_time))
# f = open("start.txt", "a")
# f.write(str(start_time))

handle_multiplefile(["video8p.mp4"],10)

# end_time = time.time()
# f = open("end.txt", "a")
# f.write(str(end_time))

# print(f"Total execution time: {end_time - start_time}")
