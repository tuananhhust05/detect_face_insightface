
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
from imutils.video import FPS 
import uuid 
import json
from flask import Flask, jsonify, request
import pymongo
import ffmpeg
import random
from threading import Thread


myclient = pymongo.MongoClient("mongodb://root:facex@192.168.50.10:27018")

mydb = myclient["faceX"]
facematches = mydb["facematches"]
appearances = mydb["appearances"]
targets = mydb["targets"]
videos = mydb["videos"]
cases = mydb["cases"]

dir_project = "/home/poc4a5000/detect/detect"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


pc = Pinecone(api_key="be4036dc-2d41-4621-870d-f9c4e8958412")
index = pc.Index("detectcamera")

weight_point = 0.4
time_per_frame_global = 0.5
ctx_id = 0 if device.type == 'cuda' else -1
app_recognize = FaceAnalysis('buffalo_l',providers=['CUDAExecutionProvider'])
app_recognize.prepare(ctx_id=ctx_id, det_thresh=0.3, det_size=(640, 640))


num_gpus = torch.cuda.device_count()
print(f"Number of GPUs available: {num_gpus}")
gpu_ids = list(range(num_gpus)) 

list_model_detect = []
for j in range(num_gpus):
    providers = [
        ('CUDAExecutionProvider', {
            'device_id': j,
        })
    ]
    model_ele = model_zoo.get_model(
        '/home/poc4a5000/.insightface/models/buffalo_l/det_10g.onnx',
        providers=providers
    )
    model_ele.prepare(ctx_id=j, det_size=(640, 640))
    list_model_detect.append(model_ele)
# torch.cuda.set_device(gpu_id)
# device = torch.device(f'cuda:{gpu_id}')

# Define providers with device_id
torch.cuda.set_device(0)
providers = [
    ('CUDAExecutionProvider', {
        'device_id': 0,
    })
]
# Load the model with  providers
model = model_zoo.get_model(
    '/home/poc4a5000/.insightface/models/buffalo_l/det_10g.onnx',
    providers=providers
)
model.prepare(ctx_id=0, det_size=(640, 640))

list_model_analyst = []
for j in range(num_gpus):
    providers = [
        ('CUDAExecutionProvider', {
            'device_id': j,
        })
    ]
    app_ele = FaceAnalysis('buffalo_l',providers=providers)
    app_ele.prepare(ctx_id=j,det_thresh=0.4, det_size=(640, 640))
    list_model_analyst.append(app_ele)

list_vector = []

list_vector_other = []

list_vector_widden = []

def getduration(file):
    data = cv2.VideoCapture(file) 
    frames = data.get(cv2.CAP_PROP_FRAME_COUNT) 
    fps = data.get(cv2.CAP_PROP_FPS) 
    data.release()
    seconds = round(frames / fps) 
    return seconds


def cosin(question, answer):
    question = torch.tensor(question).to(device)
    answer = torch.tensor(answer).to(device)
    cosine = torch.dot(question, answer) / (torch.norm(question) * torch.norm(answer))
    return cosine.item()  

def checkface(vector):
    global list_vector 
    for ele in list_vector:
        cos = cosin(ele,vector)
        if(cos > weight_point):
            return cos
    return 0


def current_date():
  format_date = "%Y-%m-%d %H:%M:%S"
  now = datetime.datetime.now()
  date_string = now.strftime(format_date)
  return datetime.datetime.strptime(date_string, format_date)

class VideoCaptureThreading:
    def __init__(self, src=0):
        self.src = src
        self.cap = cv2.VideoCapture(self.src)
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
        self.ret, self.frame = self.cap.read()
        self.running = True
        self.thread = Thread(target=self.update, args=())
        self.thread.start()

    def update(self):
        while self.running:
            self.ret, self.frame = self.cap.read()

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.running = False
        self.thread.join()

    def release(self):
        self.cap.release()

def extract_frames(folder,video_file,index_local,time_per_segment,case_id,gpu_id):
    array_em_result = []
    list_result_ele = []
    frame_count = 0 
    duration = getduration(video_file)
    cap2 = cv2.VideoCapture(video_file, cv2.CAP_FFMPEG)
    cap = VideoCaptureThreading(video_file)

    fps = cap2.get(cv2.CAP_PROP_FPS)
    fps = ( fps + 1 ) // 1
    frame_rate = time_per_frame_global * fps 


    sum_age = 0 
    sum_gender = 0 
    count_face = 0 
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        
        if frame_count % frame_rate == 0:
            print("frame_count", frame_count)
            
            # gpu_frame = cv2.cuda_GpuMat()
            # gpu_frame.upload(frame)
            # # Now the frame is on GPU memory. You can perform GPU-based processing here.
            # # For demonstration: download it back to CPU and show it
            # frame = gpu_frame.download()
            # facechecks = model.detect(frame,input_size=(640, 640))
            # frame = cv2.resize(frame, (640, 640))
            # sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            # sharpen = cv2.filter2D(frame, 0, sharpen_kernel)
            # frame = cv2.fastNlMeansDenoisingColored(sharpen, None, 10, 10, 7, 21)
            facechecks = list_model_detect[gpu_id].detect(frame,input_size=(640, 640))
            flagDetect = False
            if(len(facechecks) > 0):
                if(len(facechecks[0]) > 0):
                    flagDetect = True
            
            if(flagDetect == True):
                # gpu_frame = cv2.cuda_GpuMat()
                # gpu_frame.upload(frame)
                print("Có mặt......")
                # sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                # sharpen = cv2.filter2D(frame, 0, sharpen_kernel)
                # frame = cv2.fastNlMeansDenoisingColored(sharpen, None, 10, 10, 7, 21)
                # gpu_frame = denoiser.denoise(gpu_frame)
                # frame = gpu_frame.download()
                faces = list_model_analyst[gpu_id].get(frame)

                for face in faces:
                    if face["det_score"] > 0.5:
         
                        similarity  = checkface(face['embedding'])
                        if(similarity > 0):
                        # if True:
                            count_face = count_face + 1 
                            sum_age = sum_age + int(face['age'])
                            sum_gender = sum_gender + int(face['gender'])
 
                            if len(array_em_result) == 0:
                                array_em_result.append({
                                    "speaker": 0,
                                    "gender":int(face['gender']),
                                    "age":int(face['age']),
                                    "frames": [frame_count],
                                })
                            else:
                                if(count_face > 0):
                                    array_em_result[0]["age"] = sum_age / count_face 
                                    array_em_result[0]["gender"] = sum_gender / count_face 
                                    array_em_result[0]["frames"].append(frame_count)

                            try:
                                bbox = [int(b) for b in face['bbox']]
                                filename = f"{frame_count}_0_face.jpg"
                                if not os.path.exists(f"./faces/{case_id}/{folder}/{index_local}"):
                                    os.makedirs(f"./faces/{case_id}/{folder}/{index_local}")
                                if not os.path.exists(f"./outputs/{case_id}/{folder}/{index_local}"):
                                    os.makedirs(f"./outputs/{case_id}/{folder}/{index_local}")
                                
                                cv2.imwrite(f'./faces/{case_id}/{folder}/{index_local}/{filename}', frame[bbox[1]:bbox[3], bbox[0]:bbox[2]])

                                top_left = (bbox[0], bbox[1])
                                bottom_right = (bbox[2], bbox[3])
                                color = (255, 0, 0)
                                thickness = 2
                                cv2.rectangle(frame, top_left, bottom_right, color, thickness)
                          
                                cv2.imwrite(f'./outputs/{case_id}/{folder}/{index_local}/{filename}', frame)
                            except Exception as e:
                                print(f"Error saving frame: {e}")
                            
                            mydict = { 
                                       "id":  str(uuid.uuid4()), 
                                       "case_id": case_id,
                                       "face_id": 0,
                                       "similarity_face":similarity,
                                       "gender":int(face['gender']),
                                       "age":int(face['age']),
                                       "time_invideo":"",
                                       "proofImage":f'/home/poc4a5000/detect/detect/faces/{case_id}/{folder}/{index_local}/{filename}',
                                       "url":f'/home/poc4a5000/detect/detect/faces/{case_id}/{folder}/{index_local}/{filename}',
                                       "createdAt":current_date(),
                                       "updatedAt":current_date(),
                                       "file":folder
                                    }
                            facematches.insert_one(mydict)
                            mydict["embedding"] = face['embedding']
                            list_vector_widden.append(mydict)

                            global list_vector 
                            if(len(list_vector) < 100):
                                list_vector.append(face['embedding'])
                        else:
                          
                            try:
                                bbox = [int(b) for b in face['bbox']]
                                filename = f"{frame_count}_{str(uuid.uuid4())}_face.jpg"
                                if not os.path.exists(f"./faces/{case_id}/{folder}/{index_local}"):
                                    os.makedirs(f"./faces/{case_id}/{folder}/{index_local}")
                                if not os.path.exists(f"./outputs/{case_id}/{folder}/{index_local}"):
                                    os.makedirs(f"./outputs/{case_id}/{folder}/{index_local}")

                                cv2.imwrite(f'./faces/{case_id}/{folder}/{index_local}/{filename}', frame[bbox[1]:bbox[3], bbox[0]:bbox[2]])

                            except Exception as e:
                                print(f"Error saving frame: {e}")

                            mydict = { 
                                       "id":  str(uuid.uuid4()), 
                                       "case_id": case_id,
                                       "embedding": face['embedding'],
                                       "similarity_face":similarity,
                                       "gender":int(face['gender']),
                                       "age":int(face['age']),
                                       "time_invideo":"",
                                       "proofImage":f'/home/poc4a5000/detect/detect/faces/{case_id}/{folder}/{index_local}/{filename}',
                                       "url":f'/home/poc4a5000/detect/detect/faces/{case_id}/{folder}/{index_local}/{filename}',
                                       "createdAt":current_date(),
                                       "updatedAt":current_date(),
                                       "file":folder
                                    }
                            list_vector_other.append(mydict)

    for ele in array_em_result:
        ele["frame_count"] = frame_count
        ele["duration"] = duration
        ele["frame_rate"] = frame_rate
        
    with open(f"datas/{case_id}/{folder}/{index_local}.json", 'w') as f:
       json.dump(array_em_result, f, indent=4)
    
    with open(f"datas/{case_id}/{folder}/{index_local}.json", 'r') as file:
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
                   
                    file_path_ele = f"{dir_project}/faces/{case_id}/{folder}/{index_local}/{duration_exist[0]}_0_face.jpg"
                    if duration_exist[0] == 0 :
                        if not os.path.exists(file_path_ele):
                            stt = int(duration_exist[0]) + int(frame_rate)
                            file_path_ele = f"{dir_project}/faces/{case_id}/{folder}/{index_local}/{stt}_0_face.jpg"
                    list_time_exist.append([
                        {"path":file_path_ele, "time":duration_exist[0]*time_per_frame},
                        {"path":"","time":duration_exist[len(duration_exist) - 1] * time_per_frame}
                    ])
                    duration_exist = []
                else:
                        if( i == len(list_frame)-2):
                            duration_exist.append(list_frame[i+1])
                            file_path_ele = f"{dir_project}/faces/{case_id}/{folder}/{index_local}/{duration_exist[0]}_0_face.jpg"
                            if duration_exist[0] == 0 :
                                if not os.path.exists(file_path_ele):
                                    stt = int(duration_exist[0]) + int(frame_rate)
                                    file_path_ele = f"{dir_project}/faces/{case_id}/{folder}/{index_local}/{stt}_0_face.jpg"
                            list_time_exist.append([
                                {"path":file_path_ele, "time":duration_exist[0]*time_per_frame},
                                {"path":"","time":duration_exist[len(duration_exist) - 1] * time_per_frame}
                            ])
                            duration_exist = []
            list_result_ele.append({
                'face':em['speaker'],
                'age':em['age'],
                'gender':em['gender'],
                'duration_exist':list_time_exist
            })


    with open(f"results/{case_id}/{folder}/{index_local}.json", 'w') as f:
        json.dump(list_result_ele, f, indent=4)
    

    cap.release()
    
    return 


def groupJson(folder,video_file,count_thread,case_id):
    final_result = {
        "time":[]
    }
    duration = getduration(video_file)
    time_per_segment = duration / count_thread

    list_stt = []
    for path in os.listdir(f"results/{case_id}/{folder}"):
        if os.path.isfile(os.path.join(f"results/{case_id}/{folder}", path)):
            stt = int(path.split(".")[0])
            list_stt.append(stt)
           
    list_stt=sorted(list_stt)
    sum_age = 0 
    sum_gender = 0 
    count_face = 0 
    for stt in list_stt:
        with open(f"results/{case_id}/{folder}/{stt}.json", 'r') as file:
           data = json.load(file)
           if(len(data) > 0):
                data = data[0]
                sum_age = sum_age + int(data['age'])
                sum_gender = sum_gender + int(data['gender'])
                count_face = count_face + 1 
                for duration in data["duration_exist"]:
                    final_result["time"].append([
                        {"path":duration[0]["path"], "time":duration[0]["time"] + stt * time_per_segment},
                        {"path":"","time":duration[1]["time"] + stt * time_per_segment}
                    ])


    if count_face > 0 : 
        final_result['age'] = sum_age / count_face
        final_result['gender'] = sum_gender / count_face
        facematches.update_many(
            {
                "case_id":case_id,
                "face_id": 0
            },
            {
                "$set":{
                    "gender":sum_gender / count_face,
                    "age": sum_age / count_face
                }
            }
        )

    with open(f"final_result/{case_id}/{folder}/final_result.json", 'w') as f:
        json.dump(final_result, f, indent=4)
    
    final_result["file"] = folder 
    final_result["id"] = str(uuid.uuid4())
    final_result["case_id"] = case_id
    final_result["createdAt"] = current_date()
    final_result["updatedAt"] = current_date()
    new_arr = []

    for time in final_result["time"]:
       new_arr.append(
           {
               "path":time[0]["path"],
               "start":time[0]["time"],
               "end":time[1]["time"],
               "similiarity":random.randint(85, 99 )
           }
       )

    final_result["time"] = new_arr
    appearances.insert_one(final_result)

    return 

def create_video_apperance(case_id,thread_count,folder):
    if not os.path.exists(f"{dir_project}/video_apperance"):
        os.makedirs(f"{dir_project}/video_apperance")
    if not os.path.exists(f"{dir_project}/video_apperance/{case_id}"):
        os.makedirs(f"{dir_project}/video_apperance/{case_id}")
    list_img = []
    list_dir_file = os.listdir(f"{dir_project}/outputs/{case_id}/{folder}")
    for dir in list_dir_file:
        dir_full = f"{dir_project}/outputs/{case_id}/{folder}/{dir}"
        dir_full_new = dir_full
        print(dir_full)
        if  os.path.exists(dir_full_new):
            for path in os.listdir(dir_full_new):
                if os.path.isfile(os.path.join(dir_full_new, path)):
                    full_path = f"{dir_full_new}/{path}"
                    list_img.append(full_path)
    img_array = []
    size=(120,120)
    for filename in list_img:
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size_inter = (width,height)
        size = size_inter
        img_array.append(img)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(f"{dir_project}/video_apperance/{case_id}/{folder}_pre.mp4", fourcc, 2.0, size)
    outputpathpre= f"{dir_project}/video_apperance/{case_id}/{folder}_pre.mp4"
    output = f"{dir_project}/video_apperance/{case_id}/{folder}.mp4"
    outputfinal = f"{dir_project}/video_apperance/{case_id}/final.mp4"
    for i in range(len(img_array)):
        out.write(img_array[i])
    
    out.release()
    
    subprocess.run(f"ffmpeg -i {outputpathpre} -codec:v libx264 -profile:v baseline -level 3.0 -pix_fmt yuv420p {output} -y", shell=True, check=True)
    subprocess.run(f"rm -rf {outputpathpre}", shell=True, check=True)
    


    if not os.path.isfile(outputfinal):
       subprocess.run(f"cp {output} {outputfinal}", shell=True, check=True)
    else:
       try: 
            working_directory = f"/home/poc4a5000/detect/detect/video_apperance/{case_id}"
            listmp4file = os.listdir(working_directory)
            # List of video files
            files = []
            for file in listmp4file:
                if(file != "final.mp4" and "mp4" in file):
                    files.append(file)



            # Create a list of file arguments
            filelist = ''.join(f"file '{os.path.join(working_directory, f)}'\n" for f in files).encode('utf-8')

            # Run the FFmpeg command using subprocess and stdin pipe
            command = [
                "ffmpeg",
                "-hwaccel", "cuda",  
                "-f", "concat",
                "-safe", "0",
                "-protocol_whitelist", "file,pipe",
                "-i", "-",  
                "-c", "copy",
                "-y",  
                "final.mp4"
            ]
            print(command,filelist,working_directory)
            try:
                subprocess.run(
                    command,
                    input=filelist,
                    check=True,
                    text=False,  
                    cwd=working_directory
                )
                print("Videos concatenated successfully into final.mp4")
            except subprocess.CalledProcessError as e:
                print(f"An error occurred: {e}")
            except Exception as ex:
                print(f"An unexpected error occurred: {ex}")

          
       except Exception as e:
            print("error merge file",e)
           
    # subprocess.run(f"rm -rf {output}", shell=True, check=True)

    videos.insert_one({
        "id":str(uuid.uuid4()),
        "case_id":case_id,
        "path":outputfinal,
    })

    return 

def cutvideo(videofile,start,duration,output):
    (
        ffmpeg
        .input(videofile, ss=start, hwaccel='cuda')
        .output(output, t=duration, c='copy')
        .run(overwrite_output=True)
        # ffmpeg
        #     .input(videofile, ss=start, hwaccel='cuda')
        #     .output(output, t=duration, vf=f'scale=640:640', c:v='h264_nvenc', c:a='copy')
        #     .run(overwrite_output=True)
    )
    # (
        # ffmpeg
        # .input(videofile, ss=start, hwaccel='cuda')
        # .output(output, t=duration, vf=f'scale=640:640', vcodec='h264_nvenc', acodec='copy')
        # .run(overwrite_output=True)
        # ffmpeg
        # .input(videofile, ss=start)
        # .filter('scale', 640, 640)
        # .output(
        #     output,
        #     t=duration,
        #     vcodec='h264_nvenc',
        #     preset='fast',
        #     video_bitrate='5M',
        #     acodec='aac',
        #     audio_bitrate='128k'
        # )
        # .run(overwrite_output=True)
        # ffmpeg -ss 00:01:00 -i input.mp4 -t 00:02:00 -vf "scale=426:240" -c:v libx264 -preset fast -crf 23 -c:a aac -b:a 128k output.mp4
    # subprocess.run(f"ffmpeg -ss {start} -i {videofile} -t {duration} -vf \"scale=640:640\" -c:v libx264 -preset fast -crf 23 -c:a aac -b:a 128k {output}", shell=True, check=True)
    # subprocess.run(f"ffmpeg -ss {start} -i {videofile}  -t {duration} -vf \"scale=640:640,pad=640:640:(ow-iw)/2:(oh-ih)/2\" -c:v libx264 -crf 23 {output}", shell=True, check=True)
    
    # )

    return 

def trimvideo(folder,videofile,count_thread,case_id):
    duration = getduration(videofile)
    time_per_segment = duration / count_thread
    threads = []

    for i in range(count_thread):
        t = threading.Thread(target=cutvideo, args=(videofile,time_per_segment*i,time_per_segment,f"/home/poc4a5000/detect/detect/videos/{case_id}/{folder}/{i}.mp4"))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()
    
    return 

    
def process_videos(folder,video_file_origin,count_thread,case_id):
    duration = getduration(video_file_origin)
    time_per_segment = duration / count_thread

    trimvideo(folder,video_file_origin,count_thread,case_id)

    video_files = [f"videos/{case_id}/{folder}/{i}.mp4" for i in range(count_thread)]  
    threads = []
    for i, video_file in enumerate(video_files):
        gpu_id = gpu_ids[i % num_gpus]
        t = threading.Thread(target=extract_frames, args=(folder,video_file,i,time_per_segment,case_id,gpu_id))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    groupJson(folder,video_file_origin,count_thread,case_id)
    create_video_apperance(case_id,count_thread,folder)

    return 


def handle_multiplefile(listfile,thread,case_id):
    for file in listfile:
        file_name = file.split(".")[0]
        if "/" in file_name: 
            file_name = file_name.split("/")[len(file_name.split("/")) - 1]
        
        if not os.path.exists(f"./faces/{case_id}"):
            os.makedirs(f"./faces/{case_id}")

        if not os.path.exists(f"./faces/{case_id}/{file_name}"):
            os.makedirs(f"./faces/{case_id}/{file_name}")
        
        if not os.path.exists(f"./outputs/{case_id}"):
            os.makedirs(f"./outputs/{case_id}")

        if not os.path.exists(f"./outputs/{case_id}/{file_name}"):
            os.makedirs(f"./outputs/{case_id}/{file_name}")
        
        if not os.path.exists(f"./videos/{case_id}"):
            os.makedirs(f"./videos/{case_id}")

        if not os.path.exists(f"./videos/{case_id}/{file_name}"):
            os.makedirs(f"./videos/{case_id}/{file_name}")
        
        if not os.path.exists(f"./datas/{case_id}"):
            os.makedirs(f"./datas/{case_id}")

        if not os.path.exists(f"./datas/{case_id}/{file_name}"):
            os.makedirs(f"./datas/{case_id}/{file_name}")
       
        if not os.path.exists(f"./results/{case_id}"):
            os.makedirs(f"./results/{case_id}")

        if not os.path.exists(f"./results/{case_id}/{file_name}"):
            os.makedirs(f"./results/{case_id}/{file_name}")
        
        if not os.path.exists(f"./final_result/{case_id}"):
            os.makedirs(f"./final_result/{case_id}")

        if not os.path.exists(f"./final_result/{case_id}/{file_name}"):
            os.makedirs(f"./final_result/{case_id}/{file_name}")
    
        folder = file_name
        process_videos(folder,file,thread,case_id)
        subprocess.run("rm -rf videos/{file_name}", shell=True, check=True)
    
    return 

def checkOnArr(arr,num):
    for ele in arr:
        if ele == num:
            return True
    return False

def handle_other_face():
    try:
        face_id_max = 1 
        global list_vector_other
        print("Start.... handle_other_face")
        for i in range(len(list_vector_other)):
            face = list_vector_other[i]
            if("face_id" not in face):
                face["face_id"] = face_id_max 
                for face_compare in list_vector_other:
                    if(face_compare["id"] != face["id"]):
                        print("caculation ....",i)
                        cos = cosin(face["embedding"], face_compare["embedding"])
                        if(cos > weight_point):
                            face_compare["face_id"] = face_id_max
                face_id_max = face_id_max + 1
        list_face_not_check = []
        for i in range(len(list_vector_other)):
            if(checkOnArr(list_face_not_check, face["face_id"]) == False):
                face = list_vector_other[i]
                for face_compare in list_vector_other:
                    if(face_compare["face_id"] != face["face_id"]):
                        if(checkOnArr(list_face_not_check, face_compare["face_id"]) == False):
                            cos = cosin(face["embedding"], face_compare["embedding"])
                            print("caculation2 ....",i)
                            if(cos > weight_point):
                                list_face_not_check.append(face_compare["face_id"])
                                for face_change in list_vector_other:
                                    if(face_change["face_id"] == face_compare["face_id"]):
                                        face_change["face_id"] = face["face_id"]
        # compare to main  again     
        # for face in list_vector_other:
        #     flag = True 
        #     for face_main in list_vector_widden:
        #         if(flag == True ):
        #             cos = cosin(face["embedding"], face_main["embedding"])
        #             if(cos > weight_point):
        #                 flag = False
        #                 face["face_id"] = 0

        for face in list_vector_other:
            print("insert....")
            del(face["embedding"])
            facematches.insert_one(face)
    except Exception as e:
        print("handle_other_face.....", e)
    
def handle_main(case_id, tracking_folder, target_folder):
    try:
        global list_vector

        if not os.path.exists(f"./video_apperance"):
            os.makedirs(f"./video_apperance")
        if not os.path.exists(f"./video_apperance/{case_id}"):
            os.makedirs(f"./video_apperance/{case_id}")

        flag_target_folder = True
        for path in os.listdir(target_folder):
            if(flag_target_folder == True):
                if os.path.isfile(os.path.join(target_folder, path)):
                    full_path = f"{target_folder}/{path}"
                    img = cv2.imread(full_path)
                    print("full_path",full_path)
                    faces = app_recognize.get(img)
                    for face in faces:
                        embedding_vector = face['embedding']
                        list_vector.append(embedding_vector)
                        # check_insert_target = index.query(
                        #     vector=embedding_vector.tolist(),
                        #     top_k=1,
                        #     include_metadata=True,
                        #     include_values=True,
                        #     filter={"case_id": case_id},
                        # )
                        # matches = check_insert_target["matches"]
                        # if(len(matches) > 0):
                        #     if(matches[0]["metadata"]["case_id"] == case_id):
                        #        flag_target_folder = False
                        # if(flag_target_folder == True):
                        #     index.upsert(
                        #         vectors=[
                        #                 {
                        #                     "id": str(uuid.uuid4()),
                        #                     "values": embedding_vector,
                        #                     "metadata": {"case_id":case_id }
                        #                 },
                        #             ]
                        #     )
        list_file = []
        for path in os.listdir(tracking_folder):
            if os.path.isfile(os.path.join(tracking_folder, path)):
                full_path = f"{tracking_folder}/{path}"
                list_file.append(full_path)
        if(len(list_file) > 0):
            handle_multiplefile(list_file,80,case_id)
            cases.update_many({
                "id":case_id
            },{
                "$set":{
                    "end":current_date(),
                    "status":"grouping"
                }
            })
            handle_other_face()
        return 
    except Exception as e:
        print("error handle_main",e)
        cases.update_many({
            "id":case_id
        },{
            "$set":{
                "end":current_date(),
                "status":"completed"
            }
        })
    


api = Flask(__name__)
@api.route('/analyst', methods=["POST"])
def analyst():
    case_id = request.json['case_id']
    tracking_folder = request.json['tracking_folder']
    target_folder = request.json['target_folder']
    
    myquery = { "case_id": case_id }
    facematches.delete_many(myquery)
    appearances.delete_many(myquery)
    videos.delete_many(myquery)
    
    cases.update_many({
        "id":case_id
    },{
        "$set":{
            "start":current_date(),
            "status":"processing"
        }
    })
    
    global list_vector 
    global list_vector_other
    list_vector_other = []
    list_vector  = []


    
    subprocess.run("cd /home/poc4a5000/detect/detect && rm -rf datas && mkdir datas && rm -rf final_result && mkdir final_result && rm -rf outputs && mkdir outputs && rm -rf results && mkdir results && rm -rf final_result && mkdir final_result && rm -rf videos && mkdir videos && rm -rf faces && mkdir faces && rm -rf video_apperance && mkdir video_apperance", shell=True, check=True)
    
    handle_main(case_id,tracking_folder,target_folder)

    cases.update_many({
        "id":case_id
    },{
        "$set":{
            # "end":current_date(),
            "status":"completed"
        }
    })

    return jsonify({
        "data":"ok"
    })

if __name__ == '__main__':
    api.run(debug=True, port=5234, host='0.0.0.0')






# Run with  GPU
# dir_path = r'/home/poc4a5000/facesx '
# list_file = []
# for path in os.listdir(dir_path):
#     # check if current path is a file
#     if os.path.isfile(os.path.join(dir_path, path)):
#         full_path = f"{dir_path}/{path}"
#         list_file.append(full_path)
# # print(list_file)


# start_time = time.time()
# print("Start ......",str(start_time))
# f = open("start.txt", "a")
# f.write(str(start_time))

# handle_multiplefile(list_file[6:],50 )
# handle_multiplefile(list_file,50)
# ch02_20240904040117.mp4
# handle_multiplefile(["input/video8p.mp4"],50)
# handle_main("123456-12", "/home/poc4a5000/detect/detect/example/tracking_folder", "/home/poc4a5000/detect/detect/example/target_folder")
# end_time = time.time()
# f = open("end.txt", "a")
# f.write(str(end_time))

# print(f"Total execution time: {end_time - start_time}")
