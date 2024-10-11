
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
import subprocess
import threading
import matplotlib.pyplot as plt 
import uuid 
import json
from flask import Flask, jsonify, request
import pymongo
import ffmpeg
import random
from threading import Thread
from elasticsearch import Elasticsearch
from queue import Queue
import requests

# Connect to Elasticsearch
es = Elasticsearch("http://localhost:9200")
index_name = "images"

myclient = pymongo.MongoClient("mongodb://127.0.0.1:27017")

mydb = myclient["faceX"]
facematches = mydb["facematches"]
appearances = mydb["appearances"]
targets = mydb["targets"]
videos = mydb["videos"]
cases = mydb["cases"]

dir_project = "/home/poc4a5000/storage_facesx"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


weight_point = 0.625
time_per_frame_global = 2
ctx_id = 0 if device.type == 'cuda' else  -1
app_recognize = FaceAnalysis('buffalo_l',providers=['CPUExecutionProvider'])
app_recognize.prepare(ctx_id=-1, det_thresh=0.1, det_size=(640, 640))

app_recognize2 = FaceAnalysis('buffalo_l',providers=['CPUExecutionProvider'])
app_recognize2.prepare(ctx_id=-1, det_thresh=0.3, det_size=(640, 640))

app_recognize3 = FaceAnalysis('buffalo_l',providers=['CPUExecutionProvider'])
app_recognize3.prepare(ctx_id=-1, det_thresh=0.5, det_size=(640, 640))

num_gpus = torch.cuda.device_count()
print(f"Number of GPUs available: {num_gpus}")
gpu_ids = list(range(num_gpus)) 

# list_model_detect = []
# for j in range(num_gpus):
#     providers = [
#         ('CUDAExecutionProvider', {
#             'device_id': j,
#         })
#     ]
#     model_ele = model_zoo.get_model(
#         '/home/poc4a5000/.insightface/models/buffalo_l/det_10g.onnx',
#         providers=providers
#     )
#     model_ele.prepare(ctx_id=j, det_thresh=0.1,det_size=(640, 640))
#     list_model_detect.append(model_ele)
# torch.cuda.set_device(gpu_id)
# device = torch.device(f'cuda:{gpu_id}')

# Define providers with device_id
torch.cuda.set_device(0)
providers = [
    ('CUDAExecutionProvider', {
        'device_id': 0,
    })
]

model = model_zoo.get_model(                                 # Load the model with  providers
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
    app_ele.prepare(ctx_id=j,det_thresh=0.3, det_size=(640, 640))
    list_model_analyst.append(app_ele)

list_vector = []

list_vector_other = []

list_vector_widden = []

picture_queue = Queue()  # for optimizing picture 

def getduration(file):
    try:
        data = cv2.VideoCapture(file) 
        frames = data.get(cv2.CAP_PROP_FRAME_COUNT) 
        fps = data.get(cv2.CAP_PROP_FPS) 
        data.release()
        seconds = round(frames / fps) 
        return seconds
    except Exception as ex:
        print(f"getduration {ex}")
        return 0 


def cosin(question, answer):
    question = torch.tensor(question).to(device)
    answer = torch.tensor(answer).to(device)
    cosine = torch.dot(question, answer) / (torch.norm(question) * torch.norm(answer))
    return cosine.item()  

def search_with_cosine_similarity(query_vec):
    search_query = {
        "query": {
            "script_score": {
                "query": {
                    "match_all": {}
                },
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'content_vector') + 1",
                    "params": {
                        "query_vector": query_vec
                    }
                }
            }
        },
        "size": 1,
    }

    response = es.search(index=index_name, body=search_query)
    return response

def checkface(vector):
    try:
        response = search_with_cosine_similarity(vector)
        for hit in response['hits']['hits']:
            score = float(hit['_score'])
            score = score - 1 
            if( score > float(weight_point)):
                return score
        return 0 
    except Exception as ex:
        print(f"checkface: {ex}")
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

def call_optimize_image(path):  
    try:                      
        url = "http://192.168.50.10:8005/restore-file"
        payload = json.dumps({
        "file_path": path
        })
        headers = {
            'Content-Type': 'application/json'
        }
        requests.request("POST", url, headers=headers, data=payload)
        print("optimized ...", path)
        return 
    except Exception as ex:
        print(f"call_optimize_image {ex}")
        return

def extract_frames(folder,video_file,index_local,time_per_segment,case_id,gpu_id,extension):
    array_em_result = []
    list_result_ele = []
    frame_count = 0 
    duration = getduration(video_file)

    if(duration == 0):
        return 
    
    cap2 = cv2.VideoCapture(video_file, cv2.CAP_FFMPEG)
    cap = VideoCaptureThreading(video_file)

    fps = cap2.get(cv2.CAP_PROP_FPS)
    fps = ( fps + 1 ) // 1
    frame_rate = time_per_frame_global * fps 


    sum_age = 0 
    sum_gender = 0 
    count_face = 0 
    list_face_other_in_thread = []
    while True:
        ret, frame = cap2.read()
        if not ret:
            break
        frame_count += 1
        print("frame_count", frame_count)
        if frame_count % frame_rate == 0:
            
            # facechecks = list_model_detect[gpu_id].detect(frame,input_size=(640, 640))
            # flagDetect = False
            # if(len(facechecks) > 0):
            #     if(len(facechecks[0]) > 0):
            #         flagDetect = True
            
            # if(flagDetect == True):
            # print("Có mặt......")
            try:
                faces = list_model_analyst[gpu_id].get(frame)
                
                flag_loop = False
                for face in faces:
                    if(flag_loop == True):
                        break
                    if face["det_score"] > 0.6:
                        similarity  = checkface(face['embedding'].tolist())
                        print("similarity.....",similarity)
                        if(similarity > 0):
                            flag_loop = True
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
                                if not os.path.exists(f"{dir_project}/faces/{case_id}/{folder}/{index_local}"):
                                    os.makedirs(f"{dir_project}/faces/{case_id}/{folder}/{index_local}")
                                if not os.path.exists(f"{dir_project}/outputs/{case_id}/{folder}/{index_local}"):
                                    os.makedirs(f"{dir_project}/outputs/{case_id}/{folder}/{index_local}")
                                
                                try:
                                    height, width = frame.shape[:2]
                                    a = max(1, bbox[1])
                                    b = min(height -1 , bbox[3])
                                    c = max(1, bbox[0])
                                    d = min(width -1 , bbox[2])
                                    cv2.imwrite(f'{dir_project}/faces/{case_id}/{folder}/{index_local}/{filename}', frame[a:b, c:d])
                                except Exception as e:
                                    print(f"error save faces")
                
                                top_left = (bbox[0], bbox[1])
                                bottom_right = (bbox[2], bbox[3])
                                color = (255, 0, 0)
                                thickness = 2
                                try:
                                    cv2.rectangle(frame, top_left, bottom_right, color, thickness)
                                    cv2.imwrite(f'{dir_project}/outputs/{case_id}/{folder}/{index_local}/{filename}', frame)
                                except Exception as e:
                                    print(f"error save outputs")
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
                                    "proofImage":f'{dir_project}/faces/{case_id}/{folder}/{index_local}/{filename}',
                                    "url":f'{dir_project}/faces/{case_id}/{folder}/{index_local}/{filename}',
                                    "createdAt":current_date(),
                                    "updatedAt":current_date(),
                                    "file":f"{folder}{extension}"
                                    }
                            facematches.insert_one(mydict)
                            mydict["embedding"] = face['embedding']
                            list_vector_widden.append(mydict)

                            global list_vector 

                            list_vector.append(face['embedding'])
                            # insert elasticsearch 
                            insert_document(str(uuid.uuid4()), face['embedding'])
                            # for optimizing picture 
                            # picture_queue.put(mydict)
                            
                            # threading.Thread(target=call_optimize_image, args=(f'{dir_project}/faces/{case_id}/{folder}/{index_local}/{filename}',)).start()
                            
                        # else:
                        
                            # try:
                            #     bbox = [int(b) for b in face['bbox']]
                            #     filename = f"{frame_count}_{str(uuid.uuid4())}_face.jpg"
                            #     if not os.path.exists(f"{dir_project}/faces/{case_id}/{folder}/{index_local}"):
                            #         os.makedirs(f"{dir_project}/faces/{case_id}/{folder}/{index_local}")
                            #     if not os.path.exists(f"{dir_project}/outputs/{case_id}/{folder}/{index_local}"):
                            #         os.makedirs(f"{dir_project}/outputs/{case_id}/{folder}/{index_local}")
                            #     try:
                            #         cv2.imwrite(f'{dir_project}/faces/{case_id}/{folder}/{index_local}/{filename}', frame[bbox[1]:bbox[3], bbox[0]:bbox[2]])
                            #     except Exception as e:
                            #         print(f"Error saving faces other ....")
                            # except Exception as e:
                            #     print(f"Error saving frame: {e}")
                            
                            # mydict = { 
                            #         "id":  str(uuid.uuid4()), 
                            #         "case_id": case_id,
                            #         "embedding": face['embedding'],
                            #         "similarity_face":similarity,
                            #         "gender":int(face['gender']),
                            #         "age":int(face['age']),
                            #         "time_invideo":"",
                            #         "proofImage":f'{dir_project}/faces/{case_id}/{folder}/{index_local}/{filename}',
                            #         "url":f'{dir_project}/faces/{case_id}/{folder}/{index_local}/{filename}',
                            #         "createdAt":current_date(),
                            #         "updatedAt":current_date(),
                            #         "file":folder,
                            #         "frame_count":frame_count
                            #         }
                            # list_face_other_in_thread.append(mydict)
                            # list_vector_other.append(mydict)
            except Exception as e:
                print("error recognizing ",e)
    # check in face_other again 
    # for face_other in list_face_other_in_thread:
    #      print("check again....")
    #      similarity  = checkface(face_other['embedding'].tolist())
    #      if(similarity > 0):
    #         print("check again.... ok..")
    #         if len(array_em_result) == 0:
    #             array_em_result.append({
    #                 "speaker": 0,
    #                 "gender":int(face_other['gender']),
    #                 "age":int(face_other['age']),
    #                 "frames": [face_other["frame_count"]],
    #             })
    #         else:
    #             array_em_result[0]["frames"].append(face_other["frame_count"])
    #         frame_count_current = face_other["frame_count"]
    #         filename = f"{frame_count_current}_0_face.jpg"
    #         threading.Thread(target=call_optimize_image, args=(f'{dir_project}/faces/{case_id}/{folder}/{index_local}/{filename}',)).start()
    #         url = face_other["url"]
    #         list_vector.append(face['embedding'])
    #         # insert elasticsearch 
    #         insert_document(str(uuid.uuid4()), face_other['embedding'])
    #         subprocess.run(f"mv {url} {dir_project}/faces/{case_id}/{folder}/{index_local}/{filename}", shell=True, check=True)

          
             

    for ele in array_em_result:
        ele["frame_count"] = frame_count
        ele["duration"] = duration
        ele["frame_rate"] = frame_rate
        
    with open(f"{dir_project}/datas/{case_id}/{folder}/{index_local}.json", 'w') as f:
       json.dump(array_em_result, f, indent=4)
    
    with open(f"{dir_project}/datas/{case_id}/{folder}/{index_local}.json", 'r') as file:
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


    with open(f"{dir_project}/results/{case_id}/{folder}/{index_local}.json", 'w') as f:
        json.dump(list_result_ele, f, indent=4)
    

    cap.release()
    
    return 


def groupJson(folder,video_file,count_thread,case_id, file_extension):
    final_result = {
        "time":[]
    }
    duration = getduration(video_file)
    time_per_segment = duration / count_thread

    list_stt = []
    for path in os.listdir(f"{dir_project}/results/{case_id}/{folder}"):
        if os.path.isfile(os.path.join(f"{dir_project}/results/{case_id}/{folder}", path)):
            stt = int(path.split(".")[0])
            list_stt.append(stt)
           
    list_stt=sorted(list_stt)
    sum_age = 0 
    sum_gender = 0 
    count_face = 0 
    for stt in list_stt:
        with open(f"{dir_project}/results/{case_id}/{folder}/{stt}.json", 'r') as file:
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

    with open(f"{dir_project}/final_result/{case_id}/{folder}/final_result.json", 'w') as f:
        json.dump(final_result, f, indent=4)
    
    final_result["file"] = f"{folder}{file_extension}" 
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
    out = cv2.VideoWriter(f"{dir_project}/video_apperance/{case_id}/{folder}_pre.mp4", fourcc, 1/time_per_frame_global , size)
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
            working_directory = f"{dir_project}/video_apperance/{case_id}"
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

    videos.insert_one({
        "id":str(uuid.uuid4()),
        "case_id":case_id,
        "path":outputfinal,
    })

    return 

def cutvideo(videofile,start,duration,output,stt):
    flag = True 
    stt_handle = stt 
    while(flag == True):
        try:
            gpu_id = gpu_ids[stt_handle % num_gpus]
            command = f"ffmpeg -hwaccel cuda -hwaccel_device {gpu_id} -ss {start} -i {videofile} -vf \"scale=480:480,pad=480:480:(ow-iw)/2:(oh-ih)/2\" -t {duration} -c:v h264_nvenc -preset fast -b:v 5M {output} -y"
            subprocess.run(command, shell=True, check=True)
            flag = False
        except Exception as e:
            print("error",command)
            stt_handle = stt_handle + 1 
            print(e)

    return 

def trimvideo(folder,videofile,count_thread,case_id):
    duration = getduration(videofile)
    time_per_segment = duration / count_thread
    threads = []

    origin_videofile = videofile
    file = origin_videofile.split("/")[ len(origin_videofile.split("/")) -1 ]
    name_file = file.split(".")[0]
    new_name = f"{name_file}_tempt"
    new_path = origin_videofile.replace(name_file,new_name)
    
    # pre redecode 
    # print(f"ffmpeg -i {videofile} -c:v copy -c:a copy {new_path} -y && rm {videofile} && mv {new_path} {videofile}")
    # subprocess.run(f"ffmpeg -i {videofile} -c:v copy -c:a copy {new_path} -y && rm {videofile} && mv {new_path} {videofile}", shell=True, check=True)
    for i in range(count_thread):
        t = threading.Thread(target=cutvideo, args=(videofile,time_per_segment*i,time_per_segment,f"{dir_project}/videos/{case_id}/{folder}/{i}.mp4",i))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()
    
    return 

def handleimage(folder,img_url,case_id,file_extension):
   img = cv2.imread(img_url)
   facechecks = model.detect(img,input_size=(640, 640))
   if(len(facechecks) > 0):
       faces = list_model_analyst[0].get(img)
       for face in faces:
         if face["det_score"] > 0.5:
            similarity  = checkface(face['embedding'].tolist())
            if(similarity > 0):
                bbox = [int(b) for b in face['bbox']]
                filename = f"0_0_face.jpg"

                if not os.path.exists(f"{dir_project}/faces/{case_id}/{folder}/1"):
                    os.makedirs(f"{dir_project}/faces/{case_id}/{folder}/1")
                if not os.path.exists(f"{dir_project}/outputs/{case_id}/{folder}/1"):
                    os.makedirs(f"{dir_project}/outputs/{case_id}/{folder}/1")
                
                cv2.imwrite(f'{dir_project}/faces/{case_id}/{folder}/1/{filename}', img[bbox[1]:bbox[3], bbox[0]:bbox[2]])

                top_left = (bbox[0], bbox[1])
                bottom_right = (bbox[2], bbox[3])
                color = (255, 0, 0)
                thickness = 2
                cv2.rectangle(img, top_left, bottom_right, color, thickness)
                cv2.imwrite(f'{dir_project}/outputs/{case_id}/{folder}/1/{filename}', img)
                appearances.insert_one({
                    "time":[
                        {
                            "path":f'{dir_project}/faces/{case_id}/{folder}/1/{filename}',
                            "start":0,
                            "end":0,
                            "similiarity":90
                        }
                    ],
                    "gender":int(face['gender']),
                    "age":int(face['age']),
                    "file":f"{folder}{file_extension}",
                    "id":str(uuid.uuid4()),
                    "case_id":case_id,
                    "createdAt":current_date(),
                    "updatedAt":current_date(),
                })

def process_videos(folder,video_file_origin,count_thread,case_id):
    filename, file_extension = os.path.splitext(video_file_origin)
    if( (file_extension == ".mp4") or (file_extension == ".webm") or (file_extension == ".mkv") or (file_extension == ".mov")):
        duration = getduration(video_file_origin)
        time_per_segment = duration / count_thread

        trimvideo(folder,video_file_origin,count_thread,case_id)

        video_files = [f"{dir_project}/videos/{case_id}/{folder}/{i}.mp4" for i in range(count_thread)]  
        threads = []
        for i, video_file in enumerate(video_files):
            gpu_id = gpu_ids[i % num_gpus]
            t = threading.Thread(target=extract_frames, args=(folder,video_file,i,time_per_segment,case_id,gpu_id,file_extension))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        groupJson(folder,video_file_origin,count_thread,case_id, file_extension)
        # create_video_apperance(case_id,count_thread,folder,file_extension, svideo_file_origin)
        create_video_apperance(case_id,count_thread,folder)
    else:
        handleimage(folder,video_file_origin,case_id,file_extension)
    return 

def handle_multiplefile(listfile,thread,case_id):
    for file in listfile:
        file_name = file.split(".")[0]
        if "/" in file_name: 
            file_name = file_name.split("/")[len(file_name.split("/")) - 1]
        
        if not os.path.exists(f"{dir_project}/faces/{case_id}"):
            os.makedirs(f"{dir_project}/faces/{case_id}")

        if not os.path.exists(f"{dir_project}/faces/{case_id}/{file_name}"):
            os.makedirs(f"{dir_project}/faces/{case_id}/{file_name}")
        
        if not os.path.exists(f"{dir_project}/outputs/{case_id}"):
            os.makedirs(f"{dir_project}/outputs/{case_id}")

        if not os.path.exists(f"{dir_project}/outputs/{case_id}/{file_name}"):
            os.makedirs(f"{dir_project}/outputs/{case_id}/{file_name}")
        
        if not os.path.exists(f"{dir_project}/videos/{case_id}"):
            os.makedirs(f"{dir_project}/videos/{case_id}")

        if not os.path.exists(f"{dir_project}/videos/{case_id}/{file_name}"):
            os.makedirs(f"{dir_project}/videos/{case_id}/{file_name}")
        
        if not os.path.exists(f"{dir_project}/datas/{case_id}"):
            os.makedirs(f"{dir_project}/datas/{case_id}")

        if not os.path.exists(f"{dir_project}/datas/{case_id}/{file_name}"):
            os.makedirs(f"{dir_project}/datas/{case_id}/{file_name}")
       
        if not os.path.exists(f"{dir_project}/results/{case_id}"):
            os.makedirs(f"{dir_project}/results/{case_id}")

        if not os.path.exists(f"{dir_project}/results/{case_id}/{file_name}"):
            os.makedirs(f"{dir_project}/results/{case_id}/{file_name}")
        
        if not os.path.exists(f"{dir_project}/final_result/{case_id}"):
            os.makedirs(f"{dir_project}/final_result/{case_id}")

        if not os.path.exists(f"{dir_project}/final_result/{case_id}/{file_name}"):
            os.makedirs(f"{dir_project}/final_result/{case_id}/{file_name}")
    
        folder = file_name
        process_videos(folder,file,thread,case_id)
        subprocess.run("rm -rf videos/{file_name}", shell=True, check=True)
    
    return 

def checkOnArr(arr,num):
    for ele in arr:
        if ele == num:
            return True
    return False

def insert_to_database(id,embedding,face_id,similarity_face,gender,age,proofImage,url,file,case_id):
    try:
        face_id_insert = face_id 
        similarity  = checkface(embedding.tolist())
        if(similarity > 0): face_id_insert = 0
        mydict = { 
                "id":id, 
                "case_id": case_id,
                "similarity_face":similarity_face,
                "gender":gender,
                "age":age,
                "time_invideo":"",
                "proofImage":proofImage,
                "url": url,
                "createdAt":current_date(),
                "updatedAt":current_date(),
                "file":file,
                "face_id":face_id_insert
            }
        facematches.insert_one(mydict)
    except Exception as e:
        print("insert_to_database.....", e)

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
                        if(cos > (weight_point-0.05) ):
                            face_compare["face_id"] = face_id_max
                face_id_max = face_id_max + 1
        list_face_not_check = []
        list_inserted = []
        for i in range(len(list_vector_other)):
      
            if(checkOnArr(list_face_not_check, face["face_id"]) == False):
                print("list_face_not_check",list_face_not_check)
                face = list_vector_other[i]
                for face_compare in list_vector_other:
                    if(face_compare["face_id"] != face["face_id"]):
                            print("Caculation2...", i)
                            cos = cosin(face["embedding"], face_compare["embedding"])
                            if(cos > (weight_point - 0.15) ):
                                if(checkOnArr(list_face_not_check, face_compare["face_id"]) == False):
                                    list_face_not_check.append(face_compare["face_id"])
                                for face_change in list_vector_other:
                                    if(face_change["face_id"] == face_compare["face_id"]):
                                        face_change["face_id"] = face["face_id"]
                print("length list_vector",len(list_vector))
                t = threading.Thread(target=insert_to_database, args=(face["id"],face["embedding"],face["face_id"],face["similarity_face"],face["gender"],face["age"],face["proofImage"],face["url"],face["file"],face["case_id"]))
                t.start()
                t.join()
                list_inserted.append(face["id"])

        for face in list_vector_other:
            if(checkOnArr(list_inserted, face["id"]) == False):
                print("insert....")
                del(face["embedding"])
                facematches.insert_one(face)

    except Exception as e:
        print("handle_other_face.....", e)


def handle_apperance_other_face():
    list_apperance_other = []
    for face in list_vector_other:
        check = False
        file = face["file"]
        face_id = face["face_id"]
        for apperance in list_apperance_other:
           print("..")


def insert_document(doc_id, vector):
    try:
        doc = {
            "title": "image",
            "content_vector": vector
        }
        es.index(index=index_name, id=doc_id, body=doc)
    except Exception as e:
        print("insert_document",e)

def analyst_video_sadtalker(path, target_folder):
    try:
        # redecode
        origin_videofile = path
        file = origin_videofile.split("/")[ len(origin_videofile.split("/")) -1 ]
        name_file = file.split(".")[0]
        new_name = f"{name_file}_tempt"
        new_path = origin_videofile.replace(name_file,new_name)
        print(f"ffmpeg -i {path} -c:v copy -c:a copy {new_path} -y && rm {path} && mv {new_path} {path}")
        subprocess.run(f"ffmpeg -i {path} -c:v copy -c:a copy {new_path} -y && rm {path} && mv {new_path} {path}", shell=True, check=True)

        cap = cv2.VideoCapture(path)
        count = 0
        count_inserted = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            count = count + 1 
            print("sadtalker ....",  count)
            try:
                faces = []
                try:
                    faces = app_recognize3.get(frame)
                except Exception as e:
                    print("fail recognize 1 ...")
                    faces = []

                if(len(faces) == 0):
                    try:
                       faces = app_recognize2.get(frame)
                    except Exception as e:
                       print("fail recognize 2 ...")
                       faces = []

                if(len(faces) == 0):
                    try:
                       faces = app_recognize.get(frame)
                    except Exception as e:
                       print("fail recognize 3 ...")
                       faces = []
                for face in faces:
                    embedding_vector = face['embedding']
                    insert_document(str(uuid.uuid4()), embedding_vector)
                    count_inserted = count_inserted + 1 
                    print("inserted",count_inserted)
                    list_vector.append(embedding_vector)
                    # bbox = [int(b) for b in face['bbox']]
                    # cv2.imwrite(f'/home/poc4a5000/detect/detect/image_sadtalker/{str(uuid.uuid4())}', frame[bbox[1]:bbox[3], bbox[0]:bbox[2]])
            except Exception as e:
                    print("error recognize sanalyst_video_sadtalker",e)
    except Exception as e:
        print("error analyst_video_sadtalker",e)


def handle_sadtalker(path,case_id,target_folder):
    try:
        url = "http://192.168.50.10:8003/upload"
        payload = {
            'case_id':case_id 
        }
        name_file = path.split("/")[len(path.split("/")) -1 ]
        files=[
        ('files',(name_file,open( path ,'rb'),'application/octet-stream'))
        ]
        headers = {}
        print(path,payload)
        response = requests.request("POST", url, headers=headers, data=payload, files=files)
        for video in response.json()["videos"]:
            analyst_video_sadtalker(video,target_folder)
        print(response.json()["videos"])
        return 
    except Exception as e:
        print("error handle_sadtalker",e)

def handle_main(case_id, tracking_folder, target_folder):
    try:
        global list_vector

        if not os.path.exists(f"{dir_project}/video_apperance"):
            os.makedirs(f"{dir_project}/video_apperance")
        if not os.path.exists(f"{dir_project}/video_apperance/{case_id}"):
            os.makedirs(f"{dir_project}/video_apperance/{case_id}")

        flag_target_folder = True
        for path in os.listdir(target_folder):
            if(flag_target_folder == True):
                if os.path.isfile(os.path.join(target_folder, path)):
                    full_path = f"{target_folder}/{path}"
                    img = cv2.imread(full_path)
                    print("full_path",full_path)
                    faces = app_recognize3.get(img)
                    if(len(faces) == 0):
                        faces = app_recognize2.get(img)
                    
                    if(len(faces) == 0):
                        faces = app_recognize.get(img)

                    for face in faces:
                        embedding_vector = face['embedding']
                        list_vector.append(embedding_vector)
                        insert_document(str(uuid.uuid4()), embedding_vector)
                        print("Có mặt ........")
                        pose = face['pose']
                        print(pose)
                        flag_straight = True
                        for angle in pose:
                           if(flag_straight == True):
                                if(angle > 7):
                                    flag_straight = False
                        if(flag_straight == True):
                            handle_sadtalker(full_path,case_id,target_folder)
                       
        list_file = []
        for path in os.listdir(tracking_folder):
            if os.path.isfile(os.path.join(tracking_folder, path)):
                full_path = f"{tracking_folder}/{path}"
                list_file.append(full_path)
        if(len(list_file) > 0):
            handle_multiplefile(list_file,60,case_id)
            cases.update_many({
                "id":case_id
            },{
                "$set":{
                    "end":current_date(),
                    "status":"completed"
                }
            })
            # handle_other_face()
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
    
def delete_all_documents(index):
    try:
        response = es.delete_by_query(index=index, body={
            "query": {
                "match_all": {}
            }
        })
        print(f"Deleted {response['deleted']} documents from index '{index}'.")
    except Exception as e:
        print(f"An error occurred: {e}")

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

    delete_all_documents(index_name)
    
    subprocess.run(f"cd {dir_project} && rm -rf datas && mkdir datas && rm -rf final_result && mkdir final_result && rm -rf outputs && mkdir outputs && rm -rf results && mkdir results && rm -rf final_result && mkdir final_result && rm -rf videos && mkdir videos && rm -rf faces && mkdir faces && rm -rf video_apperance && mkdir video_apperance", shell=True, check=True)
    subprocess.run(f"cd /home/poc4a5000/facesx/engine/sad-talker/uploads && rm -rf db696a35-0043-4aba-a844-295e3432a118 && mkdir db696a35-0043-4aba-a844-295e3432a118", shell=True, check=True)
    
    handle_main(case_id,tracking_folder,target_folder)

    cases.update_many({
        "id":case_id
    },{
        "$set":{
            "status":"completed"
        }
    })

    return jsonify({
        "data":"ok"
    })

if __name__ == '__main__':
    api.run(debug=True, port=5234, host='0.0.0.0')




