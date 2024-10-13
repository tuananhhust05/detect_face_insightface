
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
time_per_frame_global = 1
gpu_id_global = 1
port = 6000 + gpu_id_global

app_recognize = FaceAnalysis('buffalo_l',providers=['CPUExecutionProvider'])
app_recognize.prepare(ctx_id=gpu_id_global, det_thresh=0.1, det_size=(640, 640))

app_recognize2 = FaceAnalysis('buffalo_l',providers=['CPUExecutionProvider'])
app_recognize2.prepare(ctx_id=gpu_id_global, det_thresh=0.3, det_size=(640, 640))

app_recognize3 = FaceAnalysis('buffalo_l',providers=['CPUExecutionProvider'])
app_recognize3.prepare(ctx_id=gpu_id_global, det_thresh=0.5, det_size=(640, 640))




# Define providers with device_id
torch.cuda.set_device(0)
providers = [
    ('CUDAExecutionProvider', {
        'device_id': gpu_id_global,
    })
]

model = model_zoo.get_model(                                 # Load the model with  providers
    '/home/poc4a5000/.insightface/models/buffalo_l/det_10g.onnx',
    providers=providers
)
model.prepare(ctx_id=gpu_id_global, det_size=(640, 640))

model_analyst = FaceAnalysis('buffalo_l',providers=providers)
model_analyst.prepare(ctx_id=gpu_id_global,det_thresh=0.3, det_size=(640, 640))


list_vector = []

list_vector_other = []

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
    fps = cap2.get(cv2.CAP_PROP_FPS)
    fps = ( fps + 1 ) // 1
    frame_rate = time_per_frame_global * fps 
    sum_age = 0 
    sum_gender = 0 
    count_face = 0 
    while True:
        ret, frame = cap2.read()
        if not ret:
            break
        frame_count += 1
        print("frame_count", frame_count)
        if frame_count % frame_rate == 0:
            try:
                faces = model_analyst.get(frame)
                flag_loop = False
                for face in faces:
                    if(flag_loop == True):
                        break
                    if face["det_score"] > 0.7:
                        similarity  = checkface(face['embedding'].tolist())
                        print("similarity.....",similarity)

                        if(similarity > 0):
                            flag_loop = True
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

                            # insert elasticsearch 
                            insert_document(str(uuid.uuid4()), face['embedding'])
                           
            except Exception as e:
                print("error recognizing ",e)


          
             

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
    

    cap2.release()
    
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

def create_video_apperance(case_id,thread_count,folder, extension):
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
        print("Creating video....",i)
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
        "file":f"{folder}{extension}"
    })

    return 

def cutvideo(videofile,start,duration,output,stt):
    flag = True 
    stt_handle = stt 
    while(flag == True):
        try:
            gpu_id = gpu_id_global
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
       faces = model_analyst.get(img)
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
            gpu_id = gpu_id_global
            t = threading.Thread(target=extract_frames, args=(folder,video_file,i,time_per_segment,case_id,gpu_id,file_extension))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        groupJson(folder,video_file_origin,count_thread,case_id, file_extension)
        # create_video_apperance(case_id,count_thread,folder,file_extension, svideo_file_origin)
        create_video_apperance(case_id,count_thread,folder,file_extension)
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


def insert_document(doc_id, vector):
    try:
        doc = {
            "title": "image",
            "content_vector": vector
        }
        es.index(index=index_name, id=doc_id, body=doc)
    except Exception as e:
        print("insert_document",e)

def handle_main(case_id, tracking_file):
    try:
        # global list_vector
        if not os.path.exists(f"{dir_project}/video_apperance"):
            os.makedirs(f"{dir_project}/video_apperance")
        if not os.path.exists(f"{dir_project}/video_apperance/{case_id}"):
            os.makedirs(f"{dir_project}/video_apperance/{case_id}")
        handle_multiplefile([tracking_file],15,case_id)
        return 
    except Exception as e:
        print("error handle_main",e)


api = Flask(__name__)
@api.route('/analyst/ele', methods=["POST"])
def analyst():
    try:
        case_id = request.json['case_id']
        tracking_file = request.json['tracking_file']
        print("case_id", case_id)
        handle_main(case_id, tracking_file)
        return jsonify({
            "data":"ok"
        })
    except Exception as e:
        print("error handle_main",e)
        return jsonify({
            "data":"error ....."
        })
if __name__ == '__main__':
    api.run(debug=True, port=port, host='0.0.0.0')




