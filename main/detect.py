
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

# Define providers with device_id
torch.cuda.set_device(0)
providers = [
    ('CUDAExecutionProvider', {
        'device_id': 0,
    })
]
weight_point = 0.65
time_per_frame_global = 2
ctx_id = 0 if device.type == 'cuda' else  -1
app_recognize = FaceAnalysis('buffalo_l',providers=['CPUExecutionProvider'])
app_recognize.prepare(ctx_id=0, det_thresh=0.1, det_size=(640, 640))

app_recognize2 = FaceAnalysis('buffalo_l',providers=['CPUExecutionProvider'])
app_recognize2.prepare(ctx_id=0, det_thresh=0.3, det_size=(640, 640))

app_recognize3 = FaceAnalysis('buffalo_l',providers=['CPUExecutionProvider'])
app_recognize3.prepare(ctx_id=0, det_thresh=0.5, det_size=(640, 640))

num_gpus = torch.cuda.device_count()
print(f"Number of GPUs available: {num_gpus}")
gpu_ids = list(range(num_gpus)) 



list_vector = []

list_vector_other = []

list_vector_widden = []


def current_date():
  format_date = "%Y-%m-%d %H:%M:%S"
  now = datetime.datetime.now()
  date_string = now.strftime(format_date)
  return datetime.datetime.strptime(date_string, format_date)

def callworker(link, case_id, file):
    try:
        url = link
        payload = json.dumps({
            "case_id": case_id,
            "tracking_file": file
        })
        headers = {
        'Content-Type': 'application/json'
        }
        requests.request("POST", url, headers=headers, data=payload)
    except Exception as e:
        print("error call worker ....", e)

def handle_multiplefile(listfile,case_id):
    try:
        print("listfile....",listfile)
        threads = []
        count = 0 
        for file in listfile:
            print("start for ...")
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
            port = 6000 + count 
            count = count + 1 
            link = f"http://192.168.50.10:{port}/analyst/ele"
            print("Call api")
            t = threading.Thread(target=callworker, args=(link, case_id, file))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        
        return 
    except Exception as e:
        print("handle_multiplefile ....", e)

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
            handle_multiplefile(list_file,case_id)
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




