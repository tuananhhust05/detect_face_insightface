
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

myclient = pymongo.MongoClient("mongodb://root:facex@192.168.50.10:27018")

mydb = myclient["faceX"]
facematches = mydb["facematches"]
appearances = mydb["appearances"]
targets = mydb["targets"]
videos = mydb["videos"]

dir_project = "/home/poc4a5000/detect/detect"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


pc = Pinecone(api_key="6bebb6ba-195f-471e-bb60-e0209bd5c697")
index = pc.Index("detectcamera")

weight_point = 0.4
time_per_frame_global = 2 
ctx_id = 0 if device.type == 'cuda' else -1
app = FaceAnalysis('buffalo_l',providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=ctx_id, det_size=(640, 640))
app_recognize = FaceAnalysis('buffalo_l',providers=['CUDAExecutionProvider'])
app_recognize.prepare(ctx_id=ctx_id, det_thresh=0.3, det_size=(640, 640))
model = model_zoo.get_model('/home/poc4a5000/.insightface/models/buffalo_l/det_10g.onnx')
model.prepare(ctx_id=ctx_id, det_size=(640, 640))

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

def current_date():
  format_date = "%Y-%m-%d %H:%M:%S"
  now = datetime.datetime.now()
  date_string = now.strftime(format_date)
  return datetime.datetime.strptime(date_string, format_date)

def extract_frames(folder,video_file,index_local,time_per_segment,case_id):
    array_em_result = []
    list_result_ele = []
    frame_count = 0 
    duration = getduration(video_file)
    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_rate = time_per_frame_global * fps 
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        print("frame_count", frame_count)
     
        

        if frame_count % frame_rate == 0:
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(frame)
            facechecks = model.detect(gpu_frame,input_size=(640, 640))
            flagDetect = False
            if(len(facechecks) > 0):
                if(len(facechecks[0]) > 0):
                    flagDetect = True
            
            # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            # faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)
            if(flagDetect == True):
                sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                sharpen = cv2.cuda.filter2D(gpu_frame, 0, sharpen_kernel)
                gpu_frame = cv2.cuda.fastNlMeansDenoisingColored(sharpen, None, 10, 10, 7, 21)
                faces = app.get(gpu_frame)

                sum_age = 0 
                sum_gender = 0 
                count_face = 0 
                for face in faces:
                    if face["det_score"] > 0.5:
                        embedding = torch.tensor(face['embedding']).to(device)  # Move embedding to GPU
                        search_result = index.query(
                            vector=embedding.tolist(),
                            top_k=1,
                            include_metadata=True,
                            include_values=True,
                            filter={"face": case_id},
                        )
                        matches = search_result["matches"]

                        if len(matches) > 0 and matches[0]['score'] > weight_point:
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
                                    array_em_result[0]["age"] = sum_age // count_face 
                                    array_em_result[0]["gender"] = sum_gender // count_face 
                                    array_em_result[0]["frames"].append(frame_count)

                            try:
                                bbox = [int(b) for b in face['bbox']]
                                filename = f"{frame_count}_0_face.jpg"
                                if not os.path.exists(f"./faces/{case_id}/{folder}/{index_local}"):
                                    os.makedirs(f"./faces/{case_id}/{folder}/{index_local}")
                                if not os.path.exists(f"./outputs/{case_id}/{folder}/{index_local}"):
                                    os.makedirs(f"./outputs/{case_id}/{folder}/{index_local}")

                                cv2.imwrite(f'./faces/{case_id}/{folder}/{index_local}/{filename}', frame[bbox[1]:bbox[3], bbox[0]:bbox[2], ::-1])

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
                                cv2.imwrite(f'./outputs/{case_id}/{folder}/{index_local}/{filename}', frame)
                            except Exception as e:
                                print(f"Error saving frame: {e}")
                            

                            mydict = { 
                                       "id":  str(uuid.uuid4()), 
                                       "case_id": case_id,
                                       "similarity_face":str(matches[0]['score']),
                                       "gender":int(face['gender']),
                                       "age":int(face['age']),
                                       "time_invideo":text,
                                       "proofImage":f'/home/poc4a5000/detect/detect/faces/{case_id}/{folder}/{index_local}/{filename}',
                                       "url":f'/home/poc4a5000/detect/detect/faces/{case_id}/{folder}/{index_local}/{filename}',
                                       "createdAt":current_date(),
                                       "updatedAt":current_date(),
                                       "file":folder
                                    }
                            facematches.insert_one(mydict)

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
                    list_time_exist.append([duration_exist[0]*time_per_frame,duration_exist[len(duration_exist) - 1] * time_per_frame])
                    duration_exist = []
                else:
                        if( i == len(list_frame)-2):
                            duration_exist.append(list_frame[i+1])
                            list_time_exist.append([duration_exist[0]*time_per_frame,duration_exist[len(duration_exist) - 1] * time_per_frame])
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
    max_age = 0 
    sum_gender = 0 
    count_face = 0 
    for stt in list_stt:
        with open(f"results/{case_id}/{folder}/{stt}.json", 'r') as file:
           data = json.load(file)
           if(len(data) > 0):
                data = data[0]
                if( int(data['age']) > max_age ):
                    max_age = int(data['age'])
                sum_gender = sum_gender + int(data['gender'])
                count_face = count_face + 1 
                for duration in data["duration_exist"]:
                    final_result["time"].append([duration[0] + stt * time_per_segment,duration[1] + stt * time_per_segment])

    final_result['age'] = max_age
    if count_face > 0 : 
        final_result['gender'] = sum_gender/ count_face
    
        facematches.update_many(
            {
                "case_id":case_id
            },
            {
                "$set":{
                    "gender":sum_gender/ count_face,
                    "age": max_age,
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
               "start":time[0],
               "end":time[1],
               "frame": (time[1] - time[0]) // time_per_frame
           }
       )
    final_result["time"] = new_arr
    appearances.insert_one(final_result)

def create_video_apperance(case_id,thread_count):
    list_img = []
    list_dir_file = os.listdir(f"{dir_project}/outputs/{case_id}")
    for dir in list_dir_file:
        dir_full = f"{dir_project}/outputs/{case_id}/{dir}"
        for i in range(thread_count):
            folder_count = i 
            dir_full_new = f"{dir_full}/{folder_count}"
            print(dir_full)
            if  os.path.exists(dir_full_new):
                for path in os.listdir(dir_full_new):
                    if os.path.isfile(os.path.join(dir_full_new, path)):
                        full_path = f"{dir_full_new}/{path}"
                        list_img.append(full_path)
    img_array = []
    for filename in list_img:
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(f"{dir_project}/video_apperance/{case_id}/video.mp4", fourcc, 5.0, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    videos.insert_one({
        "id":str(uuid.uuid4()),
        "case_id":case_id,
        "path":f"{dir_project}/video_apperance/{case_id}/video.mp4",
    })

def trimvideo(folder,videofile,count_thread,case_id):
    duration = getduration(videofile)
    time_per_segment = duration / count_thread
    for i in range(count_thread):
        command = f"ffmpeg -i {videofile} -ss {time_per_segment*i} -t {time_per_segment} -c:v copy -c:a copy  videos/{case_id}/{folder}/{i}.mp4 -y"
        subprocess.run(command, shell=True, check=True)


    
def process_videos(folder,video_file_origin,count_thread,case_id):
    # print("process_videos", folder,video_file_origin,count_thread)
    duration = getduration(video_file_origin)
    time_per_segment = duration / count_thread

    trimvideo(folder,video_file_origin,count_thread,case_id)

    video_files = [f"videos/{case_id}/{folder}/{i}.mp4" for i in range(count_thread)]  
    threads = []
    for i, video_file in enumerate(video_files):
        t = threading.Thread(target=extract_frames, args=(folder,video_file,i,time_per_segment,case_id))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    groupJson(folder,video_file_origin,count_thread,case_id)
    create_video_apperance(case_id,count_thread)


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


def handle_main(case_id, tracking_folder, target_folder):
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
                    check_insert_target = index.query(
                        vector=embedding_vector.tolist(),
                        top_k=1,
                        include_metadata=True,
                        include_values=True,
                        filter={"face": case_id},
                    )
                    matches = check_insert_target["matches"]
                    if(len(matches) > 0):
                        if(matches[0]["metadata"]["face"] == case_id):
                           flag_target_folder = False
                    if(flag_target_folder == True):
                        index.upsert(
                            vectors=[
                                    {
                                        "id": str(uuid.uuid4()),
                                        "values": embedding_vector,
                                        "metadata": {"face":case_id }
                                    },
                                ]
                        )
    list_file = []
    for path in os.listdir(tracking_folder):
        if os.path.isfile(os.path.join(tracking_folder, path)):
            full_path = f"{tracking_folder}/{path}"
            list_file.append(full_path)
            handle_multiplefile(list_file,50,case_id)

    if not os.path.exists(f"./video_apperance"):
        os.makedirs(f"./video_apperance")
    if not os.path.exists(f"./video_apperance/{case_id}"):
        os.makedirs(f"./video_apperance/{case_id}")



api = Flask(__name__)
@api.route('/analyst', methods=["POST"])
def analyst():
    case_id = request.json['case_id']
    tracking_folder = request.json['tracking_folder']
    target_folder = request.json['target_folder']
    
    myquery = { "case_id": case_id }
    facematches.delete_many(myquery)
    appearances.delete_many(myquery)
    targets.delete_many(myquery)
    videos.delete_many(myquery)

    targets.insert_one({
        "id":str(uuid.uuid4()),
        "folder":target_folder,
        "case_id":case_id
    })

    handle_main(case_id,tracking_folder,target_folder)
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
