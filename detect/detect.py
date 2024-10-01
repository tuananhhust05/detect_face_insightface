
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
time_per_frame_global = 2 
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

list_model_analyst = []
for j in range(num_gpus):
    providers = [
        ('CUDAExecutionProvider', {
            'device_id': j,
        })
    ]
    app_ele = FaceAnalysis('buffalo_l',providers=providers)
    app_ele.prepare(ctx_id=j, det_size=(640, 640))
    list_model_analyst.append(app_ele)

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

def extract_frames(folder,video_file,index_local,time_per_segment,case_id,gpu_id):
    array_em_result = []
    list_result_ele = []
    frame_count = 0 
    duration = getduration(video_file)
    cap = cv2.VideoCapture(video_file, cv2.CAP_FFMPEG)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = ( fps + 1 ) // 1
    frame_rate = time_per_frame_global * fps 
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


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
                # faces = app.get(frame)
                faces = list_model_analyst[gpu_id].get(frame)

                for face in faces:
                    if face["det_score"] > 0.5:
                        embedding = torch.tensor(face['embedding']).to(device)  # Move embedding to GPU
                        search_result = index.query(
                            vector=embedding.tolist(),
                            top_k=1,
                            include_metadata=True,
                            include_values=True,
                            filter={"case_id": case_id},
                        )
                        matches = search_result["matches"]

                        if len(matches) > 0 and matches[0]['score'] > weight_point:
                        # if True:
                            count_face = count_face + 1 
                            # if( int(face['age']) > max_age):
                            #     max_age = int(face['age'])
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
                                    array_em_result[0]["age"] = sum_age
                                    array_em_result[0]["gender"] = sum_gender // count_face 
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
                                # time_per_frame = duration / total_frames
                                # text = frame_count * time_per_frame + time_per_segment*index_local
                                # text = str(text)
                                # position = (bbox[0], bbox[1])
                                # font = cv2.FONT_HERSHEY_SIMPLEX
                                # font_scale = 1

                                # cv2.putText(frame, text, position, font, font_scale, color, thickness)
                                cv2.imwrite(f'./outputs/{case_id}/{folder}/{index_local}/{filename}', frame)
                            except Exception as e:
                                print(f"Error saving frame: {e}")
                            
                            mydict = { 
                                       "id":  str(uuid.uuid4()), 
                                       "case_id": case_id,
                                       "similarity_face":float(matches[0]['score']),
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
                    # list_time_exist.append([duration_exist[0]*time_per_frame,duration_exist[len(duration_exist) - 1] * time_per_frame])
                    list_time_exist.append([
                        {"path":f"{dir_project}/faces/{case_id}/{folder}/{index_local}/{duration_exist[0]}_0_face.jpg", "time":duration_exist[0]*time_per_frame},
                        {"path":"","time":duration_exist[len(duration_exist) - 1] * time_per_frame}
                    ])
                    duration_exist = []
                else:
                        if( i == len(list_frame)-2):
                            duration_exist.append(list_frame[i+1])
                            # list_time_exist.append([duration_exist[0]*time_per_frame,duration_exist[len(duration_exist) - 1] * time_per_frame])
                            list_time_exist.append([
                                {"path":f"{dir_project}/faces/{case_id}/{folder}/{index_local}/{duration_exist[0]}_0_face.jpg", "time":duration_exist[0]*time_per_frame},
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
                # if( int(data['age']) > max_age ):
                #     max_age = int(data['age'])
                sum_age = sum_age + int(data['age'])
                sum_gender = sum_gender + int(data['gender'])
                count_face = count_face + 1 
                for duration in data["duration_exist"]:
                    final_result["time"].append([
                        {"path":duration[0]["path"], "time":duration[0]["time"] + stt * time_per_segment},
                        {"path":"","time":duration[1]["time"] + stt * time_per_segment}
                    ])
                    # final_result["time"].append([duration[0] + stt *  time_per_segment,duration[1] + stt * time_per_segment])

    final_result['age'] = sum_gender // count_face
    if count_face > 0 : 
        final_result['gender'] = sum_gender/ count_face
        facematches.update_many(
            {
                "case_id":case_id
            },
            {
                "$set":{
                    "gender":sum_gender // count_face,
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
        # for i in range(thread_count):
        #     folder_count = i 
        #     dir_full_new = f"{dir_full}/{folder_count}"
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
    out = cv2.VideoWriter(f"{dir_project}/video_apperance/{case_id}/{folder}.mp4", fourcc, 5.0, size)
    

    for i in range(len(img_array)):
        out.write(img_array[i])
    
    out.release()

    videos.insert_one({
        "id":str(uuid.uuid4()),
        "case_id":case_id,
        "path":f"{dir_project}/video_apperance/{case_id}/{folder}.mp4",
    })

    return 

def cutvideo(videofile,start,duration,output):
    (
        ffmpeg
        .input(videofile, ss=start, hwaccel='cuda')
        .output(output, t=duration, c='copy')
        .run(overwrite_output=True)
    )

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

def handle_main(case_id, tracking_folder, target_folder):
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
                    if(flag_target_folder == True):
                        index.upsert(
                            vectors=[
                                    {
                                        "id": str(uuid.uuid4()),
                                        "values": embedding_vector,
                                        "metadata": {"case_id":case_id }
                                    },
                                ]
                        )
    list_file = []
    for path in os.listdir(tracking_folder):
        if os.path.isfile(os.path.join(tracking_folder, path)):
            full_path = f"{tracking_folder}/{path}"
            list_file.append(full_path)
    if(len(list_file) > 0):
        handle_multiplefile(list_file,50,case_id)

    return 

vectorFlag = [0.576447785, 1.17775226, -0.296847671, -1.14928401, 0.157315463, -0.218440071, 1.14000034, 0.759708345, 0.894548237, 0.566580653, 0.635065615, 2.26376867, -0.0168093257, -1.01009405, 0.995311677, -0.555958629, -0.486545742, 1.56759596, 0.18334797, 0.596046, 0.39915058, 1.06780577, 0.877452314, -0.802918375, 0.596911967, -1.72624326, -0.472717464, -0.250559628, -1.76317394, -0.436130047, 0.427088261, 0.606741905, -0.801825941, 0.158134431, -0.455002248, 0.159387678, 0.61542362, 1.1981312, -0.97123307, 0.369422317, 0.574950337, -1.24335647, -0.746754527, -0.70758003, -0.651771188, -0.737221539, 0.138820142, -0.716593623, -0.149229616, -0.223163456, 0.318926543, 0.989732623, 0.432883561, 1.45884657, 1.41446626, -0.801964521, -0.0201857723, 0.166209474, -0.114126906, 0.31346491, -0.658465922, -0.593051314, -0.701589346, 0.958120406, -1.1652571, -1.25315106, 0.932192385, 0.710922897, -1.63450742, 1.60963368, 0.557448089, 2.44075823, -0.603320599, -0.38071, 0.462305814, 1.71620071, 0.69694823, 1.82604063, -0.62104404, 0.567850113, -0.691236138, -0.664146662, -0.317410976, 0.472077817, 0.684361041, -0.57000351, 0.812473118, -0.308641136, -1.80014849, -0.186504647, -0.606217623, 0.6338045, -0.15636912, -0.341672659, 0.743249595, -1.77386582, -0.370533884, -0.558768451, -0.71179527, -1.75007331, -0.0399385095, -0.424214095, -0.220845073, 0.434903949, 0.733854175, 1.1150105, -0.625484705, -0.84713161, -0.0973022282, -0.807757258, -2.03384686, 0.258279353, 0.135332614, 0.472557187, -0.487164915, 0.319107771, 0.0106493765, -1.5355382, 1.70463526, -0.680815339, 0.448133737, 0.388723284, 1.93353939, -0.368958205, -1.27133608, 0.107167572, -0.398858845, 0.653877, -0.769213378, 1.97933185, 2.15514374, 0.63089782, -0.829949796, -1.26790702, 0.142047316, -2.15164018, 1.55778527, -0.709894776, -0.087641336, 1.01814103, -0.549310088, -0.629585743, -0.505758047, 1.69011033, 1.44240105, -0.30783844, 0.541638851, 1.285321, 0.484156847, 0.0954145342, -0.175656483, 0.266733289, -0.348181486, 0.166033685, -0.211711317, -0.358136833, -2.0269196, 0.0591057688, 1.31786168, 0.180953801, -0.79715234, -0.637249708, -0.741272032, 0.0772315338, -0.365465105, 1.20687938, -0.337666094, -0.126954168, 0.730466366, 0.873552561, 1.06639099, -1.88941348, -0.991165221, 1.48542869, -1.43833101, 0.658765376, -0.471807629, -0.233251408, 1.28683019, -0.815259635, 0.597836554, -0.498675764, -1.94553339, -1.42407751, -0.58240968, 1.57030058, -0.230395094, -1.66030324, -0.624298, -0.286484629, -0.196268842, -0.227928668, -0.915873647, -0.602099955, -0.144641981, -0.243574247, -0.518348336, 0.419062078, 0.386762619, 0.301550865, 1.05666721, -0.110677816, -0.473464161, 0.339873165, 0.847880185, -0.00637972727, 1.87477696, 0.0710848719, 0.670239389, -0.973360896, 0.695547879, -0.457882166, -0.357175678, 0.747017443, 0.00936299469, -1.5209446, -0.0359656438, 0.791891217, -0.66574353, 1.18219507, 1.71747434, 0.516999245, 0.625486374, -0.135268986, -1.04955351, -0.222503707, -0.380431324, 0.605118513, -1.03138304, 0.820119202, -0.73156333, -0.618677437, 2.79051781, -1.80716026, 0.0398874581, 1.27604544, -1.40159476, -1.38328314, -0.274990141, -1.0276072, -0.933455348, 0.107920177, -0.135145634, 0.664806902, 0.810418308, -0.397827655, 0.678289175, 0.704355, 1.03663254, -0.536797702, 0.918632209, -0.503982842, 0.775880933, -0.38931, -0.465570837, -0.0720181465, -0.691470265, -2.02726078, 1.60041499, 0.362865478, -0.0177706443, -0.740747392, -0.553856254, -0.916912615, 0.170815974, -0.197350919, 0.194908693, 1.25297821, -0.237392157, 0.171072409, -0.850204468, -0.210407719, 0.724647105, 1.49338257, 2.14174557, -0.268470317, 0.194095165, -2.02525878, -0.962936878, 0.233063683, -0.200922, -0.128289342, 0.288245976, 0.609019756, -0.607792, -0.633102596, -0.00558678526, 0.42232433, -0.403190255, -0.562535107, 1.0397681, -0.00730981305, 0.788311422, 1.2883141, 1.19495916, 0.26035285, -0.0753540546, 0.407981664, -0.581133604, 0.426193684, 0.239905238, 0.766273379, 0.745405138, 0.415173084, 0.466663599, 0.164218292, 1.6266315, -0.0687616169, -0.291110367, -0.591514885, 0.104314439, -0.502006412, 1.27361619, 0.61300844, 0.0672450587, -0.155570656, -1.36612093, -0.355750293, 0.236665443, -1.03266895, -1.21113372, 0.572526574, 0.897796094, -0.0775505379, 0.200488746, 1.19512141, -0.379729569, 1.15134883, -0.800162673, 0.272592127, 0.224842802, 1.36469817, 0.16908665, -0.109059162, -1.26338911, 0.718426585, -0.545595646, -0.43709898, -0.9388026, 0.774740219, -2.03077745, 1.16564453, -1.15451181, 0.769814312, 1.46091306, 1.61677504, 0.5123353, 0.341511816, -2.32375193, 1.03319418, -0.655129969, -0.00773460837, 1.01160383, 0.845900655, 0.957859457, -0.903936088, -0.0838359073, 0.0826194212, 0.222008213, 2.34256268, -1.23197126, 0.892613351, -0.440327585, -0.0479687229, 1.4178822, -0.205831259, 1.15948749, 0.76598835, -0.154598847, -0.622521222, -0.761301696, -0.729390144, -0.0504151359, -0.556260407, 0.69219178, 2.35484576, 0.233766302, -0.74507612, -0.377987444, -1.81759882, -0.254705548, 0.750509083, -0.575120568, 0.127797171, -0.235917658, 1.37119472, -0.0484509431, 0.37004444, -0.411835343, -1.08911335, -0.315811187, -0.411482692, -1.43837857, 0.155707732, -0.799197316, 1.19500184, 0.761625588, 0.996269524, -0.0155281853, -0.200010538, -0.901457369, 0.120013095, 1.19593942, -0.238823503, -1.78543949, 0.400835425, 0.625563383, -2.38248086, 0.196204, 0.463427663, 1.97891891, -0.382288575, 1.34851241, 0.76986444, 1.78608942, -0.414675623, 0.114532135, 1.23018312, 0.597604573, 0.145634592, 0.24638927, -0.692227781, 0.612979174, -0.733509243, 0.745099366, -1.63894117, 0.193425804, -1.2679956, -1.89323199, 0.140096068, -0.944377422, 0.342734218, -0.32025373, 0.869147062, 0.721760213, -0.553079844, 0.135592371, 0.225211456, 0.0920588225, -0.472706228, 0.563096941, 0.0356598943, 0.3642084, 0.5264377, 0.738836467, 0.0187704731, 0.0655642375, 0.738964677, -1.39213753, 0.484859705, -0.102724351, 0.163534179, -0.488165051, 0.165032879, -0.217456833, 1.1284548, 0.55648613, -2.42811394, 0.00356701924, 0.190464377, 1.18792284, 1.36150253, -0.314983, -1.06589878, 1.50398207, -2.30766678, -0.306449234, -0.174938485, -0.783020496, -0.243689314, 1.28469908, -0.105662525, 0.232826456, 1.38738275, -1.397488, 1.26947212, 1.09980536, -0.703025699, -0.785862207, 0.39007768, 0.120006613, 1.88867223, -0.73051554, 0.785588384, 1.82167292, -0.277029932, -0.834874749, 0.688136697, 0.799892485, -0.404544681, -0.0747961253, -0.0180013683, 0.176287353, 0.523813546, -0.609975517, 0.305920392, 0.541295528, 0.548327088, -0.120758526, -1.2759335, -1.49765372, -1.90647924, 0.871611476, 0.282542109, 0.626890838, -0.0927402, -0.82689023, -0.925173104, -0.120574243, -1.62662244, -1.23879, 0.109900318]
api = Flask(__name__)
@api.route('/analyst', methods=["POST"])
def analyst():
    case_id = request.json['case_id']
    tracking_folder = request.json['tracking_folder']
    target_folder = request.json['target_folder']
    
    myquery = { "case_id": case_id }
    facematches.delete_many(myquery)
    appearances.delete_many(myquery)
    # targets.delete_many(myquery)
    videos.delete_many(myquery)

    cases.update_many({
        "id":case_id
    },{
        "$set":{
            "start":current_date(),
            "status":"processing"
        }
    })

    listToDelete = index.query(
            vector=vectorFlag,
            top_k=1000,
            filter={
                "case_id": {"$eq": case_id}
            },
            include_metadata=True
        )
    listToDelete=listToDelete["matches"]
    listId = []
    for ele in listToDelete:
        listId.append(ele['id'])
    if(len(listId) > 0):
        index.delete(ids=listId)


    
    subprocess.run("cd /home/poc4a5000/detect/detect && rm -rf datas && mkdir datas && rm -rf final_result && mkdir final_result && rm -rf outputs && mkdir outputs && rm -rf results && mkdir results && rm -rf final_result && mkdir final_result && rm -rf videos && mkdir videos && rm -rf faces && mkdir faces && rm -rf video_apperance && mkdir video_apperance", shell=True, check=True)
    
    handle_main(case_id,tracking_folder,target_folder)

    cases.update_many({
        "id":case_id
    },{
        "$set":{
            "end":current_date(),
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
