import datetime
import numpy as np
import os
import cv2
import insightface
import torch
import json
import time
from insightface.app import FaceAnalysis
from insightface.model_zoo import model_zoo
from pinecone import Pinecone
import subprocess
import multiprocessing
import uuid
from flask import Flask, jsonify, request
import pymongo

# Set multiprocessing start method
multiprocessing.set_start_method('spawn', force=True)

# MongoDB connection
myclient = pymongo.MongoClient("mongodb://root:facex@192.168.50.10:27018")

mydb = myclient["faceX"]
facematches = mydb["facematches"]
appearances = mydb["appearances"]
targets = mydb["targets"]
videos = mydb["videos"]

dir_project = "/home/poc4a5000/detect/detect"

# Initialize Pinecone
pc = Pinecone(api_key="6bebb6ba-195f-471e-bb60-e0209bd5c697")

index = pc.Index("detectcamera")

weight_point = 0.4
time_per_frame_global = 2 

# Detect available GPUs
num_gpus = torch.cuda.device_count()
print(f"Number of GPUs available: {num_gpus}")
gpu_ids = list(range(num_gpus))

def getduration(file):
    data = cv2.VideoCapture(file) 
    frames = data.get(cv2.CAP_PROP_FRAME_COUNT) 
    fps = data.get(cv2.CAP_PROP_FPS) 
    data.release()
    if fps == 0:
        return 0
    seconds = round(frames / fps) 
    return seconds

def current_date():
    format_date = "%Y-%m-%d %H:%M:%S"
    now = datetime.datetime.now()
    date_string = now.strftime(format_date)
    return datetime.datetime.strptime(date_string, format_date)

def extract_frames(folder, video_file, index_local, time_per_segment, case_id, gpu_id):
    print(f"Process started with GPU ID: {gpu_id}")
    try:
        # Set the device
        torch.cuda.set_device(gpu_id)
        device = torch.device(f'cuda:{gpu_id}')

        # Define providers with device_id
        providers = [
            ('CUDAExecutionProvider', {
                'device_id': gpu_id,
            })
        ]

        # Initialize FaceAnalysis with providers
        app = FaceAnalysis(
            name='buffalo_l',
            providers=providers,
            allowed_modules=['detection', 'recognition']
        )
        app.prepare(ctx_id=gpu_id, det_size=(640, 640))

        # Load the model with providers
        model = model_zoo.get_model(
            '/home/poc4a5000/.insightface/models/buffalo_l/det_10g.onnx',
            providers=providers
        )
        model.prepare(ctx_id=gpu_id, det_size=(640, 640))

        array_em_result = []
        list_result_ele = []
        frame_count = 0 
        duration = getduration(video_file)
        cap = cv2.VideoCapture(video_file)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            print(f"FPS is zero for video {video_file}")
            cap.release()
            return
        # frame_rate = int(fps)
        frame_rate = 60 

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            if frame_count % frame_rate == 0:
                print("Processing frame", frame_count)
                facechecks = model.detect(frame, input_size=(640, 640))
                flagDetect = False
                if len(facechecks) > 0 and len(facechecks[0]) > 0:
                    flagDetect = True

                if flagDetect:
                    print("Face detected...")
                    sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                    sharpen = cv2.filter2D(frame, -1, sharpen_kernel)
                    frame_denoised = cv2.fastNlMeansDenoisingColored(sharpen, None, 10, 10, 7, 21)
                    faces = app.get(frame_denoised)

                    sum_age = 0 
                    sum_gender = 0 
                    count_face = 0 
                    for face in faces:
                        if face.get("det_score", 0) > 0.5:
                            embedding = torch.tensor(face['embedding']).to(device)
                            search_result = index.query(
                                vector=embedding.cpu().numpy().tolist(),
                                top_k=1,
                                include_metadata=True,
                                include_values=True,
                                filter={"face": case_id},
                            )
                            matches = search_result["matches"]

                            if len(matches) > 0 and matches[0]['score'] > weight_point:
                                count_face += 1 
                                sum_age += int(face.get('age', 0))
                                sum_gender += int(face.get('gender', 0))
 
                                if len(array_em_result) == 0:
                                    array_em_result.append({
                                        "speaker": 0,
                                        "gender": int(face.get('gender', -1)),
                                        "age": int(face.get('age', -1)),
                                        "frames": [frame_count],
                                    })
                                    
                                else:
                                    if count_face > 0:
                                        array_em_result[0]["age"] = sum_age // count_face 
                                        array_em_result[0]["gender"] = sum_gender // count_face 
                                        array_em_result[0]["frames"].append(frame_count)

                                try:
                                    bbox = [int(b) for b in face['bbox']]
                                    filename = f"{frame_count}_0_face.jpg"
                                    face_dir = f"./faces/{case_id}/{folder}/{index_local}"
                                    output_dir = f"./outputs/{case_id}/{folder}/{index_local}"
                                    os.makedirs(face_dir, exist_ok=True)
                                    os.makedirs(output_dir, exist_ok=True)

                                    cv2.imwrite(f'{face_dir}/{filename}', frame[bbox[1]:bbox[3], bbox[0]:bbox[2], ::-1])

                                    top_left = (bbox[0], bbox[1])
                                    bottom_right = (bbox[2], bbox[3])
                                    color = (255, 0, 0)
                                    thickness = 2
                                    cv2.rectangle(frame, top_left, bottom_right, color, thickness)
                                    time_per_frame = duration / total_frames
                                    text = frame_count * time_per_frame + time_per_segment * index_local
                                    text = str(text)
                                    position = (bbox[0], bbox[1])
                                    font = cv2.FONT_HERSHEY_SIMPLEX
                                    font_scale = 1

                                    cv2.putText(frame, text, position, font, font_scale, color, thickness)
                                    cv2.imwrite(f'{output_dir}/{filename}', frame)
                                except Exception as e:
                                    print(f"Error saving frame: {e}")

                                mydict = { 
                                    "id":  str(uuid.uuid4()), 
                                    "case_id": case_id,
                                    "similarity_face": str(matches[0]['score']),
                                    "gender": int(face.get('gender', -1)),
                                    "age": int(face.get('age', -1)),
                                    "time_invideo": text,
                                    "proofImage": os.path.abspath(f'{face_dir}/{filename}'),
                                    "url": os.path.abspath(f'{face_dir}/{filename}'),
                                    "createdAt": current_date(),
                                    "updatedAt": current_date(),
                                    "file": folder
                                }
                                facematches.insert_one(mydict)

        cap.release()
    except Exception as e:
        print(f"Error in process: {e}")

# Move process_targets to global scope
def process_targets(case_id, target_folder, gpu_id):
    torch.cuda.set_device(gpu_id)
    device = torch.device(f'cuda:{gpu_id}')

    providers = [
        ('CUDAExecutionProvider', {
            'device_id': gpu_id,
        })
    ]

    app_recognize = FaceAnalysis(
        name='buffalo_l',
        providers=providers,
        allowed_modules=['detection', 'recognition']
    )
    app_recognize.prepare(ctx_id=gpu_id, det_thresh=0.3, det_size=(640, 640))

    flag_target_folder = True
    for path in os.listdir(target_folder):
        if flag_target_folder and os.path.isfile(os.path.join(target_folder, path)):
            full_path = os.path.join(target_folder, path)
            img = cv2.imread(full_path)
            print("Processing target image:", full_path)
            faces = app_recognize.get(img)
            for face in faces:
                embedding_vector = face['embedding']
                search_result = index.query(
                    vector=embedding_vector.tolist(),
                    top_k=1,
                    include_metadata=True,
                    include_values=True,
                    filter={"face": case_id},
                )
                matches = search_result["matches"]
                if matches and matches[0]["metadata"]["face"] == case_id:
                    flag_target_folder = False
                if flag_target_folder:
                    index.upsert(
                        vectors=[
                            {
                                "id": str(uuid.uuid4()),
                                "values": embedding_vector.tolist(),
                                "metadata": {"face": case_id}
                            },
                        ]
                    )

def groupJson(folder, video_file, count_thread, case_id):
    # Your existing code for groupJson...
    pass  # For brevity

def create_video_apperance(case_id, thread_count):
    # Your existing code for create_video_apperance...
    pass  # For brevity

def trimvideo(folder, videofile, count_thread, case_id):
    duration = getduration(videofile)
    if duration == 0:
        print(f"Video duration is zero for file {videofile}")
        return
    time_per_segment = duration / count_thread
    os.makedirs(f"videos/{case_id}/{folder}", exist_ok=True)
    for i in range(count_thread):
        start_time = time_per_segment * i
        command = f"ffmpeg -i {videofile} -ss {start_time} -t {time_per_segment} -c:v copy -c:a copy videos/{case_id}/{folder}/{i}.mp4 -y"
        subprocess.run(command, shell=True, check=True)

def process_videos(folder, video_file_origin, count_thread, case_id):
    duration = getduration(video_file_origin)
    if duration == 0:
        print(f"Video duration is zero for file {video_file_origin}")
        return
    time_per_segment = duration / count_thread

    trimvideo(folder, video_file_origin, count_thread, case_id)

    video_files = [f"videos/{case_id}/{folder}/{i}.mp4" for i in range(count_thread)]  
    processes = []
    for i, video_file in enumerate(video_files):
        gpu_id = gpu_ids[i % num_gpus]
        p = multiprocessing.Process(target=extract_frames, args=(folder, video_file, i, time_per_segment, case_id, gpu_id))
        processes.append(p)
        p.start()

        # Limit to one process per GPU to avoid out-of-memory errors
        if (i + 1) % num_gpus == 0:
            for p in processes:
                p.join()
            processes = []

    # Join any remaining processes
    for p in processes:
        p.join()

    groupJson(folder, video_file_origin, count_thread, case_id)
    create_video_apperance(case_id, count_thread)

def handle_multiplefile(listfile, thread, case_id):
    for file in listfile:
        file_name = os.path.splitext(os.path.basename(file))[0]
        # Create necessary directories
        # ...

        folder = file_name
        process_videos(folder, file, thread, case_id)
        subprocess.run(f"rm -rf videos/{file_name}", shell=True, check=True)

def handle_main(case_id, tracking_folder, target_folder):
    # Start target processing
    p_target = multiprocessing.Process(target=process_targets, args=(case_id, target_folder, gpu_ids[0]))
    p_target.start()
    p_target.join()

    # Process videos
    list_file = []
    for path in os.listdir(tracking_folder):
        if os.path.isfile(os.path.join(tracking_folder, path)):
            full_path = os.path.join(tracking_folder, path)
            list_file.append(full_path)
    handle_multiplefile(list_file, 8, case_id)

    # Create appearance video directory
    os.makedirs(f"./video_apperance/{case_id}", exist_ok=True)

api = Flask(__name__)

@api.route('/analyst', methods=["POST"])
def analyst():
    case_id = request.json.get('case_id')
    tracking_folder = request.json.get('tracking_folder')
    target_folder = request.json.get('target_folder')
    
    if not all([case_id, tracking_folder, target_folder]):
        return jsonify({"error": "Missing parameters."}), 400

    # Remove existing records
    myquery = { "case_id": case_id }
    facematches.delete_many(myquery)
    appearances.delete_many(myquery)
    targets.delete_many(myquery)
    videos.delete_many(myquery)

    # Insert new target
    targets.insert_one({
        "id": str(uuid.uuid4()),
        "folder": target_folder,
        "case_id": case_id
    })

    handle_main(case_id, tracking_folder, target_folder)
    return jsonify({"data": "ok"})

if __name__ == '__main__':
    api.run(debug=True, port=5235, host='0.0.0.0')
