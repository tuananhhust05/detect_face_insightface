import datetime
import numpy as np
import os
import cv2
import torch
import json
import time
from insightface.app import FaceAnalysis
from insightface.model_zoo import model_zoo
from pinecone import Pinecone, ServerlessSpec  # Updated import
import subprocess
import multiprocessing
import uuid
from flask import Flask, jsonify, request
import pymongo

# Initialize MongoDB connection
myclient = pymongo.MongoClient("mongodb://root:facex@192.168.50.10:27018")
mydb = myclient["faceX"]
facematches = mydb["facematches"]
appearances = mydb["appearances"]
targets = mydb["targets"]
videos = mydb["videos"]

dir_project = "/home/poc4a5000/detect/detect"

# Initialize Pinecone client
pc = Pinecone(api_key="6bebb6ba-195f-471e-bb60-e0209bd5c697")

# Check if the index exists; if not, create it
if 'detectcamera' not in pc.list_indexes().names():
    pc.create_index(
        name='detectcamera', 
        dimension=512,  # Adjust to match your embedding dimension
        metric='cosine',  # Use 'cosine' or 'euclidean' as needed
        spec=ServerlessSpec(
            cloud='gcp',  # Adjust cloud provider if necessary
            region='us-west1'  # Adjust region as needed
        )
    )

# Access the index
index = pc.index('detectcamera')

weight_point = 0.4
time_per_frame_global = 2  # seconds per frame to process

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
    seconds = frames / fps 
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

        # Initialize FaceAnalysis with providers within the process
        app = FaceAnalysis(
            name='buffalo_l',
            providers=providers,
            allowed_modules=['detection', 'recognition']
        )
        app.prepare(ctx_id=gpu_id, det_size=(640, 640))

        cap = cv2.VideoCapture(video_file)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            print(f"FPS is zero for video {video_file}")
            cap.release()
            return
        frame_interval = int(fps * time_per_frame_global)
        if frame_interval == 0:
            frame_interval = 1  # Ensure at least one frame is processed
        print(f"Processing video at {fps} fps, frame interval {frame_interval}.")

        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            if frame_count % frame_interval == 0:
                faces = app.get(frame)
                if faces:
                    for idx, face in enumerate(faces):
                        if face.get("det_score", 0) > 0.5:
                            embedding = torch.tensor(face['embedding']).to(device)
                            # Use the 'index' from the Pinecone client instance
                            search_result = index.query(
                                vector=embedding.cpu().numpy().tolist(),
                                top_k=1,
                                include_metadata=True,
                                include_values=True,
                                filter={"face": case_id},
                            )
                            matches = search_result["matches"]

                            if len(matches) > 0 and matches[0]['score'] > weight_point:
                                # Save face image and annotated frame
                                bbox = [int(b) for b in face['bbox']]
                                filename = f"{frame_count}_{idx}_face.jpg"
                                face_dir = f"./faces/{case_id}/{folder}/{index_local}"
                                output_dir = f"./outputs/{case_id}/{folder}/{index_local}"
                                os.makedirs(face_dir, exist_ok=True)
                                os.makedirs(output_dir, exist_ok=True)

                                cv2.imwrite(f'{face_dir}/{filename}', frame[bbox[1]:bbox[3], bbox[0]:bbox[2]])

                                # Draw rectangle and timestamp
                                top_left = (bbox[0], bbox[1])
                                bottom_right = (bbox[2], bbox[3])
                                color = (255, 0, 0)
                                thickness = 2
                                cv2.rectangle(frame, top_left, bottom_right, color, thickness)

                                frame_time = (frame_count / fps) + (time_per_segment * index_local)
                                text = f"{frame_time:.2f}s"
                                position = (bbox[0], bbox[1] - 10)
                                font = cv2.FONT_HERSHEY_SIMPLEX
                                font_scale = 0.5
                                cv2.putText(frame, text, position, font, font_scale, color, thickness)

                                cv2.imwrite(f'{output_dir}/{filename}', frame)

                                # Save to database
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

    for path in os.listdir(target_folder):
        full_path = os.path.join(target_folder, path)
        if os.path.isfile(full_path):
            img = cv2.imread(full_path)
            print("Processing target image:", full_path)
            faces = app_recognize.get(img)
            for face in faces:
                embedding_vector = face['embedding']
                # Use the 'index' from the Pinecone client instance
                search_result = index.query(
                    vector=embedding_vector.tolist(),
                    top_k=1,
                    include_metadata=True,
                    include_values=True,
                    filter={"face": case_id},
                )
                matches = search_result["matches"]
                if not matches:
                    index.upsert(
                        vectors=[
                            {
                                "id": str(uuid.uuid4()),
                                "values": embedding_vector.tolist(),
                                "metadata": {"face": case_id}
                            },
                        ]
                    )

def trimvideo(folder, videofile, count_thread, case_id):
    duration = getduration(videofile)
    if duration == 0:
        print(f"Video duration is zero for file {videofile}")
        return
    time_per_segment = duration / count_thread
    os.makedirs(f"videos/{case_id}/{folder}", exist_ok=True)
    for i in range(count_thread):
        start_time = time_per_segment * i
        command = f"ffmpeg -ss {start_time} -t {time_per_segment} -i {videofile} -c:v libx264 -c:a aac videos/{case_id}/{folder}/{i}.mp4 -y"
        subprocess.run(command, shell=True, check=True)

def process_videos(folder, video_file_origin, count_thread, case_id):
    duration = getduration(video_file_origin)
    time_per_segment = duration / count_thread

    trimvideo(folder, video_file_origin, count_thread, case_id)

    video_files = [f"videos/{case_id}/{folder}/{i}.mp4" for i in range(count_thread)]
    processes = []
    for i, video_file in enumerate(video_files):
        gpu_id = gpu_ids[i % num_gpus]
        p = multiprocessing.Process(target=extract_frames, args=(folder, video_file, i, time_per_segment, case_id, gpu_id))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

def handle_multiplefile(listfile, thread, case_id):
    for file in listfile:
        file_name = os.path.splitext(os.path.basename(file))[0]
        folder = file_name
        process_videos(folder, file, num_gpus, case_id)
        subprocess.run(f"rm -rf videos/{case_id}/{folder}", shell=True, check=True)

def handle_main(case_id, tracking_folder, target_folder):
    # Start target processing
    p_target = multiprocessing.Process(target=process_targets, args=(case_id, target_folder, gpu_ids[0]))
    p_target.start()
    p_target.join()

    # Process videos
    list_file = []
    for path in os.listdir(tracking_folder):
        full_path = os.path.join(tracking_folder, path)
        if os.path.isfile(full_path):
            list_file.append(full_path)
    handle_multiplefile(list_file, num_gpus, case_id)

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
