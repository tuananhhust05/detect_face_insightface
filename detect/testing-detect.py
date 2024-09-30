import os
import datetime
import numpy as np
import cv2
import torch
import json
import logging
import uuid
from flask import Flask, jsonify, request
import pymongo
from multiprocessing import Process, current_process, set_start_method
from insightface.app import FaceAnalysis
from insightface.model_zoo import model_zoo
from pinecone import Pinecone
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# MongoDB connection
myclient = pymongo.MongoClient("mongodb://root:facex@192.168.50.10:27018")
mydb = myclient["faceX"]
facematches = mydb["facematches"]
appearances = mydb["appearances"]
targets = mydb["targets"]
videos = mydb["videos"]

# Project directory
dir_project = "/home/poc4a5000/detect/detect"

# Pinecone initialization
api_key = "6bebb6ba-195f-471e-bb60-e0209bd5c697"
if not api_key:
    logging.error("Pinecone API key not found.")
    raise ValueError("Pinecone API key not found.")

pc = Pinecone(api_key=api_key)
index_name = "detectcamera"
index = pc.Index(index_name)

weight_point = 0.4
time_per_frame_global = 2

# GPU allocation
app_gpu_id = 0  # For app_recognize
video_gpu_ids = []  # For video processing

# Detect available GPUs (do not use torch.cuda here)
num_gpus = int(os.environ.get("NUM_GPUS", 4))  # Assuming 4 GPUs
if num_gpus >= 1:
    logging.info(f"Number of GPUs available: {num_gpus}")
    video_gpu_ids = list(range(1, num_gpus))  # For video processing
else:
    logging.warning("No GPUs detected. Exiting.")
    exit(1)

# Utility functions
def getduration(file):
    data = cv2.VideoCapture(file)
    frames = data.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = data.get(cv2.CAP_PROP_FPS)
    data.release()
    seconds = round(frames / fps)
    return seconds

def current_date():
    format_date = "%Y-%m-%d %H:%M:%S"
    now = datetime.datetime.now()
    date_string = now.strftime(format_date)
    return datetime.datetime.strptime(date_string, format_date)

# Batch processing function
def process_batch(frames_batch, frame_indices, folder, video_file, index_local, time_per_segment, case_id, duration, total_frames):
    try:
        # Initialize FaceAnalysis and model in this process
        face_analysis = FaceAnalysis('buffalo_l')
        face_analysis.prepare(ctx_id=0, det_size=(640, 640))
        model = model_zoo.get_model('/home/poc4a5000/.insightface/models/buffalo_l/det_10g.onnx')
        model.prepare(ctx_id=0, det_size=(640, 640))

        # Device string
        device_str = 'cuda:0'

        for frame, frame_count in zip(frames_batch, frame_indices):
            detections, _ = model.detect(frame, input_size=(640, 640))
            if detections is not None and len(detections) > 0:
                sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                sharpen = cv2.filter2D(frame, -1, sharpen_kernel)
                denoised = cv2.fastNlMeansDenoisingColored(sharpen, None, 10, 10, 7, 21)
                faces = face_analysis.get(denoised)

                sum_age = 0 
                sum_gender = 0 
                count_face = 0 

                for face in faces:
                    if face["det_score"] > 0.5:
                        embedding = torch.tensor(face['embedding']).to(device_str)
                        search_result = index.query(
                            vector=embedding.tolist(),
                            top_k=1,
                            include_metadata=True,
                            include_values=True,
                            filter={"face": case_id},
                        )
                        matches = search_result["matches"]

                        if len(matches) > 0 and matches[0]['score'] > weight_point:
                            count_face += 1 
                            sum_age += int(face['age'])
                            sum_gender += int(face['gender'])

                            # Record the result
                            mydict = { 
                                "id":  str(uuid.uuid4()), 
                                "case_id": case_id,
                                "similarity_face":str(matches[0]['score']),
                                "gender":int(face['gender']),
                                "age":int(face['age']),
                                "time_invideo":str(frame_count * time_per_frame_global + time_per_segment * index_local),
                                "proofImage":f'/home/poc4a5000/detect/detect/faces/{case_id}/{folder}/{index_local}/{frame_count}_0_face.jpg',
                                "url":f'/home/poc4a5000/detect/detect/faces/{case_id}/{folder}/{index_local}/{frame_count}_0_face.jpg',
                                "createdAt":current_date(),
                                "updatedAt":current_date(),
                                "file":folder
                            }
                            facematches.insert_one(mydict)

                            # Draw rectangle around the face
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
                            time_in_video = frame_count * time_per_frame_global + time_per_segment * index_local
                            text = str(time_in_video)
                            position = (bbox[0], bbox[1])
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            font_scale = 1

                            cv2.putText(frame, text, position, font, font_scale, color, thickness)
                            cv2.imwrite(f'{output_dir}/{filename}', frame)

    except Exception as e:
        logging.error(f"Error processing batch: {e}")
    finally:
        torch.cuda.empty_cache()

# Video processing function
def process_video(folder, video_file, index_local, time_per_segment, case_id, duration, total_frames):
    frame_count = 0
    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_rate = time_per_frame_global * fps 

    frames_batch = []
    frame_indices = []
    batch_size = 16  # Adjust batch size as needed

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % frame_rate == 0:
            frames_batch.append(frame)
            frame_indices.append(frame_count)
            if len(frames_batch) == batch_size:
                process_batch(frames_batch, frame_indices, folder, video_file, index_local, time_per_segment, case_id, duration, total_frames)
                frames_batch = []
                frame_indices = []
    if frames_batch:
        process_batch(frames_batch, frame_indices, folder, video_file, index_local, time_per_segment, case_id, duration, total_frames)
    cap.release()

# Function to trim video into segments
def trimvideo(folder, videofile, count_thread, case_id):
    duration = getduration(videofile)
    time_per_segment = duration / count_thread
    os.makedirs(f"videos/{case_id}/{folder}", exist_ok=True)
    for i in range(count_thread):
        start_time = time_per_segment * i
        command = f"ffmpeg -i {videofile} -ss {start_time} -t {time_per_segment} -c:v copy -c:a copy videos/{case_id}/{folder}/{i}.mp4 -y"
        subprocess.run(command, shell=True, check=True)

# Worker process function
def worker_process(gpu_id, folder, video_file, index_local, time_per_segment, case_id, duration, total_frames):
    # Set CUDA_VISIBLE_DEVICES for this process before any CUDA code
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    logging.info(f"Process {current_process().name} started with GPU ID: {gpu_id}")
    process_video(folder, video_file, index_local, time_per_segment, case_id, duration, total_frames)

# Function to process target images
def target_processing(case_id, target_folder):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(app_gpu_id)
    app_recognize = FaceAnalysis('buffalo_l')
    app_recognize.prepare(ctx_id=0, det_thresh=0.3, det_size=(640, 640))
    logging.info("app_recognize initialized on GPU 0")

    flag_target_folder = True
    for path in os.listdir(target_folder):
        if flag_target_folder and os.path.isfile(os.path.join(target_folder, path)):
            full_path = os.path.join(target_folder, path)
            img = cv2.imread(full_path)
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
                                "values": embedding_vector,
                                "metadata": {"face": case_id}
                            },
                        ]
                    )

# Function to handle multiple files using multiprocessing
def handle_multiplefile(listfile, thread, case_id):
    processes = []
    process_count = 0  # Counter for processes created

    for idx, file in enumerate(listfile):
        folder_name = os.path.splitext(os.path.basename(file))[0]
        duration = getduration(file)
        time_per_segment = duration / thread
        total_frames = int(cv2.VideoCapture(file).get(cv2.CAP_PROP_FRAME_COUNT))

        # Trim video into segments
        trimvideo(folder_name, file, thread, case_id)

        video_files = [f"videos/{case_id}/{folder_name}/{i}.mp4" for i in range(thread)]

        for i, vf in enumerate(video_files):
            gpu_id = video_gpu_ids[process_count % len(video_gpu_ids)]  # Round-robin GPU assignment per process
            p = Process(target=worker_process, args=(gpu_id, folder_name, vf, i, time_per_segment, case_id, duration, total_frames))
            p.start()
            processes.append(p)
            process_count += 1  # Increment process counter

    # Wait for all processes to complete
    for p in processes:
        p.join()

    # Additional processing (e.g., groupJson, create_video_apperance) if needed

# Main processing function
def handle_main(case_id, tracking_folder, target_folder):
    # Start target processing in a separate process
    target_process = Process(target=target_processing, args=(case_id, target_folder))
    target_process.start()
    target_process.join()

    # List of files to process
    list_file = [
        os.path.join(tracking_folder, f) 
        for f in os.listdir(tracking_folder) 
        if os.path.isfile(os.path.join(tracking_folder, f))
    ]
    handle_multiplefile(list_file, 8, case_id)

    # Create appearance video directory
    os.makedirs(f"./video_apperance/{case_id}", exist_ok=True)

# Flask API setup
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
    import multiprocessing
    multiprocessing.set_start_method('spawn')  # Use 'spawn' instead of 'fork'
    api.run(debug=True, port=5235, host='0.0.0.0')
