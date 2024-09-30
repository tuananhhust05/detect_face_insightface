import os
import datetime
import numpy as np
import cv2
import torch
from mutagen.mp4 import MP4
import json
import time
from numpy.linalg import norm
from insightface.app import FaceAnalysis
from insightface.model_zoo import model_zoo
from pinecone import Pinecone
import subprocess
import logging
import uuid
from flask import Flask, jsonify, request
import pymongo
from multiprocessing import Process, Queue, current_process
from torch.cuda.amp import autocast
import sys
# Function to set up logging
def setup_logging():
    logger = logging.getLogger()
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(processName)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

# Cấu hình logging in the main process
setup_logging()

# Kết nối MongoDB
myclient = pymongo.MongoClient("mongodb://root:facex@192.168.50.10:27018")
mydb = myclient["faceX"]
facematches = mydb["facematches"]
appearances = mydb["appearances"]
targets = mydb["targets"]
videos = mydb["videos"]

# Thư mục dự án
dir_project = "/home/poc4a5000/detect/detect"

# Thiết lập thiết bị
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# Khởi tạo Pinecone với API Key từ biến môi trường
api_key = "6bebb6ba-195f-471e-bb60-e0209bd5c697"
if not api_key:
    logging.error("Pinecone API key not found in environment variables.")
    raise ValueError("Pinecone API key not found in environment variables.")

pc = Pinecone(api_key=api_key)
index_name = "detectcamera"

index = pc.Index(index_name)

weight_point = 0.4
time_per_frame_global = 2 

# Phát hiện số lượng GPU có sẵn
num_gpus = torch.cuda.device_count()
if num_gpus >= 1:
    logging.info(f"Number of GPUs available: {num_gpus}")
else:
    logging.warning("No GPUs detected. Exiting.")
    exit(1)

# Phân bổ GPU:
# GPU 0: Dành cho app_recognize
# GPU 1, 2, 3: Dành cho video processing
app_gpu_id = 0
video_gpu_ids = list(range(1, num_gpus))  # [1, 2, 3]
# If you want to include GPU 0 in video processing, uncomment the following line:
# video_gpu_ids = list(range(0, num_gpus))  # [0, 1, 2, 3]

# Hàm tiện ích
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

# Hàm xử lý batch trong từng tiến trình
def process_batch(frames_batch, frame_indices, folder, video_file, index_local, time_per_segment, case_id, duration, total_frames, gpu_id):
    try:
        if gpu_id >= 0:
            device_str = f'cuda:{gpu_id}'
            providers = ['CUDAExecutionProvider']
        else:
            device_str = 'cpu'
            providers = []

        # Khởi tạo FaceAnalysis và model trong tiến trình này
        logging.info(f"Initializing models on GPU {gpu_id}")
        face_analysis = FaceAnalysis('buffalo_l', providers=providers)
        face_analysis.prepare(ctx_id=0, det_size=(640, 640))  # ctx_id should be 0 since CUDA_VISIBLE_DEVICES is set
        model = model_zoo.get_model('/home/poc4a5000/.insightface/models/buffalo_l/det_10g.onnx')
        model.prepare(ctx_id=0, det_size=(640, 640))

        logging.info(f"Process {current_process().name} using device {device_str}")

        with torch.cuda.amp.autocast(enabled=(gpu_id >=0)):
            # Giả định rằng model.detect có thể xử lý batch, nếu không cần xử lý từng frame một
            detections = [model.detect(frame, input_size=(640, 640)) for frame in frames_batch]

        for frame, frame_count, detection in zip(frames_batch, frame_indices, detections):
            if detection and len(detection) > 0:
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

                            # Ghi lại kết quả
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

                            # Vẽ hình chữ nhật quanh khuôn mặt
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
        logging.error(f"Error processing batch on GPU {gpu_id}: {e}")
    finally:
        torch.cuda.empty_cache()
        logging.info(f"Process {current_process().name} on GPU {gpu_id} finished processing batch.")

# Hàm xử lý từng video
def process_video(folder, video_file, index_local, time_per_segment, case_id, duration, total_frames, gpu_id):
    logging.info(f"Process {current_process().name} started processing video segment {index_local} on GPU {gpu_id}")
    frame_count = 0
    cap = cv2.VideoCapture(video_file, cv2.CAP_FFMPEG)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_rate = time_per_frame_global * fps 
    
    frames_batch = []
    frame_indices = []
    batch_size = 16  # Bạn có thể điều chỉnh batch size tùy theo nhu cầu

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % frame_rate == 0:
            frames_batch.append(frame)
            frame_indices.append(frame_count)
            if len(frames_batch) == batch_size:
                process_batch(frames_batch, frame_indices, folder, video_file, index_local, time_per_segment, case_id, duration, total_frames, gpu_id)
                frames_batch = []
                frame_indices = []
    if frames_batch:
        process_batch(frames_batch, frame_indices, folder, video_file, index_local, time_per_segment, case_id, duration, total_frames, gpu_id)
    cap.release()
    logging.info(f"Process {current_process().name} finished processing video segment {index_local} on GPU {gpu_id}")

# Hàm nhóm kết quả JSON
def groupJson(folder, video_file, count_thread, case_id):
    final_result = {
        "time": []
    }
    duration = getduration(video_file)
    time_per_segment = duration / count_thread

    list_stt = []
    results_path = f"results/{case_id}/{folder}"
    os.makedirs(results_path, exist_ok=True)
    for path in os.listdir(results_path):
        if os.path.isfile(os.path.join(results_path, path)):
            stt = int(path.split(".")[0])
            list_stt.append(stt)
           
    list_stt = sorted(list_stt)
    max_age = 0 
    sum_gender = 0 
    count_face = 0 
    for stt in list_stt:
        with open(f"{results_path}/{stt}.json", 'r') as file:
            data = json.load(file)
            if len(data) > 0:
                data = data[0]
                if int(data['age']) > max_age:
                    max_age = int(data['age'])
                sum_gender += int(data['gender'])
                count_face += 1 
                for duration_range in data["duration_exist"]:
                    final_result["time"].append([
                        duration_range[0] + stt * time_per_segment,
                        duration_range[1] + stt * time_per_segment
                    ])

    final_result['age'] = max_age
    if count_face > 0 : 
        final_result['gender'] = sum_gender / count_face
        
        facematches.update_many(
            {"case_id": case_id},
            {
                "$set": {
                    "gender": sum_gender / count_face,
                    "age": max_age,
                }
            }
        )

    os.makedirs(f"final_result/{case_id}/{folder}", exist_ok=True)
    with open(f"final_result/{case_id}/{folder}/final_result.json", 'w') as f:
        json.dump(final_result, f, indent=4)
    
    final_result["file"] = folder 
    final_result["id"] = str(uuid.uuid4())
    final_result["case_id"] = case_id
    final_result["createdAt"] = current_date()
    final_result["updatedAt"] = current_date()
    new_arr = []

    for time_range in final_result["time"]:
        new_arr.append({
            "start": time_range[0],
            "end": time_range[1],
            "frame": int((time_range[1] - time_range[0]) // (duration / getduration(video_file)))
        })
    final_result["time"] = new_arr
    appearances.insert_one(final_result)

# Hàm tạo video xuất hiện
def create_video_apperance(case_id, thread_count):
    list_img = []
    output_base_dir = f"{dir_project}/outputs/{case_id}"
    for dir in os.listdir(output_base_dir):
        dir_full = os.path.join(output_base_dir, dir)
        for i in range(thread_count):
            folder_count = i 
            dir_full_new = os.path.join(dir_full, str(folder_count))
            if os.path.exists(dir_full_new):
                for path in os.listdir(dir_full_new):
                    full_path = os.path.join(dir_full_new, path)
                    if os.path.isfile(full_path):
                        list_img.append(full_path)
    img_array = []
    size = None
    for filename in list_img:
        img = cv2.imread(filename)
        if img is not None:
            height, width, layers = img.shape
            size = (width, height)
            img_array.append(img)

    if not img_array:
        logging.warning("Không có hình ảnh để tạo video.")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video_output_dir = f"{dir_project}/video_apperance/{case_id}"
    os.makedirs(video_output_dir, exist_ok=True)
    out = cv2.VideoWriter(f"{video_output_dir}/video.mp4", fourcc, 5.0, size)

    for img in img_array:
        out.write(img)
    out.release()
    videos.insert_one({
        "id": str(uuid.uuid4()),
        "case_id": case_id,
        "path": f"{video_output_dir}/video.mp4",
    })

# Hàm cắt video thành các phân đoạn
def trimvideo(folder, videofile, count_thread, case_id):
    duration = getduration(videofile)
    time_per_segment = duration / count_thread
    os.makedirs(f"videos/{case_id}/{folder}", exist_ok=True)
    for i in range(count_thread):
        command = f"ffmpeg -i {videofile} -ss {time_per_segment*i} -t {time_per_segment} -c:v copy -c:a copy videos/{case_id}/{folder}/{i}.mp4 -y"
        subprocess.run(command, shell=True, check=True)

# Hàm xử lý trong từng tiến trình
def worker_process(gpu_id, folder, video_file, index_local, time_per_segment, case_id, duration, total_frames):
    # Initialize logging in the child process
    setup_logging()
    logging.info(f"Process {current_process().name} started with GPU ID: {gpu_id}")
    
    # Set the environment variable for CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    logging.info(f"Process {current_process().name} setting CUDA_VISIBLE_DEVICES to {gpu_id}")
    
    # Initialize models inside the worker process
    if gpu_id >= 0:
        device_str = f'cuda:0'  # After setting CUDA_VISIBLE_DEVICES, the assigned GPU is visible as cuda:0
        providers = ['CUDAExecutionProvider']
    else:
        device_str = 'cpu'
        providers = []
    
    logging.info(f"Process {current_process().name} initializing FaceAnalysis on device {device_str}")
    face_analysis = FaceAnalysis('buffalo_l', providers=providers)
    face_analysis.prepare(ctx_id=0, det_size=(640, 640))  # ctx_id=0 because CUDA_VISIBLE_DEVICES is set
    model = model_zoo.get_model('/home/poc4a5000/.insightface/models/buffalo_l/det_10g.onnx')
    model.prepare(ctx_id=0, det_size=(640, 640))
    logging.info(f"Process {current_process().name} initialized models on GPU {gpu_id}")
    
    # Start processing the video
    process_video(folder, video_file, index_local, time_per_segment, case_id, duration, total_frames, gpu_id)
    
    logging.info(f"Process {current_process().name} on GPU {gpu_id} finished processing.")

# Hàm xử lý nhiều tệp sử dụng multiprocessing
def handle_multiplefile(listfile, thread, case_id):
    processes = []
    
    for idx, file in enumerate(listfile):
        folder_name = os.path.splitext(os.path.basename(file))[0]
        duration = getduration(file)
        time_per_segment = duration / thread
        total_frames = int(cv2.VideoCapture(file).get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Cắt video thành các phân đoạn
        trimvideo(folder_name, file, thread, case_id)
        
        video_files = [f"videos/{case_id}/{folder_name}/{i}.mp4" for i in range(thread)]
        
        for i, vf in enumerate(video_files):
            gpu_id = video_gpu_ids[(idx * thread + i) % len(video_gpu_ids)]  # Distribute GPUs in a round-robin fashion
            logging.info(f"Assigning GPU {gpu_id} to process video segment {i} of file {folder_name}")
            p = Process(target=worker_process, args=(gpu_id, folder_name, vf, i, time_per_segment, case_id, duration, total_frames))
            p.start()
            processes.append(p)
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    # Group JSON results and create appearance videos
    for file in listfile:
        folder_name = os.path.splitext(os.path.basename(file))[0]
        groupJson(folder_name, file, thread, case_id)
        create_video_apperance(case_id, thread)
    
    # Delete the videos directory after processing
    for file in listfile:
        file_name = os.path.splitext(os.path.basename(file))[0]
        subprocess.run(f"rm -rf videos/{case_id}/{file_name}", shell=True, check=True)

# Hàm chính xử lý
def handle_main(case_id, tracking_folder, target_folder):
    # Xử lý mục tiêu ban đầu trên GPU 0
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
    
    # Danh sách các tệp cần xử lý
    list_file = [
        os.path.join(tracking_folder, f) 
        for f in os.listdir(tracking_folder) 
        if os.path.isfile(os.path.join(tracking_folder, f))
    ]
    handle_multiplefile(list_file, 50, case_id)
    
    # Tạo thư mục video xuất hiện
    os.makedirs(f"./video_apperance/{case_id}", exist_ok=True)

# Thiết lập Flask API
api = Flask(__name__)

@api.route('/analyst', methods=["POST"])
def analyst():
    case_id = request.json.get('case_id')
    tracking_folder = request.json.get('tracking_folder')
    target_folder = request.json.get('target_folder')
    
    if not all([case_id, tracking_folder, target_folder]):
        return jsonify({"error": "Missing parameters."}), 400

    # Xóa các bản ghi hiện tại
    myquery = { "case_id": case_id }
    facematches.delete_many(myquery)
    appearances.delete_many(myquery)
    targets.delete_many(myquery)
    videos.delete_many(myquery)

    # Thêm mục tiêu mới
    targets.insert_one({
        "id": str(uuid.uuid4()),
        "folder": target_folder,
        "case_id": case_id
    })

    handle_main(case_id, tracking_folder, target_folder)
    return jsonify({"data": "ok"})

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    
    # Initialize logging in the main process
    setup_logging()
    logging.info("Main process started.")
    
    # Initialize app_recognize on GPU 0
    app_recognize = FaceAnalysis('buffalo_l', providers=['CUDAExecutionProvider'])
    app_recognize.prepare(ctx_id=app_gpu_id, det_thresh=0.3, det_size=(640, 640))
    logging.info("app_recognize initialized on GPU 0")
    
    # Start the Flask app
    api.run(debug=True, port=5235, host='0.0.0.0')
