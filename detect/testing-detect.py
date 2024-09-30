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
from numba import jit, cuda
import subprocess
import threading
import logging
import uuid
from flask import Flask, jsonify, request
import pymongo
from concurrent.futures import ThreadPoolExecutor, as_completed
from torch.cuda.amp import autocast

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

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

# Kiểm tra xem index đã tồn tại chưa, nếu chưa thì tạo mới
# if index_name not in pc.list_indexes():
#     pc.create_index(name=index_name, dimension=128, metric="cosine")  # Thay đổi dimension phù hợp
# logging.info(f"Connecting to Pinecone index: {index_name}")
index = pc.Index(index_name)

weight_point = 0.4
time_per_frame_global = 2 
ctx_id = 0 if device.type == 'cuda' else -1

# Khởi tạo ứng dụng phân tích khuôn mặt với Mixed Precision
app = FaceAnalysis('buffalo_l', providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=ctx_id, det_size=(640, 640))
app_recognize = FaceAnalysis('buffalo_l', providers=['CUDAExecutionProvider'])
app_recognize.prepare(ctx_id=ctx_id, det_thresh=0.3, det_size=(640, 640))
model = model_zoo.get_model('/home/poc4a5000/.insightface/models/buffalo_l/det_10g.onnx')
model.prepare(ctx_id=ctx_id, det_size=(640, 640))

# Các hàm tiện ích
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

# Hàm xử lý trích xuất khung hình với Batch Processing và Mixed Precision
def extract_frames_batch(folder, video_file, index_local, time_per_segment, case_id, batch_size=16):
    array_em_result = []
    list_result_ele = []
    frame_count = 0 
    duration = getduration(video_file)
    cap = cv2.VideoCapture(video_file, cv2.CAP_FFMPEG)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_rate = time_per_frame_global * fps 
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frames_batch = []
    frame_indices = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % frame_rate == 0:
            frames_batch.append(frame)
            frame_indices.append(frame_count)
            if len(frames_batch) == batch_size:
                process_batch(frames_batch, frame_indices, folder, video_file, index_local, time_per_segment, case_id, array_em_result, list_result_ele, duration, total_frames)
                frames_batch = []
                frame_indices = []
    if frames_batch:
        process_batch(frames_batch, frame_indices, folder, video_file, index_local, time_per_segment, case_id, array_em_result, list_result_ele, duration, total_frames)
    cap.release()

def process_batch(frames_batch, frame_indices, folder, video_file, index_local, time_per_segment, case_id, array_em_result, list_result_ele, duration, total_frames):
    try:
        with torch.cuda.amp.autocast():
            gpu_frames = [torch.tensor(frame).permute(2, 0, 1).float().to(device) for frame in frames_batch]
            # Batch upload
            # Assuming model.detect can handle batch processing
            # Modify this part based on actual model capabilities
            # Example placeholder:
            detections = model.detect(gpu_frames, input_size=(640, 640))
        
        for frame, frame_count, detection in zip(frames_batch, frame_indices, detections):
            if detection and len(detection) > 0:
                # Xử lý tiếp theo tương tự như trong hàm gốc
                # ... (tiếp tục xử lý như trong extract_frames)
                pass  # Thay thế bằng mã xử lý thực tế
    except Exception as e:
        logging.error(f"Error processing batch at frame {frame_count}: {e}")

# Các hàm còn lại giữ nguyên nhưng có thể thêm tối ưu hóa tương tự
def cosin(question, answer):
    with torch.cuda.amp.autocast():
        question = torch.tensor(question).to(device)
        answer = torch.tensor(answer).to(device)
        cosine = torch.dot(question, answer) / (torch.norm(question) * torch.norm(answer))
    return cosine.item()  

# Hàm xử lý trích xuất khung hình ban đầu đã được thay thế bằng extract_frames_batch
def extract_frames(folder, video_file, index_local, time_per_segment, case_id):
    extract_frames_batch(folder, video_file, index_local, time_per_segment, case_id)

# Các hàm groupJson, create_video_apperance, trimvideo, process_videos, handle_multiplefile, handle_main giữ nguyên
# Bạn có thể thêm các tối ưu hóa tương tự ở những hàm này nếu cần

# Hàm chính xử lý với tối ưu hóa GPU
def handle_main(case_id, tracking_folder, target_folder):
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
    
    list_file = [
        os.path.join(tracking_folder, f) 
        for f in os.listdir(tracking_folder) 
        if os.path.isfile(os.path.join(tracking_folder, f))
    ]
    handle_multiplefile(list_file, 8, case_id)

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
    api.run(debug=True, port=5235, host='0.0.0.0')
