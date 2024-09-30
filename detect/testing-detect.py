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
from itertools import cycle

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

# Phát hiện số lượng GPU có sẵn
num_gpus = torch.cuda.device_count()
gpu_ids = list(range(num_gpus))
if num_gpus > 0:
    logging.info(f"Number of GPUs available: {num_gpus}")
else:
    logging.warning("No GPUs detected. Using CPU.")
    gpu_ids = [-1]  # CPU

# Tạo một vòng lặp tuần tự các GPU IDs
gpu_cycle = cycle(gpu_ids)

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
def extract_frames_batch(folder, video_file, index_local, time_per_segment, case_id, ctx_id, batch_size=16):
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
                process_batch(frames_batch, frame_indices, folder, video_file, index_local, time_per_segment, case_id, array_em_result, list_result_ele, duration, total_frames, ctx_id)
                frames_batch = []
                frame_indices = []
    if frames_batch:
        process_batch(frames_batch, frame_indices, folder, video_file, index_local, time_per_segment, case_id, array_em_result, list_result_ele, duration, total_frames, ctx_id)
    cap.release()

def process_batch(frames_batch, frame_indices, folder, video_file, index_local, time_per_segment, case_id, array_em_result, list_result_ele, duration, total_frames, ctx_id):
    try:
        # Khởi tạo FaceAnalysis và model trong hàm xử lý để gán cho GPU cụ thể
        app = FaceAnalysis('buffalo_l', providers=['CUDAExecutionProvider'] if ctx_id >=0 else [])
        app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        app_recognize = FaceAnalysis('buffalo_l', providers=['CUDAExecutionProvider'] if ctx_id >=0 else [])
        app_recognize.prepare(ctx_id=ctx_id, det_thresh=0.3, det_size=(640, 640))
        model = model_zoo.get_model('/home/poc4a5000/.insightface/models/buffalo_l/det_10g.onnx')
        model.prepare(ctx_id=ctx_id, det_size=(640, 640))
        
        with torch.cuda.amp.autocast(enabled=(ctx_id >=0)):
            gpu_frames = [torch.tensor(frame).permute(2, 0, 1).float().to(ctx_id) if ctx_id >=0 else torch.tensor(frame).permute(2, 0, 1).float() for frame in frames_batch]
            # Giả định rằng model.detect có thể xử lý batch, nếu không cần xử lý từng frame một
            detections = [model.detect(frame, input_size=(640, 640)) for frame in gpu_frames]
        
        for frame, frame_count, detection in zip(frames_batch, frame_indices, detections):
            if detection and len(detection) > 0:
                # Xử lý tiếp theo tương tự như trong hàm gốc
                sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                sharpen = cv2.filter2D(frame, -1, sharpen_kernel)
                denoised = cv2.fastNlMeansDenoisingColored(sharpen, None, 10, 10, 7, 21)
                faces = app.get(denoised)

                sum_age = 0 
                sum_gender = 0 
                count_face = 0 

                for face in faces:
                    if face["det_score"] > 0.5:
                        embedding = torch.tensor(face['embedding']).to(ctx_id)
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

                            if not array_em_result:
                                array_em_result.append({
                                    "speaker": 0,
                                    "gender": int(face['gender']),
                                    "age": int(face['age']),
                                    "frames": [frame_count],
                                })
                            else:
                                array_em_result[0]["age"] = sum_age // count_face 
                                array_em_result[0]["gender"] = sum_gender // count_face 
                                array_em_result[0]["frames"].append(frame_count)

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

    except Exception as e:
        logging.error(f"Error processing batch at frame {frame_count}: {e}")
    finally:
        # Sau khi xử lý batch, có thể giải phóng bộ nhớ GPU nếu cần
        del gpu_frames
        torch.cuda.empty_cache()

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

# Hàm xử lý video với giới hạn số luồng
def process_videos(folder, video_file_origin, count_thread, case_id, ctx_ids):
    duration = getduration(video_file_origin)
    time_per_segment = duration / count_thread

    trimvideo(folder, video_file_origin, count_thread, case_id)

    video_files = [f"videos/{case_id}/{folder}/{i}.mp4" for i in range(count_thread)]  
    with ThreadPoolExecutor(max_workers=num_gpus) as executor:
        futures = [
            executor.submit(extract_frames_batch, folder, vf, i, time_per_segment, case_id, next(ctx_ids))
            for i, vf in enumerate(video_files)
        ]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                logging.error(f"Error in extract_frames_batch: {exc}")

    groupJson(folder, video_file_origin, count_thread, case_id)
    create_video_apperance(case_id, count_thread)

# Hàm xử lý nhiều tệp
def handle_multiplefile(listfile, thread, case_id):
    ctx_ids = cycle(gpu_ids) if num_gpus >0 else cycle([-1])
    with ThreadPoolExecutor(max_workers=num_gpus) as executor:
        futures = [
            executor.submit(process_videos, os.path.splitext(os.path.basename(file))[0], file, 50, case_id, ctx_ids)
            for file in listfile
        ]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                logging.error(f"Error processing file: {exc}")
    # Xóa thư mục videos sau khi xử lý
    for file in listfile:
        file_name = os.path.splitext(os.path.basename(file))[0]
        subprocess.run(f"rm -rf videos/{case_id}/{file_name}", shell=True, check=True)

# Hàm chính xử lý
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
