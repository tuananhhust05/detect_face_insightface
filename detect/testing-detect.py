import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import logging
import uuid
import json
from flask import Flask, jsonify, request
import pymongo
import cv2
import torch
from insightface.app import FaceAnalysis
from insightface.model_zoo import model_zoo
from pinecone import Pinecone

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize MongoDB
myclient = pymongo.MongoClient("mongodb://root:facex@192.168.50.10:27018")
mydb = myclient["faceX"]
facematches = mydb["facematches"]
appearances = mydb["appearances"]
targets = mydb["targets"]
videos = mydb["videos"]

# Project directory
dir_project = "/home/poc4a5000/detect/detect"

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# Initialize Pinecone
pc = Pinecone(api_key="YOUR_PINECONE_API_KEY")
index = pc.Index("detectcamera")

weight_point = 0.4
time_per_frame_global = 2
ctx_id = 0 if device.type == 'cuda' else -1

# Initialize Face Analysis Models
app = FaceAnalysis('buffalo_l', providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=ctx_id, det_size=(640, 640))
app_recognize = FaceAnalysis('buffalo_l', providers=['CUDAExecutionProvider'])
app_recognize.prepare(ctx_id=ctx_id, det_thresh=0.3, det_size=(640, 640))
model = model_zoo.get_model('/home/poc4a5000/.insightface/models/buffalo_l/det_10g.onnx')
model.prepare(ctx_id=ctx_id, det_size=(640, 640))

# Define utility functions (getduration, cosin, current_date, etc.) here
# ... [Omitted for brevity]

def extract_frames(folder, video_file, index_local, time_per_segment, case_id):
    array_em_result = []
    list_result_ele = []
    frame_count = 0
    duration = getduration(video_file)
    cap = cv2.VideoCapture(video_file, cv2.CAP_FFMPEG)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_rate = time_per_frame_global * fps
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % frame_rate == 0:
            try:
                gpu_frame = cv2.cuda_GpuMat()
                gpu_frame.upload(frame)
                facechecks = model.detect(gpu_frame, input_size=(640, 640))
                flagDetect = len(facechecks) > 0 and len(facechecks[0]) > 0

                if flagDetect:
                    sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                    sharpen = cv2.cuda.filter2D(gpu_frame, 0, sharpen_kernel)
                    gpu_frame = cv2.cuda.fastNlMeansDenoisingColored(sharpen, None, 10, 10, 7, 21)
                    faces = app.get(gpu_frame)

                    sum_age = 0
                    sum_gender = 0
                    count_face = 0

                    for face in faces:
                        if face["det_score"] > 0.5:
                            embedding = torch.tensor(face['embedding']).to(device)
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
                                    "id": str(uuid.uuid4()),
                                    "case_id": case_id,
                                    "similarity_face": str(matches[0]['score']),
                                    "gender": int(face['gender']),
                                    "age": int(face['age']),
                                    "time_invideo": text,
                                    "proofImage": f'/home/poc4a5000/detect/detect/faces/{case_id}/{folder}/{index_local}/{filename}',
                                    "url": f'/home/poc4a5000/detect/detect/faces/{case_id}/{folder}/{index_local}/{filename}',
                                    "createdAt": current_date(),
                                    "updatedAt": current_date(),
                                    "file": folder
                                }
                                facematches.insert_one(mydict)

            except Exception as e:
                logging.error(f"Error processing frame {frame_count}: {e}")

    # Post-processing: JSON handling
    # ... [Omitted for brevity]

    cap.release()

def process_videos(folder, video_file_origin, count_thread, case_id):
    duration = getduration(video_file_origin)
    time_per_segment = duration / count_thread

    trimvideo(folder, video_file_origin, count_thread, case_id)

    video_files = [f"videos/{case_id}/{folder}/{i}.mp4" for i in range(count_thread)]
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(extract_frames, folder, vf, i, time_per_segment, case_id) 
                   for i, vf in enumerate(video_files)]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                logging.error(f"Error in extract_frames: {exc}")

    groupJson(folder, video_file_origin, count_thread, case_id)
    create_video_apperance(case_id, count_thread)

# Flask API Setup
api = Flask(__name__)

@api.route('/analyst', methods=["POST"])
def analyst():
    case_id = request.json['case_id']
    tracking_folder = request.json['tracking_folder']
    target_folder = request.json['target_folder']
    
    # Clear existing records
    myquery = { "case_id": case_id }
    facematches.delete_many(myquery)
    appearances.delete_many(myquery)
    targets.delete_many(myquery)
    videos.delete_many(myquery)

    # Insert target
    targets.insert_one({
        "id": str(uuid.uuid4()),
        "folder": target_folder,
        "case_id": case_id
    })

    handle_main(case_id, tracking_folder, target_folder)
    return jsonify({"data": "ok"})

def handle_main(case_id, tracking_folder, target_folder):
    flag_target_folder = True
    for path in os.listdir(target_folder):
        if flag_target_folder and os.path.isfile(os.path.join(target_folder, path)):
            full_path = os.path.join(target_folder, path)
            img = cv2.imread(full_path)
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
    
    list_file = [os.path.join(tracking_folder, f) for f in os.listdir(tracking_folder) 
                 if os.path.isfile(os.path.join(tracking_folder, f))]
    handle_multiplefile(list_file, max_threads=8, case_id=case_id)

    os.makedirs(f"./video_apperance/{case_id}", exist_ok=True)

if __name__ == '__main__':
    api.run(debug=True, port=5234, host='0.0.0.0')
