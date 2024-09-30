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

def extract_frames(folder, video_file, index_local, time_per_segment, case_id, gpu_id, start_frame, duration_total, total_frames_total):
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
        frame_rate = int(fps)
        process_interval = frame_rate * 5  # Process every 5 seconds
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            global_frame_count = int(start_frame + frame_count)

            if frame_count % process_interval == 0:
                print("Processing frame", global_frame_count)
                start_time = time.time()
                facechecks = model.detect(frame, input_size=(640, 640))
                detection_time = time.time() - start_time
                # print(f"Face detection took {detection_time:.2f} seconds")

                flagDetect = False
                if len(facechecks) > 0 and len(facechecks[0]) > 0:
                    flagDetect = True

                if flagDetect:
                    # print("Face detected...")
                    # Preprocessing
                    sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                    sharpen = cv2.filter2D(frame, -1, sharpen_kernel)
                    frame_denoised = cv2.fastNlMeansDenoisingColored(sharpen, None, 10, 10, 7, 21)

                    start_time = time.time()
                    faces = app.get(frame_denoised)
                    recognition_time = time.time() - start_time
                    # print(f"Face recognition took {recognition_time:.2f} seconds")

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
                                        "frames": [global_frame_count],
                                    })
                                    
                                else:
                                    if count_face > 0:
                                        array_em_result[0]["age"] = sum_age // count_face 
                                        array_em_result[0]["gender"] = sum_gender // count_face 
                                        array_em_result[0]["frames"].append(global_frame_count)

                                try:
                                    bbox = [int(b) for b in face['bbox']]
                                    filename = f"{global_frame_count}_0_face.jpg"
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
                                    time_per_frame = duration_total / total_frames_total
                                    text = global_frame_count * time_per_frame
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
        torch.cuda.empty_cache()
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
        final_result = {
            "time": []
        }
        duration = getduration(video_file)
        time_per_segment = duration / count_thread

        list_stt = []
        for path in os.listdir(f"results/{case_id}/{folder}"):
            if os.path.isfile(os.path.join(f"results/{case_id}/{folder}", path)):
                stt = int(os.path.splitext(path)[0])
                list_stt.append(stt)
               
        list_stt = sorted(list_stt)
        max_age = 0 
        sum_gender = 0 
        count_face = 0 
        for stt in list_stt:
            with open(f"results/{case_id}/{folder}/{stt}.json", 'r') as file:
               data = json.load(file)
               if len(data) > 0:
                    data = data[0]
                    if int(data['age']) > max_age:
                        max_age = int(data['age'])
                    sum_gender += int(data['gender'])
                    count_face += 1 
                    for duration in data["duration_exist"]:
                        final_result["time"].append([duration[0] + stt * time_per_segment, duration[1] + stt * time_per_segment])

        final_result['age'] = max_age
        if count_face > 0:
            final_result['gender'] = sum_gender / count_face

            facematches.update_many(
                {"case_id": case_id},
                {"$set": {
                    "gender": sum_gender / count_face,
                    "age": max_age,
                }}
            )

        with open(f"final_result/{case_id}/{folder}/final_result.json", 'w') as f:
            json.dump(final_result, f, indent=4)
        
        final_result["file"] = folder 
        final_result["id"] = str(uuid.uuid4())
        final_result["case_id"] = case_id
        final_result["createdAt"] = current_date()
        final_result["updatedAt"] = current_date()
        new_arr = []

        for time_entry in final_result["time"]:
           new_arr.append(
               {
                   "start": time_entry[0],
                   "end": time_entry[1],
                   "frame": (time_entry[1] - time_entry[0]) // time_per_frame_global
               }
           )
        final_result["time"] = new_arr
        appearances.insert_one(final_result)

    def create_video_apperance(case_id, thread_count):
        list_img = []
        list_dir_file = os.listdir(f"{dir_project}/outputs/{case_id}")
        for dir in list_dir_file:
            dir_full = f"{dir_project}/outputs/{case_id}/{dir}"
            for i in range(thread_count):
                folder_count = i 
                dir_full_new = f"{dir_full}/{folder_count}"
                if os.path.exists(dir_full_new):
                    for path in os.listdir(dir_full_new):
                        if os.path.isfile(os.path.join(dir_full_new, path)):
                            full_path = f"{dir_full_new}/{path}"
                            list_img.append(full_path)
        img_array = []
        size = (120, 120)
        for filename in list_img:
            img = cv2.imread(filename)
            if img is not None:
                height, width, layers = img.shape
                size_inter = (width, height)
                size = size_inter
                img_array.append(img)

        if not img_array:
            print("No images found for video appearance.")
            return

        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        os.makedirs(f"{dir_project}/video_apperance/{case_id}", exist_ok=True)
        out = cv2.VideoWriter(f"{dir_project}/video_apperance/{case_id}/video.mp4", fourcc, 5.0, size)

        for img in img_array:
            out.write(img)
        out.release()
        videos.insert_one({
            "id": str(uuid.uuid4()),
            "case_id": case_id,
            "path": f"{dir_project}/video_apperance/{case_id}/video.mp4",
        })

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
        max_processes = num_gpus  # Limit to one process per GPU

        for i, video_file in enumerate(video_files):
            gpu_id = gpu_ids[i % num_gpus]
            p = multiprocessing.Process(target=extract_frames, args=(folder, video_file, i, time_per_segment, case_id, gpu_id))
            processes.append(p)
            p.start()

            if len(processes) >= max_processes:
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
            os.makedirs(f"./faces/{case_id}/{file_name}", exist_ok=True)
            os.makedirs(f"./outputs/{case_id}/{file_name}", exist_ok=True)
            os.makedirs(f"./videos/{case_id}/{file_name}", exist_ok=True)
            os.makedirs(f"./datas/{case_id}/{file_name}", exist_ok=True)
            os.makedirs(f"./results/{case_id}/{file_name}", exist_ok=True)
            os.makedirs(f"./final_result/{case_id}/{file_name}", exist_ok=True)

            folder = file_name
            process_videos(folder, file, thread, case_id)
            subprocess.run(f"rm -rf videos/{case_name}", shell=True, check=True)

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
