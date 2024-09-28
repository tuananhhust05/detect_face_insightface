import datetime
import numpy as np
import os
import cv2
import insightface
import torch
import json
import time
from pinecone import Pinecone
import subprocess
import threading

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = model_zoo.get_model('/home/poc4a5000/.insightface/models/buffalo_l/det_10g.onnx')
model.prepare(ctx_id=0, det_size=(640, 640))

def get_duration(file):
        data = cv2.VideoCapture(file)
        frames = data.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = data.get(cv2.CAP_PROP_FPS)
        data.release()
        if fps > 0 :
            return round(frames / fps)
        else:
            return 0 

class VideoProcessor:
    def __init__(self, api_key, index_name, weight_point=0.4):
        self.pc = Pinecone(api_key=api_key)
        self.index = self.pc.Index(index_name)
        self.weight_point = weight_point

    def get_duration(self, file):
        data = cv2.VideoCapture(file)
        frames = data.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = data.get(cv2.CAP_PROP_FPS)
        data.release()
        if fps > 0 :
            return round(frames / fps)
        else:
            return 0 

    def trim_video(self, folder, videofile, count_thread):
        duration = self.get_duration(videofile)
        time_per_segment = duration / count_thread
        for i in range(count_thread):
            command = f"ffmpeg -i {videofile} -ss {time_per_segment * i} -t {time_per_segment} -c:v copy -c:a copy videos/{folder}/{i}.mp4 -y"
            subprocess.run(command, shell=True, check=True)

    def process_videos(self, folder, video_file_origin, count_thread):
        self.trim_video(folder, video_file_origin, count_thread)
        video_files = [f"videos/{folder}/{i}.mp4" for i in range(count_thread)]
        threads = []

        for i, video_file in enumerate(video_files):
            extractor = FrameExtractor(video_file, i, folder, self.index, self.weight_point)
            t = threading.Thread(target=extractor.extract_frames)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

class FaceAnalyzer:
    def __init__(self):
        self.app = insightface.app.FaceAnalysis('buffalo_l', providers=['CUDAExecutionProvider'])
        self.app.prepare(ctx_id=0 if device.type == 'cuda' else -1, det_size=(640, 640))
        self.model = insightface.model_zoo.get_model('/home/poc4a5000/.insightface/models/buffalo_l/det_10g.onnx')
        self.model.prepare(ctx_id=0, det_size=(640, 640))
    def analyze(self, frame):
        return self.app.get(frame)
    
    def detect(self, frame, input_size):
        return self.model.detect(frame,input_size)
    
class FrameExtractor:
    def __init__(self, video_file, index_local, folder, index, weight_point):
        self.video_file = video_file
        self.index_local = index_local
        self.folder = folder
        self.index = index
        self.weight_point = weight_point
        self.face_analyzer = FaceAnalyzer()

    def extract_frames(self):
        array_em_result = []
        frame_count = 0
        frame_rate = 60
        duration = get_duration(self.video_file)
        cap = cv2.VideoCapture(self.video_file)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % frame_rate == 0:
                facechecks = model.detect(frame,input_size=(640, 640))
                flagDetect = False
                if(len(facechecks) > 0):
                    if(len(facechecks[0]) > 0):
                        flagDetect = True
                if flagDetect == True:
                    self.process_faces(frame, frame_count, duration, total_frames, array_em_result)

        self.save_results(array_em_result, duration, frame_count)

    def process_faces(self, frame,  frame_count, duration, total_frames, array_em_result):
        faces = self.face_analyzer.analyze(frame)
        sum_age = 0 
        sum_gender = 0 
        count_face = 0 
        for face in faces:
            if face["det_score"] > 0.5:
                embedding = torch.tensor(face['embedding']).to(device)
                search_result = self.index.query(vector=embedding.tolist(), top_k=1, include_metadata=True, include_values=True, filter={"face": 0})
                matches = search_result["matches"]

                if matches and matches[0]['score'] > self.weight_point:
                    count_face = count_face + 1 
                    sum_age = sum_age + int(face['age'])
                    sum_gender = sum_gender + int(face['gender'])
                    if not array_em_result:
                        array_em_result.append({
                                    "speaker": 0,  
                                    "gender":int(face['gender']),
                                    "age":int(face['age']),
                                    "frames": [frame_count]}
                                )
                    else:
                        array_em_result[0]["age"] = sum_age // count_face 
                        array_em_result[0]["gender"] = sum_gender // count_face 
                        array_em_result[0]["frames"].append(frame_count)

                    self.save_face_image(frame, face, frame_count)

    def save_face_image(self, frame, face, frame_count):
        bbox = [int(b) for b in face['bbox']]
        filename = f"{frame_count}_0_face.jpg"
        folder_path = f"./faces/{self.folder}/{self.index_local}"
        os.makedirs(folder_path, exist_ok=True)
        cv2.imwrite(f'{folder_path}/{filename}', frame[bbox[1]:bbox[3], bbox[0]:bbox[2], ::-1])
        cv2.imwrite(f'./outputs/{self.folder}/{self.index_local}/{filename}', frame)


    def save_results(self, array_em_result, duration, frame_count):
        for ele in array_em_result:
            ele["frame_count"] = frame_count
            ele["duration"] = duration
            ele["frame_rate"] = 60
        
        with open(f"datas/{self.folder}/{self.index_local}.json", 'w') as f:
            json.dump(array_em_result, f, indent=4)


class JsonHandler:
    @staticmethod
    def group_json(folder, video_file, count_thread):
        final_result = []
        duration = get_duration(video_file)
        time_per_segment = duration / count_thread

        list_stt = sorted(int(path.split(".")[0]) for path in os.listdir(f"results/{folder}") if os.path.isfile(os.path.join(f"results/{folder}", path)))
        
        max_age = 0 
        sum_gender = 0 
        count_face = 0 
        for stt in list_stt:
            with open(f"results/{folder}/{stt}.json", 'r') as file:
                data = json.load(file)
                if(len(data) > 0):
                    data = data[0]
                    if( int(data['age']) > max_age ):
                        max_age = int(data['age'])
                    sum_gender = sum_gender + int(data['gender'])
                    count_face = count_face + 1 
                    for duration in data["duration_exist"]:
                        final_result["time"].append([duration[0] + stt * time_per_segment,duration[1] + stt * time_per_segment])

        with open(f"final_result/{folder}/final_result.json", 'w') as f:
            json.dump(final_result, f, indent=4)

def handle_multiple_files(listfile, count_thread):
    for file in listfile:
        file_name = os.path.splitext(os.path.basename(file))[0]
        if not os.path.exists(f"./faces/{file_name}"):
            os.makedirs(f"./faces/{file_name}")
        
        if not os.path.exists(f"./outputs/{file_name}"):
            os.makedirs(f"./outputs/{file_name}")
        
        if not os.path.exists(f"./videos/{file_name}"):
            os.makedirs(f"./videos/{file_name}")
        
        if not os.path.exists(f"./datas/{file_name}"):
            os.makedirs(f"./datas/{file_name}")
       
        if not os.path.exists(f"./results/{file_name}"):
            os.makedirs(f"./results/{file_name}")
       
        if not os.path.exists(f"./final_result/{file_name}"):
            os.makedirs(f"./final_result/{file_name}")

        video_processor = VideoProcessor(api_key="6bebb6ba-195f-471e-bb60-e0209bd5c697", index_name="detectcamera")
        video_processor.process_videos(file_name, file, count_thread)

# Run with GPU
dir_path = r'/home/poc4a5000/facesx'
list_file = [os.path.join(dir_path, path) for path in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, path))]

start_time = time.time()
print("Start ......", str(start_time))

# handle_multiple_files(list_file, 50)
handle_multiple_files(["input/video8p.mp4"],50)

end_time = time.time()
print(f"Total execution time: {end_time - start_time}")
