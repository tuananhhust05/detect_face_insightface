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
from pinecone import Pinecone

import threading
# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

pc = Pinecone(api_key="be4036dc-2d41-4621-870d-f9c4e8958412")
index = pc.Index("detectface2")

weight_point = 0.4

# torch for handling vector 
def cosin(question, answer):
    question = torch.tensor(question).to(device)
    answer = torch.tensor(answer).to(device)
    cosine = torch.dot(question, answer) / (torch.norm(question) * torch.norm(answer))
    return cosine.item()  # Return as scalar

array_em = []
app = FaceAnalysis('buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))  # Ensure InsightFace uses GPU
list_result = []

def extract_frames(video_file):
    frame_count = 0
    frame_rate = 30  # default 1s with 30 frames
    audio = MP4(video_file)
    duration = audio.info.length
    print("duration", duration)
    cap = cv2.VideoCapture(video_file)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        print("frame_count", frame_count)

        if frame_count % frame_rate == 0:
            # Sharpen and denoise the image
            sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            sharpen = cv2.filter2D(frame, 0, sharpen_kernel)
            frame = cv2.fastNlMeansDenoisingColored(sharpen, None, 10, 10, 7, 21)

            faces = app.get(frame)
            for face in faces:
                if face["det_score"] > 0.5:
                    embedding = torch.tensor(face['embedding']).to(device)  # Move embedding to GPU
                    search_result = index.query(
                        vector=embedding.tolist(),
                        top_k=1,
                        include_metadata=True,
                        include_values=True,
                        filter={"face": 0},
                    )
                    matches = search_result["matches"]

                    # print("matches", matches)
                    if len(matches) > 0 and matches[0]['score'] > weight_point:
                        if len(array_em) == 0:
                            array_em.append({
                                "speaker": 0,
                                "frames": [frame_count],
                            })
                        else:
                            array_em[0]["frames"].append(frame_count)

                        try:
                            bbox = [int(b) for b in face['bbox']]
                            filename = f"{frame_count}_0_face.jpg"
                            cv2.imwrite(f'./faces/{filename}', frame[bbox[1]:bbox[3], bbox[0]:bbox[2], ::-1])
                            cv2.imwrite(f'./outputs/{filename}', frame)
                        except Exception as e:
                            print(f"Error saving frame: {e}")

    cap.release()
    print("End video")

def process_videos(video_files):
    threads = []
    for i, video_file in enumerate(video_files):
        t = threading.Thread(target=extract_frames, args=(video_file,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    print("Processing complete")

# Run with  GPU
video_files = [f"videos/{i}.mp4" for i in range(32)]  # Example video file list
start_time = time.time()
process_videos(video_files)
end_time = time.time()
print(f"Total execution time: {end_time - start_time}")
