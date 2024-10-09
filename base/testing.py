import cv2
import numpy as np
import os
from insightface.app import FaceAnalysis
from numpy.linalg import norm
import time

# Initialize InsightFace
app = FaceAnalysis('buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))  # Use GPU 0

def cosin(question, answer):
    cosine = np.dot(question, answer) / (norm(question) * norm(answer))
    return cosine

array_cosin = []
array_em = []
vectorFlag = [...]  # Your vectorFlag list

def extract_frames(video_file):
    frame_count = 0
    frame_rate = 60  # Every 60th frame
    cap = cv2.VideoCapture(video_file)
    
    # Get video name without extension
    video_name = os.path.splitext(os.path.basename(video_file))[0]
    
    # Create output directory
    output_directory = f"{video_name}_frames"
    os.makedirs(output_directory, exist_ok=True)
    
    # Define sharpen kernel
    sharpen_kernel = np.array([[-1, -1, -1], 
                               [-1,  9, -1], 
                               [-1, -1, -1]], dtype=np.float32)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        print("frame_count", frame_count)
        
        if frame_count % frame_rate == 0:
            # Upload frame to GPU
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(frame)
            
            # Apply sharpening filter on GPU
            gpu_sharpen = cv2.cuda.filter2D(gpu_frame, -1, sharpen_kernel)
            
            # Apply bilateral filter on GPU
            gpu_bilateral = cv2.cuda.createBilateralFilter(9, 75, 75)
            gpu_denoised = gpu_bilateral.apply(gpu_sharpen)
            
            # Download processed frame back to CPU
            processed_frame = gpu_denoised.download()
            
            # Save frame to disk
            output_path = os.path.join(output_directory, f"frame_{frame_count}.jpg")
            cv2.imwrite(output_path, processed_frame)
            
            # Detect faces
            faces = app.get(processed_frame)
            
            for face in faces:
                embedding = face.embedding
                array_em.append(embedding)
                
                # Calculate cosine similarity
                cosine_sim = cosin(vectorFlag, embedding)
                array_cosin.append(cosine_sim)
                
                # Optionally, upsert to Pinecone
                # index.upsert([...])

    cap.release()
    print("End video")

if __name__ == "__main__":
    start = time.time()
    extract_frames('video.mp4')
    end = time.time()
    print("Execution time", end - start)
