import cv2
import os
# from deblur import apply_blur
import numpy as np
def extract_frames(video_file):
    cap = cv2.VideoCapture(video_file)
    
    frame_rate = 2  # Desired frame rate (1 frame every 0.5 seconds)
    frame_count = 0
    
    # Get the video file's name without extension
    video_name = os.path.splitext(os.path.basename(video_file))[0]
    
    # Create an output folder with a name corresponding to the video
    output_directory = f"{video_name}_frames"
    os.makedirs(output_directory, exist_ok=True)
    
    count = 0
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpen = cv2.filter2D(frame, 0 , sharpen_kernel)

        frame = cv2.fastNlMeansDenoisingColored(sharpen,None,10,10,7,21)
        # frame = apply_blur(frame, kernel_size, sigma)
        output_file = f"{output_directory}/frame_{frame_count}.jpg"
        cv2.imwrite(output_file, frame)
        count = count + 1
        print(f"Frame {frame_count} has been extracted and saved as {output_file}")
    
    cap.release()
    cv2.destroyAllWindows()

extract_frames('video.mp4')