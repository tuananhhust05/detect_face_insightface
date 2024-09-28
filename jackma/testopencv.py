import av
import cv2
import matplotlib.pyplot as plt

def read_video_with_cuda(video_path):
    # Create a VideoReader with CUDA support
    video_reader = cv2.cudacodec.createVideoReader(video_path)
    count = 0 
    while True:
        # Read a frame from the video
        ret, gpu_frame = video_reader.nextFrame()
        if not ret:
            break
        count = count + 1 
        print(count)
        # Download the frame to CPU for processing or display
        frame = gpu_frame.download()

        # # Display the frame
        # cv2.imshow('Video', frame)

        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    cv2.destroyAllWindows()

# Example use
video_path = '2.mp4'  # Replace with your actual video file path
read_video_with_cuda(video_path)

