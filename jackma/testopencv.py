import av
import cv2
import matplotlib.pyplot as plt

def read_video_with_ffmpeg_cuda(video_path, frame_skip=2):
    container = av.open(video_path)
    for i, frame in enumerate(container.decode(video=0)):
        if i % frame_skip == 0:  # Skip frames based on frame_skip value
            img = frame.to_ndarray(format='bgr24')
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            print(i) 
    #            plt.imshow(img_rgb)
    #       plt.axis('off')
    #        plt.pause(0.001)

    plt.close()

video_path = '2.mp4'
read_video_with_ffmpeg_cuda(video_path)

