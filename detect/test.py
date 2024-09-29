

import os 
import cv2 
from PIL import Image  

dir_project="."
def create_video_apperance(case_id,thread_count):
    cap = cv2.VideoCapture(0)

    list_img = []
    list_dir_file = os.listdir(f"{dir_project}/outputs/{case_id}")
    for dir in list_dir_file:
        dir_full = f"{dir_project}/outputs/{case_id}/{dir}"
        for i in range(thread_count):
            folder_count = i 
            dir_full_new = f"{dir_full}/{folder_count}"
            print(dir_full)
            if  os.path.exists(dir_full_new):
                for path in os.listdir(dir_full_new):
                    if os.path.isfile(os.path.join(dir_full_new, path)):
                        full_path = f"{dir_full_new}/{path}"
                        list_img.append(full_path)
    img_array = []
    for filename in list_img:
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    print(size)
    out = cv2.VideoWriter('video.mp4', fourcc, 5.0, size)

    for i in range(len(img_array)):
        out.write(img_array[i])

    out.release()
create_video_apperance("123456-12",50)