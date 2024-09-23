
import os
import cv2 


dir_path = r'/mnt/casehdd16tb/DataVideoHTC'
list_file = []
for path in os.listdir(dir_path):
    # check if current path is a file
    if os.path.isfile(os.path.join(dir_path, path)):
        full_path = f"{dir_path}/{path}"
        list_file.append(full_path)
print(list_file)


total = 0

def getduration(file):
    data = cv2.VideoCapture(file) 
    # count the number of frames 
    frames = data.get(cv2.CAP_PROP_FRAME_COUNT) 
    fps = data.get(cv2.CAP_PROP_FPS) 
    data.release()
    # calculate duration of the video 
    seconds = round(frames / fps) 
    return seconds

for file in list_file:
    print(getduration(file))
    total = total + getduration(file)
    # print(total)
