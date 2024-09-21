from mutagen.mp4 import MP4
import os
import json
def groupJson(count_thread):
    final_result = []
    audio = MP4("../video.mp4")
    duration = audio.info.length
    time_per_segment = duration / count_thread
    print("duration",time_per_segment, duration)
    list_stt = []
    for path in os.listdir("results"):
        if os.path.isfile(os.path.join("results", path)):
            stt = int(path.split(".")[0])
            list_stt.append(stt)
           
    list_stt=sorted(list_stt)
    for stt in list_stt:
        with open(f"results/{stt}.json", 'r') as file:
           data = json.load(file)
           print(data)
           if(len(data) > 0):
                data = data[0]
                for duration in data["duration_exist"]:
                    final_result.append([duration[0] + stt * time_per_segment,duration[1] + stt * time_per_segment])
           print(f"Result after file {stt}",final_result )
    with open(f"final_result.json", 'w') as f:
        json.dump(final_result, f, indent=4)
        print("End video") 
groupJson(40)