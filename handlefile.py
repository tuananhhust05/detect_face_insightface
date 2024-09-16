import json

list_result = []
# Open and read the JSON file
with open('data.json', 'r') as file:
    data = json.load(file)
    for em in data:
        frame_rate = em["frame_rate"] 
        time_per_frame = em["duration"] / em["frame_count"]
        list_time_exist = []
        duration_exist = []
        list_frame = em["frames"]
        print(list_frame)
        print("so frame", len(list_frame))
        for i in range(len(list_frame)-1):
           if(list_frame[i] == frame_rate):
              duration_exist.append(0)
           duration_exist.append(list_frame[i])
           if( (list_frame[i + 1] - list_frame[i]) > frame_rate):
               list_time_exist.append(duration_exist[0]*time_per_frame)
               list_time_exist.append(duration_exist[len(duration_exist) - 1] * time_per_frame)
               duration_exist = []
           else:
                if( i == len(list_frame)-2):
                    duration_exist.append(list_frame[i+1])
                    list_time_exist.append([duration_exist[0]*time_per_frame,duration_exist[len(duration_exist) - 1] * time_per_frame])
                    duration_exist = []
        list_result.append({
            'face':em['speaker'],
            'duration_exist':list_time_exist
        })
       

print(list_result)