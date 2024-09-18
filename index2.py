
import datetime
import numpy as np
import os
import os.path as osp
import glob
import cv2
import insightface
import matplotlib.pyplot as plt 
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
from numpy.linalg import norm
import time 
from mutagen.mp4 import MP4
from deblur import apply_blur,wiener_filter,gaussian_kernel
from pinecone import Pinecone, ServerlessSpec
import uuid
import json 
import numba
from numba import jit
pc = Pinecone(api_key="be4036dc-2d41-4621-870d-f9c4e8958412")
index = pc.Index("detectface2")

weight_point = 0.4

def cosin(question,answer):
    cosine = np.dot(question,answer)/(norm(question)*norm(answer))
    return cosine

array_cosin = []
array_em = []
app = FaceAnalysis('buffalo_l')
app.prepare(ctx_id = 0, det_size=(640,640))
img = cv2.imread("./videotest_frames/frame_11.jpg") 
fig,axs = plt.subplots(1,6,figsize=(12,5))



# faces = app.get(img)
# for i,face in enumerate(faces):
#     bbox = face['bbox']
#     bbox = [int(b) for b in bbox]
#     filename = f"{i}test.jpg"
#     cv2.imwrite('./outputs/%s'%filename,img[bbox[1] : bbox[3], bbox[0]: bbox[2], ::-1])
#     array_em.append(face['embedding'])
# img2 = cv2.imread("./videotest_frames/frame_22.jpg")
# faces2 = app.get(img2)
# for i,face in enumerate(faces2):
#     bbox = face['bbox']
#     bbox = [int(b) for b in bbox]
#     filename = f"{i}test.jpg"
#     cv2.imwrite('./outputs/%s'%filename,img[bbox[1] : bbox[3], bbox[0]: bbox[2], ::-1])
#     for em in array_em:
#       cosin_value = cosin(em,face['embedding'])
#       array_cosin.append(cosin_value)
#   array_em.append(face['embedding'])

vectorFlag = [-1.58570838, -1.01461756, 1.93423057, 1.02658987, -0.262339085, -0.40912348, -0.701766789, -1.97888124, -0.815337896, 1.93691909, -1.06913912, 1.29624271, -1.87130582, -0.350940049, -0.859327316, 0.00685280561, -1.36580098, -0.177038938, -0.82324481, 0.0725998655, 1.19169629, -0.537574112, -0.143889934, 0.423788548, -1.28997815, -0.691303134, 0.9505108, 0.456822634, -1.21541953, 0.509877801, 2.54147744, -0.797174454, 1.07721353, 1.99444842, 0.294461638, -1.13751602, -0.443599463, -0.185655609, -0.419868261, -0.649171412, 1.68312, -0.449149638, 0.738450885, 0.084679395, -0.303134, 1.81445384, -0.214072466, -1.01161408, 1.42919338, 0.337690711, 0.214674398, 0.666396797, -1.48953962, 0.777798295, 1.02892601, -0.793158, -1.13800788, -0.118170679, -0.135574281, -0.156096846, -0.20306024, -0.931823611, -1.07792974, 3.12676215, 0.413814664, -1.11166072, -0.200057358, 0.512803137, -2.13515401, -0.18164517, 0.981290281, 0.821194232, -0.0990702361, -0.255505383, -0.0719047636, -1.47680557, 1.33777714, 0.427234232, -0.0795939863, 0.755731702, -0.702543199, -0.401173592, -0.204442799, -0.119734, -2.21403098, -0.613474071, -1.61753058, 0.538163722, 1.78350031, 0.0405711308, 0.1249865, -0.0482687838, -0.90767, 2.00961494, -2.39648175, -0.403967857, -0.0998002663, -0.896526814, -0.818, -2.11158776, 0.360288143, -0.552250266, 0.783427835, 0.417064667, 0.851778626, -0.211211234, -0.671403706, 0.537235439, 0.279551715, 0.111030832, -0.0719338506, 1.02374601, -2.37059975, 0.746250212, 0.741552472, -0.844312072, -0.229856282, -1.8302635, 0.25256744, -1.05965745, 0.963579297, -0.0908905342, 0.52214849, 0.661085129, -1.02702987, -1.15120721, 2.34170794, -0.354084194, 0.387622058, 0.514790297, -0.645891547, 1.09428704, -0.905988276, -0.446118385, 1.55914235, -1.23618448, 1.74633706, -0.835257769, 0.716828942, 1.04121554, -0.767461419, 0.150288, 0.52922684, 1.5611825, -0.0379444957, 0.0306240618, 0.87446022, 0.22342363, -0.524650097, 0.302835077, -1.76020181, 0.141529888, 0.718532562, 1.04413557, 0.308248878, 1.55928922, -0.261788, -1.39485192, 2.29168439, 0.693808079, 0.31667906, -0.603613496, -0.873401463, 0.45156917, 1.38169253, -1.55057073, 0.420952857, 1.40656209, -1.4385134, -1.1013155, -0.51657027, -1.1080544, -1.1898607, -0.0723044574, -0.896446526, -1.5482291, 1.58974469, 1.20911705, 2.09068298, -1.45776331, 0.588041067, 0.0471709818, -0.00350829959, 0.049234286, 0.269619405, 0.652528346, 0.0932857096, -0.156500012, 0.445772499, 1.96090984, 0.672240376, -1.65474117, -1.0729531, -0.346226931, 1.26495767, 0.186568022, 1.52603614, -0.89064914, 3.19054866, 2.07121396, 0.121653683, -0.40191105, 1.4116987, 0.0982992798, -0.917591572, -1.06331134, -0.0101665733, 0.677299678, 0.256419092, -0.364866585, 0.284340739, 0.658279181, -0.660404146, 0.110865377, 1.42601871, -0.546788752, 0.20716463, 1.0968523, 0.554207385, 3.52193141, 0.5833534, -0.301090777, -0.135285199, -2.47521949, 0.236523196, 1.28117204, -0.335194021, 0.155164704, -0.0307774469, 1.35680699, -1.14772224, 0.274384409, -1.20712543, 1.46988094, -0.367777497, 1.11583364, -0.700105309, 0.758013844, -1.08062375, 0.10050042, 1.15464056, 0.0929967761, 0.540971458, 0.0212178566, 0.503034949, 0.83255738, 0.46647051, 0.204722509, -1.27211988, -0.444015801, -1.69553638, -0.167419329, 0.143785626, 0.585121214, 0.57625556, 0.766840935, -0.19056502, -0.755467176, -0.427627087, -1.94814777, 0.537046909, -0.522750497, 0.331165344, 0.605711877, 0.562003255, 0.736680806, -1.39441252, -0.327535, 1.01853919, 0.116443828, 1.47550285, 0.656630814, -0.495611131, -1.10432601, 0.0967561, 0.29351303, -0.538147807, -0.0545213893, 0.0892568082, 0.401088953, -0.560582221, 1.08641076, -1.32994449, -0.445924222, 0.0628995895, 0.852968097, -1.05429208, -1.21338463, 0.188410252, -1.32656896, 2.27283859, 0.736774087, 0.86888063, -0.604602695, 0.614079833, 0.427769542, 0.578614533, -1.66400027, 0.2681638, 0.589362919, -0.591804683, 0.348048955, 1.31347501, -0.772396326, -2.10487413, 0.665272057, 0.853588164, 0.605928123, 0.72008884, -0.223881125, -0.394566655, -0.101686031, -0.331118852, -0.462497324, 1.38949108, -0.521819651, 1.39591086, -1.42767239, 1.93704259, 0.81604588, 0.0171508733, -0.640056431, -1.3012681, 1.19163561, -0.410113811, -1.73526335, 1.53767645, -1.04359388, -1.03642619, 1.89930785, -0.324641526, 1.51593268, 0.797907114, -0.572063386, -2.04110432, 0.031835407, 1.9947058, 1.20736992, 0.209628358, -0.567395151, 0.896474063, -0.0174190253, 0.991189063, -1.85954738, 1.42311418, 2.48760486, 0.063005507, 1.12650847, -3.11773896, 0.838043809, 1.25322545, -0.0175480414, -0.576092601, -0.82751286, -0.515049517, -0.974002957, -0.917269826, -0.833883166, -2.34605551, -0.683281422, 0.812363923, -0.515273392, -2.14664221, 1.90705419, 0.337061167, -1.04682851, -0.461431801, 0.959011614, -1.68747807, 0.863064945, 0.705874503, -0.100304469, -0.173330382, -0.0423407555, 0.511646032, -0.0373991951, -0.607062101, -2.50103807, -1.31582701, -0.953041434, 0.837779045, 0.0764538348, -0.810819387, 0.280284435, -0.236213714, -1.11943829, -0.0877167434, -0.164113134, 0.0495470762, 0.0518188849, 1.53973544, 0.469400644, -2.52379799, -1.73402917, 0.590020418, -0.393764257, -1.2683934, 0.143220857, -2.49708271, -0.518765092, 0.659234941, 1.24489355, -0.289200783, -1.2754811, -1.48785651, -0.35259527, 0.80408448, 1.63016224, 0.775638461, -0.759862423, 0.0902961642, 2.72458577, -1.18986583, 0.388371646, -0.219125047, 0.719351411, 0.518003643, 0.618203282, -0.45225516, 1.0927422, -0.932548165, -0.125248849, 0.549942911, 0.373292357, -1.24849963, 0.248122454, 1.00109971, 1.81128609, 0.199679971, 1.95777166, 1.29997993, 0.369897634, 0.912774324, 0.718598127, 1.4967773, -0.218519092, -0.762729585, 2.00808382, -0.586423159, 0.0979146063, -0.220159858, -2.12936592, 1.26146567, 1.8149302, -0.191354349, -2.65172577, 1.02468348, -2.95829439, 1.59597397, 0.151465356, -0.34589237, -0.302029073, 1.86898232, 1.75901365, 0.0111959949, -0.528900385, 0.47963959, -0.0871549323, -0.579753637, -0.183226317, -1.12163806, 0.475734562, -1.43882334, -0.749636829, 0.192253247, -1.01695478, 0.25153473, -0.67637831, -0.865114629, -0.48405847, -0.812186, -0.215609238, -0.322096676, -1.15442681, -0.688220382, -1.57351613, 0.203673214, -0.48019442, -1.68918633, 0.218946338, 0.100395821, 0.495316148, 0.754354596, -0.656868756, 0.639537454, -2.15639162, -0.082046479, 0.960244536, 1.58503592, 0.321607292, 1.37255871, 1.70459878, 0.236400947, -0.511867464, 0.938811898, -1.45468295, 0.311385959, 0.320807397, -2.11566138, -0.0946450382, 1.27652907, 0.143718213, -0.409889668, -2.37199855, -0.0662486851, -1.05720687, 0.292150408, 0.609753489, 0.190045491, -1.05734956, 2.46642, 0.548794568]

def resetPincone():
    listToDelete = index.query(
            vector=vectorFlag,
            top_k=10000,
            
            include_metadata=False
        )
    listToDelete=listToDelete["matches"]
    listId = []
    for ele in listToDelete:
        listId.append(ele['id'])
    if(len(listId) > 0):
        index.delete(ids=listId)


list_result = []

@jit(target_backend='cuda')
def extract_frames(video_file):
    resetPincone()
    frame_count = 0
    frame_rate = 60  # default 1s with 30 frame
    duration = 0 
    audio = MP4("video.mp4")
    duration = audio.info.length
    print("duration",duration)
    
    cap = cv2.VideoCapture(video_file)
    
    
    
    # Get the video file's name without extension
    video_name = os.path.splitext(os.path.basename(video_file))[0]
    
    # Create an output folder with a name corresponding to the video
    output_directory = f"{video_name}_frames"
    os.makedirs(output_directory, exist_ok=True)
    
   
    while True :
        # time.sleep(0.1)
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # if(frame_count > 5):
        #     break
        frame_count = frame_count +  1
        print("frame_count",frame_count)
     
        if(frame_count % frame_rate == 0):
   

            # Deblur the image
            sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpen = cv2.filter2D(frame, 0 , sharpen_kernel)

            frame = cv2.fastNlMeansDenoisingColored(sharpen,None,10,10,7,21)
            cv2.imwrite('tempt.jpg',frame)
            # frame = cv2.fastNlMeansDenoisingColored(sharpen,None,10,10,7,21)
            faces = app.get(frame)
            for i,face in enumerate(faces):
              if(face["det_score"] > 0.7):
                print("dimension .....")
                print(face['embedding'].shape)
                # print(face['embedding'])
                # print(len(face['embedding'][0]))
                
                # print(type(face))
                # print(face)
                if(len(array_em) == 0):
                    bbox = face['bbox']
                    bbox = [int(b) for b in bbox]
                    filename=f"0.0.jpg"
                    cv2.imwrite('./faces/%s'%filename,frame[bbox[1] : bbox[3], bbox[0]: bbox[2], ::-1])
                    array_em.append({
                        "speaker":0,
                        "frames":[frame_count],
                        # "embeddings":[face['embedding']]
                    })
                    filename=f"{frame_count}_0.jpg"
#                    frame = cv2.resize(frame, (250,200), interpolation=cv2.INTER_CUBIC)
                    cv2.imwrite('./outputs/%s'%filename,frame)
                    index.upsert(
                        vectors=[
                                {
                                    "id": str(uuid.uuid4()),
                                    "values": face['embedding'],
                                    "metadata": {
                                        "face":0
                                     }},
                            ]
                    )
                else:
                    flag  = False 
                    print("so luong  nguoi hien tai", len(array_em))
                    stt = 0  # số thứ tự của face khớp với face mới detect được 
                    search_result = index.query(
                        vector=face['embedding'].tolist(),
                        top_k=1,
                        include_metadata=True
                    )
                    matches = search_result['matches']
                    if(len(matches)):
                        face_match = matches[0]
                        if(face_match["score"] > weight_point):
                            flag = True
                            stt = int(face_match["metadata"]["face"])
          
                    # for j in range(len(array_em)):
                    #     em = array_em[j]
                    #     # print("em",em)
                    #     print("so mat", len(faces))
                    #     print("so phan tu con",len(em["embeddings"]), frame_count)
                    #     # print("phan tu dau tien", em["embeddings"][0])
                    #     count = 0
                    #     for x in range(len(em["embeddings"])):
                    #         # if(  ( ( len(em["embeddings"]) > 6 ) and (x >  (len(em["embeddings"]) - 6 )) ) or (len(em["embeddings"]) <= 6) ):
                    #             # time.sleep(0.1)
                    #             # print("phan tu con",x)
                    #             cosin_value = cosin(em["embeddings"][x],face['embedding'])
                    #             # print("cosin_value",cosin_value)
                    #             # print("count speaker", len(array_em))
                    #             if(cosin_value >  weight_point):
                    #                 stt = j
                    #                 flag = True
                    #             count = count + 1 
                            
                    #     print("so lan tinh ....... ", count)
                    if (flag == False): 
                        
                        array_em.append({
                                "speaker":len(array_em),
                                "frames":[frame_count],
                                # "embeddings":[face['embedding']]
                            }
                        )
                        filename = f"{len(array_em) -1 }_face.jpg"
                        bbox = face['bbox']
                        bbox = [int(b) for b in bbox]
                        try:
                            filename = f"{frame_count}_{filename}"
                            cv2.imwrite('./faces/%s'%filename,frame[bbox[1] : bbox[3], bbox[0]: bbox[2], ::-1])
 #                           frame = cv2.resize(frame, (250,200), interpolation=cv2.INTER_CUBIC)
                            cv2.imwrite('./outputs/%s'%filename,frame)
                        except:
                            print("Saving error") 
                            # return
                        index.upsert(
                            vectors=[
                                    {
                                        "id": str(uuid.uuid4()),
                                        "values": face['embedding'],
                                        "metadata": {
                                            "face":len(array_em) -1 
                                        }},
                                ]
                        )
                    if(flag == True):
                        # array_em[stt]["embeddings"].append(face['embedding'])
                        array_em[stt]["frames"].append(frame_count)

                        filename = f"{stt}_face.jpg"
                        bbox = face['bbox']
                        bbox = [int(b) for b in bbox]
                        try:
                            filename = f"{frame_count}_{filename}"
                            cv2.imwrite('./faces/%s'%filename,frame[bbox[1] : bbox[3], bbox[0]: bbox[2], ::-1])
  #                          frame = cv2.resize(frame, (250,200), interpolation=cv2.INTER_CUBIC)
                            cv2.imwrite('./outputs/%s'%filename,frame)
                        except:
                            print("Error saving") 
                        index.upsert(
                            vectors=[
                                    {
                                        "id": str(uuid.uuid4()),
                                        "values": face['embedding'],
                                        "metadata": {
                                            "face":stt
                                        }},
                                ]
                        )
          
            # tempt = []
            # for ele in array_em:
            #     tempt.append({
            #         "frame_count": frame_count, 
            #         "duration": duration,
            #         "frame_rate": frame_rate,
            #         "speaker":ele["speaker"],
            #         "frames":ele["frames"]
            #     })
            # with open('data.json', 'w') as f:
            #     json.dump(tempt, f, indent=4)
            # with open('data.json', 'r') as file:
            #     data = json.load(file)
            #     for em in data:
            #         frame_rate = em["frame_rate"] 
            #         time_per_frame = em["duration"] / em["frame_count"]
            #         list_time_exist = []
            #         duration_exist = []
            #         list_frame = em["frames"]
            #         print(list_frame)
            #         print("so frame", len(list_frame))
            #         for i in range(len(list_frame)-1):
            #             if(list_frame[i] == frame_rate):
            #                 duration_exist.append(0)
            #             duration_exist.append(list_frame[i])
            #             if( (list_frame[i + 1] - list_frame[i]) > frame_rate):
            #                 list_time_exist.append([duration_exist[0]*time_per_frame,duration_exist[len(duration_exist) - 1] * time_per_frame])
            #                 duration_exist = []
            #             else:
            #                     if( i == len(list_frame)-2):
            #                         duration_exist.append(list_frame[i+1])
            #                         list_time_exist.append([duration_exist[0]*time_per_frame,duration_exist[len(duration_exist) - 1] * time_per_frame])
            #                         duration_exist = []
            #         list_result.append({
            #             'face':em['speaker'],
            #             'duration_exist':list_time_exist
            #         })

            # with open('result.json', 'w') as f:
            #     json.dump(list_result, f, indent=4)
        # print(f"Frame {frame_count} has been extracted and saved as {output_file}")
    
    for ele in array_em:
#        del(ele['embeddings'])
        ele["frame_count"] = frame_count
        ele["duration"] = duration
        ele["frame_rate"] = frame_rate
    cap.release()
    print("End video")


start = time.time() 
extract_frames('video.mp4')



    
print("array_em",array_em)
print("array_em",len(array_em))

with open('data.json', 'w') as f:
    json.dump(array_em, f, indent=4)

# Open and read the JSON file
with open('data.json', 'r') as file:
    data = json.load(file)
    for em in data:
        frame_rate = em["frame_rate"] 
        time_per_frame = em["duration"] / em["frame_count"]
        list_time_exist = []
        duration_exist = []
        list_frame = em["frames"]
        # print(list_frame)
        print("so frame", len(list_frame))
        for i in range(len(list_frame)-1):
           if(list_frame[i] == frame_rate):
              duration_exist.append(0)
           duration_exist.append(list_frame[i])
           if( (list_frame[i + 1] - list_frame[i]) > frame_rate):
               list_time_exist.append([duration_exist[0]*time_per_frame,duration_exist[len(duration_exist) - 1] * time_per_frame])
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

with open('result.json', 'w') as f:
    json.dump(list_result, f, indent=4)

end = time.time() 
print("excution time", end - start)
