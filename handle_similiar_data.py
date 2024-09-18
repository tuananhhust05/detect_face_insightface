
import json

def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def checkExist(num,arr):
    flag = False
    for e in arr:
        if(num == e):
            flag = True
    return flag 

with open('face_similiar_data.json', 'r') as file:
    print("Start ....")
    data = json.load(file)
    flag = True 
    for ele in data:
          similiars = ele["similiar"]
          for similiar in similiars:
              similiar = int(similiar)
              if( int(similiar) != int(ele['face'])):
                    index = 0
                    for i in range(len(data)): 
                        if( int(data[i]['face']) == int(similiar)):
                            index = i
                    if(index != 0):
                      del(data[index])



    print("1....")
    print(data)
    print("2...")
    ele_deleted = []
    for ele in data:
        check = True
        for check_ele in ele_deleted:
            if check_ele  == ele['face']: check = False 
        if(check == True):
            for ele2 in data:
                if(ele2['face'] != ele['face']):
                    inter = intersection(ele['similiar'], ele2['similiar'])
                    if(len(inter) > 0):
                        for similiar in ele2['similiar']:
                            ele['similiar'].append(similiar)
                        ele['similiar'].append(ele2['face'])

                        # xoa du lieu
                        index = 0
                        for i in range(len(data)): 
                            if(data[i]['face'] == ele2['face']):
                                index = i
                        if(index != 0):
                            del(data[index])

                        ele_deleted.append(ele2['face'])
    
    print("3...")
    for ele in data:
        print(list(set(ele["similiar"])))
        ele["similiar"] = list(set(ele["similiar"]))
        new_similiar = []
        similiars = ele["similiar"]
        for similiar in similiars:
            new_similiar.append(int(similiar))
        ele["similiar"] = new_similiar
    print(data)
    
    final_data = []
    with open('data.json', 'r') as file2:
        data2 = json.load(file2)
        for data_ele in data:
            frames = []
            list_face = [data_ele['face']]
            print("data_ele['similiar']",data_ele['similiar'])
            for similiar in data_ele['similiar']:
                list_face.append(similiar)
   
            for ele in data2:
                if( checkExist(ele['speaker'], list_face) == True):
                    for frame in ele['frames']:
                        frames.append(frame)
            final_data.append({
                "speaker": data_ele['face'],
                "frames":sorted(frames),
                "frame_count": 36001,
                "duration": 1200.0072562358278,
                "frame_rate": 30,
                "array_face":list_face
            })
    with open('final_data.json', 'w') as f:
        json.dump(final_data, f, indent=4)
    # while(flag):
       
