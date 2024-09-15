#!pip install deepface
from deepface import DeepFace
obj = DeepFace.verify("102_1_face.jpg", "102_2_face.jpg"
          , model_name = 'ArcFace', detector_backend = 'retinaface')
print(obj["verified"])