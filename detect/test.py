
from insightface.app import FaceAnalysis
import cv2
import torch 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
ctx_id = 0 if device.type == 'cuda' else  -1
appmain = FaceAnalysis('buffalo_l',providers=['CUDAExecutionProvider'])
appmain.prepare(ctx_id=ctx_id, det_thresh=0.5, det_size=(640, 640))

img = cv2.imread("/home/ubuntua5000/detect/detect/outputs/db696a35-0043-4aba-a844-295e3432a118/ch02_20240904131243_000_640x640/15/1575_0_face.jpg")

faces = appmain.get(img)

print(faces)
