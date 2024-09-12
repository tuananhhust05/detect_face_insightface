import numpy as np 
import os 
import glob 
import cv2
import matplotlib.pyplot as plt 

import insightface 
from insightface.app import FaceAnalyst 
from insightface.data import get_image as ins_get_image 
print('insightface',insightface.__version__ )