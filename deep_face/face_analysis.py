from deepface import DeepFace
import cv2
import logging
import os
import math

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
logging.getLogger('tensorflow').setLevel(logging.FATAL)

img_path = "Bhargav.jpeg"

img = cv2.imread(img_path)

demography = DeepFace.analyze(img_path)

print(demography)
