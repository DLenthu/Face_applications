from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt
import logging
import os
import math

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
logging.getLogger('tensorflow').setLevel(logging.FATAL)

img1_path = "test1.jpg"
img2_path = "test2.jpg"
img3_path = "test3.jpeg"
img4_path = "test4.jpeg"

img1 = cv2.imread(img1_path) 
img2 = cv2.imread(img2_path)
img3 = cv2.imread(img3_path)
img4 = cv2.imread(img4_path)

def verify(image1,image2):
    result = DeepFace.verify(image1,image2)
    similarity = math.acos(result["distance"])/(math.pi/2)
    print("Both people have a similarity score of :",similarity)

verify(img1,img3)