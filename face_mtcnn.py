from mtcnn import MTCNN
import cv2
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# img = cv2.imread("face.jpg")

cap = cv2.VideoCapture(0)

while True:
    _,img = cap.read()
    detector = MTCNN()
    face_dict = detector.detect_faces(img)
    if len(face_dict) > 0:
        face_dict = detector.detect_faces(img)[0]
        print("Face Identified")
        face_box = face_dict["box"]
        conf = face_dict["confidence"]
        keypoints = face_dict["keypoints"]
        cv2.rectangle(img,(face_box[0],face_box[1]),(face_box[0]+face_box[2],face_box[1]+face_box[3]),(0,0,255),2)
        cv2.putText(img,f"{int(conf*100)}%",(face_box[0],face_box[1]-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)
        img = cv2.circle(img, keypoints["left_eye"], radius=5, color=(0, 0, 255), thickness=-1)
        img = cv2.circle(img, keypoints["right_eye"], radius=5, color=(0, 0, 255), thickness=-1)
        img = cv2.circle(img, keypoints["nose"], radius=5, color=(0, 0, 255), thickness=-1)
        img = cv2.circle(img, keypoints["mouth_left"], radius=5, color=(0, 0, 255), thickness=-1)
        img = cv2.circle(img, keypoints["mouth_right"], radius=5, color=(0, 0, 255), thickness=-1)
    else:
        print("No face identified")
    cv2.imshow("face_mtcnn",img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()

