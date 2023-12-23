import os
import argparse
from typing import List, Dict
from itertools import groupby
from tqdm import tqdm
import numpy as np

import cv2

import torch

import face_alignment

from align.face_align import *
from utils import enumerate_images, alignment_procedure, align_face

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data_dir", type=str, default=r"F:\Faces\VN-celeb")
    parser.add_argument("--save_dir", type=str, default=r"F:\Faces\VN-celeb-aligned")
    args = parser.parse_args()
    
    return args



if __name__ == "__main__":
    
    args = get_args()
    
    data_dir = args.data_dir
    save_dir = args.save_dir
    
    images_list: List[str] = enumerate_images(data_dir=data_dir)
    
    group_to_images: Dict[str, List[str]] = {}
    
    for key, items in groupby(images_list, key=lambda x: x.split(os.sep)[-2]):
        group_to_images[key] = list(items)
        
    #print(group_to_images)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fa = face_alignment.FaceAlignment(landmarks_type=face_alignment.LandmarksType.TWO_D, device=device, face_detector="sfd")
    
    for person in tqdm(group_to_images):
        person_dir = os.path.join(save_dir, person)
        if not os.path.exists(person_dir):
            os.makedirs(person_dir, exist_ok=True)
        images = group_to_images[person]
        for image in images:
            try:
                img = cv2.imread(image)
                dets = fa.get_landmarks(img, return_bboxes=True, return_landmark_score=True)
                bbox = dets[2][0]
                landmark = dets[0][0]
                x_1, y_1, x_2, y_2, prob = bbox
                face_img = img[max(int(y_1), 0):min(int(y_2), img.shape[0]), max(int(x_1), 0):min(int(x_2), img.shape[1])]
                left_eye_indices = list(range(36, 42))
                right_eye_indices = list(range(42, 48))
                left_eye = np.mean(landmark[left_eye_indices], axis=0).astype(np.int16).astype(np.float32)
                right_eye = np.mean(landmark[right_eye_indices], axis=0).astype(np.int16).astype(np.float32)
                nose = landmark[30]
                left_mouth = landmark[48]
                right_mouth = landmark[54]
                lmk = np.array([left_eye, right_eye, nose, left_mouth, right_mouth], dtype=np.float32)
                #face_img = alignment_procedure(img=face_img, left_eye=left_eye, right_eye=right_eye, nose=nose)
                #face_img = align_face(image, face_landmarks=landmark, output_size=112, enable_padding=False)
                face_img = norm_crop(face_img, lmk)
                save_path = os.path.join(person_dir, os.path.basename(image))
                cv2.imwrite(save_path, face_img)
            except Exception as e:
                print(e)