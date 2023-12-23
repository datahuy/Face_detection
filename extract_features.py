import os
from itertools import groupby
from typing import List, Dict
import argparse
import pickle
from tqdm import tqdm
import numpy as np
import torch
import cv2

from utils import read_image, enumerate_images, preprocess
from arcface import get_model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

network = "r50"
weights = r"F:\Face_Models\Arcface\backbone.pth"

model = get_model(name=network, fp16=False)
model.load_state_dict(torch.load(weights, map_location=torch.device("cpu")))
model.eval()
model.to(device)


def extract_feature(files) -> np.ndarray:
    imgs = []
    for file in files:
        face = cv2.imread(file)[:, :, ::-1]
        if face is None:
            return face

        img = np.array(face)
        #img = np.transpose(img, (2, 0, 1))
        img = preprocess(img=img)
        imgs.append(img)
        
    imgs = np.stack(imgs, axis=0)
    imgs = torch.from_numpy(imgs).float()
    #print(img.shape)
    imgs.div_(255).sub_(0.5).div_(0.5)
        
    imgs = imgs.to(device=device)
    with torch.no_grad():
        features = model(imgs).cpu().numpy()

    features = features / np.linalg.norm(features, axis=1)[:, np.newaxis]

    return features


def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data_dir", type=str, default=r"F:\Faces\VN-celeb-aligned")
    parser.add_argument("--save_dir", type=str, default=".")
    args = parser.parse_args()
    
    return args

if __name__ == "__main__":
    args = get_args()
    
    data_dir = args.data_dir
    save_dir = args.save_dir
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        
    images_list: List[str] = enumerate_images(data_dir=data_dir)
    
    group_to_images: Dict[str, List[str]] = {}
    
    image_features = []
    image_labels = []
    
    for key, items in groupby(images_list, key=lambda x: x.split(os.sep)[-2]):
        group_to_images[key] = list(items)
        
    for person in tqdm(group_to_images):
        images = group_to_images[person]
        features = extract_feature(files=images)
        labels = int(person) * np.ones(shape=[features.shape[0]])

        image_features.append(features)
        image_labels.append(labels)
        #print(features.shape, labels.shape)
        
    image_features = np.concatenate(image_features, axis=0)
    image_labels = np.concatenate(image_labels, axis=0)
    
    with open(os.path.join(save_dir, "features_labels.pkl"), "wb") as f:
        pickle.dump({"features": image_features,
                     "labels": image_labels}, f)