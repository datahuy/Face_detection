import os
from itertools import groupby
from typing import List, Dict
import argparse
from tqdm import tqdm
import numpy as np
import torch
import cv2

from utils import read_image, enumerate_images, preprocess
from arcface import get_model
from pymilvus import Milvus, MilvusClient


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

network = "r50"
weights = r"F:\Face_Models\Arcface\backbone.pth"

model = get_model(name=network, fp16=False)
model.load_state_dict(torch.load(weights, map_location=torch.device("cpu")))
model.eval()
model.to(device)


def extract_feature(file) -> np.ndarray:

    if not isinstance(file, np.ndarray):
        face = cv2.imread(file)[:, :, ::-1]
    else:
        face = file
    if face is None:
        return face
    img = np.array(face)
    #img = np.transpose(img, (2, 0, 1))
    img = preprocess(img=img)
        
    img = torch.from_numpy(img).unsqueeze(0).float()
    #print(img.shape)
    img.div_(255).sub_(0.5).div_(0.5)
        
    img = img.to(device=device)
    with torch.no_grad():
        features = model(img).cpu().numpy()

    features = features / np.linalg.norm(features, axis=1)[:, np.newaxis]
    print(features.shape)
    features = np.mean(features, axis=0)

    return features


def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data_dir", type=str, default=r"F:\Faces\VN-celeb-aligned")
    args = parser.parse_args()
    
    return args

if __name__ == "__main__":
    args = get_args()
    
    data_dir = args.data_dir
    
    with open("./authen.txt", "r") as f:
        lines = f.read()
        
    api_endpoint, token = lines.split("\n")
    print(api_endpoint, token)
    
    # Connect to cluster
    client = MilvusClient(uri=api_endpoint, token=token)
    
    print(client.list_collections())
    collection_name = client.list_collections()[0]
    
    #image_path = r"F:\Faces\VN-celeb-aligned\1\0.png"
    image_path = r"./image/face_10.jpg"
    
    feature = extract_feature(image_path)
    print(feature)
    
    res = client.search(collection_name=collection_name,
                        data=[feature.tolist()],
                        output_fields=["name"])
    
    print(res)