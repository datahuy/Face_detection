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
    
    images_list: List[str] = enumerate_images(data_dir=data_dir)
    
    group_to_images: Dict[str, List[str]] = {}
    
    for key, items in groupby(images_list, key=lambda x: x.split(os.sep)[-2]):
        group_to_images[key] = list(items)
        
    entities = []
        
    for person in tqdm(group_to_images):

        images = group_to_images[person]
        
        features = extract_feature(files=images)
        print(person, features.shape)
        entities.append({"name": person,
                         "vector": features.tolist()})
    res = client.insert(collection_name=collection_name,
                        data=entities,
                        progress_bar=True)
    
    print(res)