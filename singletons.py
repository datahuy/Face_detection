import os
import numpy as np
import torch
import face_alignment
from pymilvus import MilvusClient
from arcface import get_model

class Detector(object):
    
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            cls._instance = face_alignment.FaceAlignment(landmarks_type=face_alignment.LandmarksType.TWO_D, device=device, face_detector="sfd")
        return cls._instance
    
    # def get_landmarks(self, image, return_bboxes=True, return_landmark_score=True):
    #     dets = self._instance.get_landmarks(image, return_bboxes=True, return_landmark_score=True)
    #     return dets
    
# detector_1 = Detector()
# detector_2 = Detector()

# print(detector_1 is detector_2)


class RecognitionModel(object):
    _instance = None
    
    def __new__(cls, network, weights):
        if cls._instance is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            cls._instance = get_model(name=network, fp16=False)
            cls._instance.load_state_dict(torch.load(weights, map_location=torch.device("cpu")))
            cls._instance.eval()
            cls._instance.to(device)
        return cls._instance
    
    # def __call__(self, imgs):
    #     with torch.no_grad():
    #         features = self._instance(imgs).cpu().numpy()
            
    #     return features

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# network = "r50"
# weights = r"F:\Face_Models\Arcface\backbone.pth"

# model_1 = RecognitionModel(network=network, weights=weights)
# model_2 = RecognitionModel(network=network, weights=weights)
# print(model_1 is model_2)

# img = np.random.rand(112, 112, 3)
# img = np.transpose(img, (2, 0, 1))
# img = torch.from_numpy(img).unsqueeze(0).float()
# img.div_(255).sub_(0.5).div_(0.5)
# img = img.to(device=device)
# with torch.no_grad():
#     features = model_1(img).cpu().numpy()
# features = features / np.linalg.norm(features)
# print(features)

# with torch.no_grad():
#     features = model_2(img).cpu().numpy()
# features = features / np.linalg.norm(features)
# print(features)

class MilvusConnection(object):
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            with open("./authen.txt", "r") as f:
                lines = f.read()
            api_endpoint, token = lines.split("\n")
            cls._instance = MilvusClient(uri=api_endpoint, token=token)
        return cls._instance