import numpy as np
import torch
from arcface import get_model
from utils import enumerate_images, preprocess


if __name__ == "__main__":

    data_dir = "image"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = "r50"
    weights = r"F:\Face_Models\Arcface\backbone.pth"
    
    model = get_model(name=network, fp16=False)
    model.load_state_dict(torch.load(weights, map_location=torch.device("cpu")))
    model.eval()
    model.to(device)
    
    images_list = enumerate_images(data_dir=data_dir)
    
    images_array = []
    for image in images_list:
        images_array.append(preprocess(img=image))
        
    images_array = np.stack(images_array, axis=0)
    
    images_array = torch.from_numpy(images_array).float()
    images_array.div_(255).sub_(0.5).div_(0.5)
    
    with torch.no_grad():
        feat = model(images_array).numpy()
    
    feat = feat / np.sqrt(np.sum(feat**2, axis=-1, keepdims=True))
    print(feat)
    
    dist =  np.sum((feat[:, None, :] - feat[None, :, :])**2, axis=-1)
    print(dist)
    