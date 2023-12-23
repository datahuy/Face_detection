import pickle

import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc 


def plot_roc_curve(tpr, fpr):
    roc_auc = auc(fpr, tpr)
    plt.title("Receiver Operating Characteristic")
    plt.plot(fpr, tpr, "b", label = "AUC = %0.2f" % roc_auc)
    plt.legend(loc = "lower right")
    plt.plot([0, 1], [0, 1],"r--")
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.grid()
    plt.show()
    

def compute_gmeans(tpr: np.ndarray, fpr: np.ndarray) -> np.ndarray:
    return tpr * (1 - fpr)    


if __name__ == "__main__":
    feature_file = "./features_labels.pkl"
    
    with open(feature_file, "rb") as f:
        infos = pickle.load(f)
        
    print(infos["features"].shape, infos["labels"].shape)
    
    features = infos["features"][:1000]
    labels = infos["labels"].astype(np.uint16)[:1000]
    
    dist_matrix = np.sqrt(np.sum((features[:, np.newaxis, :] - features[np.newaxis, :, :])**2, axis=-1))
    label_matrix = labels[:, np.newaxis] == labels[np.newaxis, :]
    
    print(dist_matrix, label_matrix)
    
    # positive = np.argwhere(label_matrix == True).T
    # negative = np.argwhere(label_matrix == False).T
    # postive_dist = dist_matrix[positive[0], positive[1]]
    # print(postive_dist)
    # negative_dist = dis
    
    dist = dist_matrix.flatten()
    label = label_matrix.flatten().astype(np.uint8)
    
    print(dist, label)
    
    fpr, tpr, thresholds = roc_curve(label, -dist, pos_label=1)
    
    print(fpr, tpr, thresholds)
    
    #plot_roc_curve(tpr=tpr, fpr=fpr)
    
    roc_auc = auc(fpr, tpr)
    plt.title("Receiver Operating Characteristic")
    plt.plot(fpr, tpr, "b", label = "AUC = %0.2f" % roc_auc)
    plt.legend(loc = "lower right")
    plt.plot([0, 1], [0, 1],"r--")
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.grid()
    
    
    gmeans = compute_gmeans(tpr=tpr, fpr=fpr)
            
    idx = np.argmax(gmeans)
    prob_star = - thresholds[idx]
    gmean_star = gmeans[idx]
    
    plt.text(fpr[idx] - 0.1, tpr[idx] + 0.1, str(prob_star))
    plt.plot([fpr[idx]], [tpr[idx]], marker="o", markersize=10, markerfacecolor="red")
    
    plt.show()