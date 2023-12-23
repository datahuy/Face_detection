import numpy as np


class FaceDetection(object):
    def __init__(self,bbox, landmark, detection_id):
        self.bbox = bbox
        self.landmark = landmark
        self.detection_id = detection_id