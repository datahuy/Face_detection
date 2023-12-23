import os
from typing import List
import numpy as np
import cv2
import torch
from PyQt5 import QtCore, QtWidgets
from singletons import Detector, RecognitionModel, MilvusConnection
from face_tracking.face_tracker import FaceTracker
from face_tracking.face_detection import FaceDetection
from videostream.videostream import QueuedStream
from align.face_align import norm_crop
from utils import alignment_procedure, compute_eyes_angle, preprocess


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DefaultProcess(QtCore.QThread):
    update_frame = QtCore.pyqtSignal(np.ndarray)
    
    def __init__(self, network, weights) -> None:
        super(DefaultProcess, self).__init__()
        self.network = network
        self.weights = weights
        self.threshold = 0.6
        self.stopped = False
        
    def run(self):
        stream = QueuedStream(uri=[], drop=True, fps=25)
        detector = Detector()
        face_recog_model = RecognitionModel(network=self.network, weights=self.weights)
        tracker = FaceTracker(inactive_steps_before_removed=20,
                              reid_iou_threshold=0.6,
                              max_traject_steps=30,
                              tentative_steps_before_accepted=3)
        client = MilvusConnection()
        collection_name = client.list_collections()[0]
        stream.start()
        
        if not stream.isOpened():
            print("Can not open video")
            return
        
        
        while not self.stopped:
            ret, frame, frame_id = stream.read()
            if not ret:
                break
            
            dets = detector.get_landmarks_from_image(frame, return_bboxes=True, return_landmark_score=True)
            bboxes = dets[2]
            landmarks = dets[0]
            
            
            try:
                
                detections_list = []
                for i, (landmark, bbox) in enumerate(zip(landmarks, bboxes)):
            
                    x_1, y_1, x_2, y_2, prob = bbox
                    if prob > 0.7:
                        # w = x_2 - x_1
                        # h = y_2 - y_1

                        detections_list.append(FaceDetection(bbox=bbox, landmark=landmark, detection_id=i))
                
                tracker.step(face_detections=detections_list)
                face_tracks = tracker.get_result()
                
                order_to_track_id = {}
                track_id_to_bbox = {}
                images_list = []
                frame_copy = frame.copy()
                for i, face_track in enumerate(face_tracks):
                    bbox = face_track.bbox
                    landmark = face_track.landmark
                    #print(bbox)
                    track_id = face_track.track_id
                    order_to_track_id[i] = track_id
                    x_min, y_min, x_max, y_max, prob = bbox
                    result = "Track id: {}".format(track_id)
                    x_min = int(x_min)
                    x_max = int(x_max)
                    y_min = int(y_min)
                    y_max = int(y_max)
                    c_x = int((x_min + x_max)/2)
                    c_y = int((y_min + y_max)/2)
                    
                    w = x_max - x_min
                    h = y_max - y_min
                    
                    if h > w:
                        h_max = True
                    else:
                        h_max = False
                    
                    if h_max:
                        x_min = max(c_x - int(0.5 * h), 0)
                        x_max = min(c_x + int(0.5 * h), frame.shape[1])
                    else:
                        y_min = max(c_y - int(0.5 * w), 0)
                        y_max = min(c_y + int(0.5 * w), frame.shape[0])
                    face_img = frame_copy[y_min:y_max, x_min:x_max]
                    left_eye_indices = list(range(36, 42))
                    right_eye_indices = list(range(42, 48))
                    left_eye = np.mean(landmark[left_eye_indices], axis=0).astype(np.int16).astype(np.float32)
                    right_eye = np.mean(landmark[right_eye_indices], axis=0).astype(np.int16).astype(np.float32)
                    nose = landmark[30]
                    left_mouth = landmark[48]
                    right_mouth = landmark[54]
                    lmk = np.array([left_eye, right_eye, nose, left_mouth, right_mouth], dtype=np.float32)
                    #lmk = lmk - np.array([max(x_min - int(ext * (x_max - x_min)), 0), max(y_min - int(ext*(y_max - y_min)), 0)], dtype=np.float32)
                    lmk = lmk - np.array([x_min, y_min], dtype=np.float32)
                    face_img = norm_crop(face_img, lmk)
                    #face_img = alignment_procedure(img=face_img, left_eye=left_eye, right_eye=right_eye, nose=nose)
                    #face_img = np.transpose(face_img, (2, 0, 1))
                    images_list.append(preprocess(face_img))
                    
                    track_id_to_bbox[track_id] = (x_min, y_min, x_max, y_max)
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 255, 0), 1)
                    #cv2.putText(frame, result, (x_min + int(0.05*(x_max - x_min)), y_min - int(0.05*(y_max - y_min))), cv2.FONT_HERSHEY_COMPLEX, 0.3, (255, 255, 0), 1)
                    
                    for j, point in enumerate(landmark):
                        if j in [8, 30, 36, 45, 48, 54]:
                            # 8: Cằm
                            # 30: Mũi
                            # 36: Mắt trái
                            # 45: Mắt phải
                            # 48: Miệng trái
                            # 54: Miệng phải
                            cv2.circle(frame, (int(point[0]), int(point[1])), 2, (0, 0, 255), -1)
                            cv2.putText(frame, str(j), (int(point[0]) + 3, int(point[1]) - 3),  cv2.FONT_HERSHEY_COMPLEX, 0.3, (255, 255, 0), 1)
                
                try:
                    images = np.stack(images_list, axis=0)
                    images = torch.from_numpy(images).float()
                    images.div_(255).sub_(0.5).div_(0.5)
                    images = images.to(device)

                    with torch.no_grad():
                        features = face_recog_model(images).cpu().numpy()

                    features = features / np.linalg.norm(features, axis=1)

                    features = features.tolist()
                    print(len(features), len(features[0]))
                    
                    query_res = client.search(collection_name=collection_name,
                                              data=features,
                                              limit=3,
                                              output_fields=["name"])
                    print("Query result: {}".format(query_res))
                    for i, res in enumerate(query_res):
                        dist = res[0]["distance"]
                        if dist > self.threshold:
                            person = "Unknown"
                        else:
                            person = res[0]["entity"]["name"]
                            
                        cv2.putText(frame, person, (x_min + int(0.05*(x_max - x_min)), y_min - int(0.05*(y_max - y_min))), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 0), 1)
                except Exception as e:
                    print("Extract feature error: {}".format(e))
                
                self.update_frame.emit(frame)
                
            except Exception as e:
                print("Exception: {}".format(e))
                continue
            
        stream.stop()
        
    
    def set_start(self):
        self.stopped = False
    
            
    def stop(self):
        self.stopped = True
        
        
        
class RegisterProcess(QtCore.QThread):
    
    #update_frame = QtCore.pyqtSignal(np.ndarray, np.ndarray, bool)
    update_frame = QtCore.pyqtSignal(np.ndarray)
    update_vector = QtCore.pyqtSignal(np.ndarray)
        
    def __init__(self, network, weights) -> None:
        super(RegisterProcess, self).__init__()
        self.network = network
        self.weights = weights
        self.threshold = 0.8
        self.stopped = False
        
    def run(self):
        stream = QueuedStream(uri=[], drop=True, fps=25)
        detector = Detector()
        face_recog_model = RecognitionModel(network=self.network, weights=self.weights)
        tracker = FaceTracker(inactive_steps_before_removed=20,
                              reid_iou_threshold=0.6,
                              max_traject_steps=30,
                              tentative_steps_before_accepted=3)
        client = MilvusConnection()
        collection_name = client.list_collections()[0]
        stream.start()
        
        if not stream.isOpened():
            print("Can not open video")
            return
        satisfy = False
        already_db = True
        satisfy_angle = False
        vector = None
        while not self.stopped:
            ret, frame, frame_id = stream.read()
            if not ret:
                break
            
            
            dets = detector.get_landmarks_from_image(frame, return_bboxes=True, return_landmark_score=True)
            bboxes = dets[2]
            landmarks = dets[0]
            
            try:
                detections_list = []
                for i, (landmark, bbox) in enumerate(zip(landmarks, bboxes)):
            
                    x_1, y_1, x_2, y_2, prob = bbox
                    if prob > 0.7:

                        detections_list.append(FaceDetection(bbox=bbox, landmark=landmark, detection_id=i))
                
                tracker.step(face_detections=detections_list)
                face_tracks = tracker.get_result()
                
                order_to_track_id = {}
                images_list = []
                frame_copy = frame.copy()
                center_frame_x = int(frame.shape[1]/2)
                center_frame_y = int(frame.shape[0]/2)
                if len(face_tracks) > 0:
                    face_tracks.sort(key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse=True)
                    face_track = face_tracks[0]
                    bbox = face_track.bbox
                    landmark = face_track.landmark
                    x_min, y_min, x_max, y_max, prob = bbox
                    x_min = int(x_min)
                    x_max = int(x_max)
                    y_min = int(y_min)
                    y_max = int(y_max)
                    c_x = int((x_min + x_max)/2)
                    c_y = int((y_min + y_max)/2)
                    
                    w = x_max - x_min
                    h = y_max - y_min
                    
                    if h > w:
                        h_max = True
                    else:
                        h_max = False
                    
                    if h_max:
                        x_min = max(c_x - int(0.5 * h), 0)
                        x_max = min(c_x + int(0.5 * h), frame.shape[1])
                    else:
                        y_min = max(c_y - int(0.5 * w), 0)
                        y_max = min(c_y + int(0.5 * w), frame.shape[0])
                    
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 255, 0), 1)
                    
                    if (abs(c_x - center_frame_x) / center_frame_x) > 0.5 or (abs(c_y - center_frame_y) / center_frame_y) > 0.5:
                        
                        cv2.putText(frame, "Move to center of camera", (int(0.1 * frame.shape[0]), int(0.1 * frame.shape[1])), cv2.FONT_HERSHEY_COMPLEX, 1.0, (255, 255, 0), 2)
                        #self.update_frame.emit(frame, vector, satisfy)
                        #vector = np.random.rand(3)
                        self.update_frame.emit(frame)
                        continue
                    
                    
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 255, 0), 1)
                    
                    for j, point in enumerate(landmark):
                        if j in [8, 30, 36, 45, 48, 54]:
                            # 8: Cằm
                            # 30: Mũi
                            # 36: Mắt trái
                            # 45: Mắt phải
                            # 48: Miệng trái
                            # 54: Miệng phải
                            cv2.circle(frame, (int(point[0]), int(point[1])), 2, (0, 0, 255), -1)
                            cv2.putText(frame, str(j), (int(point[0]) + 3, int(point[1]) - 3),  cv2.FONT_HERSHEY_COMPLEX, 0.3, (255, 255, 0), 1)
                    
                    angle = compute_eyes_angle(landmark=landmark)
                    
                    if abs(angle) > 15:
                        cv2.putText(frame, "Balance your face", (int(0.1 * frame.shape[0]), int(0.1 * frame.shape[1])), cv2.FONT_HERSHEY_COMPLEX, 1.0, (255, 255, 0), 2)
                        #vector = np.random.rand(3)
                        self.update_frame.emit(frame)
                        continue
                    print("Angle: {}".format(angle))
                    #self.update_frame.emit(frame)
                    
                    face_img = frame_copy[y_min:y_max, x_min:x_max]
                    left_eye_indices = list(range(36, 42))
                    right_eye_indices = list(range(42, 48))
                    left_eye = np.mean(landmark[left_eye_indices], axis=0).astype(np.int16).astype(np.float32)
                    right_eye = np.mean(landmark[right_eye_indices], axis=0).astype(np.int16).astype(np.float32)
                    nose = landmark[30]
                    left_mouth = landmark[48]
                    right_mouth = landmark[54]
                    lmk = np.array([left_eye, right_eye, nose, left_mouth, right_mouth], dtype=np.float32)
                    lmk = lmk - np.array([x_min, y_min], dtype=np.float32)
                    face_img = norm_crop(face_img, lmk)
                    face_img = preprocess(img=face_img)
                    
                    img = torch.from_numpy(face_img).unsqueeze(0).float()
                    img.div_(255).sub_(0.5).div_(0.5)
                    
                    with torch.no_grad():
                        features = face_recog_model(img).cpu().numpy()
                        
                    features = features / np.linalg.norm(features, axis=1)
                    features_list = features.tolist()
                    print(len(features), len(features[0]))
                      
                    query_res = client.search(collection_name=collection_name,
                                              data=features_list,
                                              limit=3,
                                              output_fields=["name"])
                    
                    # dist = query_res[0][0]["distance"]
                    
                    # if dist < self.threshold:
                    #     print(query_res)
                    #     already_name = query_res[0][0]["entity"]["name"]
                    #     cv2.putText(frame, "Are you registered as {}".format(already_name), (int(0.1 * frame.shape[0]), int(0.1 * frame.shape[1])), cv2.FONT_HERSHEY_COMPLEX, 1.0, (255, 255, 0), 2)
                    #     #vector = np.random.rand(3)
                    #     self.update_frame.emit(frame)
                    #     continue
                    
                    print("Query result: {}".format(query_res))
                    already_name = None
                    for i, res in enumerate(query_res):
                        dist = res[0]["distance"]
                        if dist < self.threshold:
                            already_name = res[0]["entity"]["name"]
                            #already_name = "1"
                            cv2.putText(frame, "Are you registered as {}?".format(already_name), (int(0.1 * frame.shape[0]), int(0.1 * frame.shape[1])), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 1)
                        break
                    
                    if already_name is not None:
                        self.update_frame.emit(frame)
                        continue
                    
                    if vector is None:
                        vector = features[0]
                    satisfy = True
                    cv2.putText(frame, "Type your name and confirm", (int(0.1 * frame.shape[0]), int(0.1 * frame.shape[1])), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 1)
                    self.update_frame.emit(frame)
                    
                    if vector is not None and satisfy:
                        self.update_vector.emit(vector)
            except Exception as e:
                print("Exception: {}".format(e))
                self.update_frame.emit(frame)
                continue
            
            #cv2.putText(frame, "Register Mode", (int(0.1 * frame.shape[0]), int(0.1 * frame.shape[1])), cv2.FONT_HERSHEY_COMPLEX, 1.0, (255, 255, 0), 2)
            
            #self.update_frame.emit(frame, vector, satisfy)
            
        stream.stop()
        
    
    def set_start(self):
        self.stopped = False
    
        
    def stop(self):
        self.stopped = True
            
        