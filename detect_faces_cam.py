import os

import numpy as np
from PIL import Image
import cv2
import torch
# from mtcnn import MTCNN
# from mtcnn_pytorch.src.visualization_utils import show_bboxes

# from retinaface.RetinaFace import RetinaModel
import face_alignment

from videostream.videostream import QueuedStream

from face_tracking.face_detection import FaceDetection
from face_tracking.face_tracker import FaceTracker

from align.face_align import *
from utils import face_orientation, compute_head_pose, alignment_procedure


if __name__ == "__main__":

    save_dir = "image"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    face_id = 0
    
    extend = 0.0
    device = "cuda" if torch.cuda.is_available() else "cpu"

    #mtcnn = MTCNN()
    #retina = RetinaModel()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fa = face_alignment.FaceAlignment(landmarks_type=face_alignment.LandmarksType.TWO_D, device=device, face_detector="sfd")

    stream = QueuedStream(uri=[], drop=True, fps=25)
    tracker = FaceTracker(inactive_steps_before_removed=20,
                          reid_iou_threshold=0.6,
                          max_traject_steps=30,
                          tentative_steps_before_accepted=3)
    stream.start()

    while True:
        ret, frame, frame_id = stream.read()

        if not ret:
            break
        
        # resp = retina.detect_faces(frame)
        # print(resp)
        # faces = retina.extract_faces(frame, obj=resp)
        
        
        # image = Image.fromarray(np.uint8(frame)).convert('RGB')
        # bboxes, landmarks = mtcnn.detect_faces(image)
        
        #bboxes, faces = mtcnn.align_intermediate(img=image, boxes=bboxes, landmarks=landmarks)
        
        dets = fa.get_landmarks(frame, return_bboxes=True, return_landmark_score=True)
        #print("det: {}".format(dets))
        bboxes = dets[2]
        landmarks = dets[0]
        #print("landmarks: {}".format(len(landmarks)))        
        # for face in faces:
        #     face_id += 1
        #     face_img = np.asarray(face)
        #     cv2.imwrite(os.path.join(save_dir, "face_{}.jpg".format(face_id)), face_img)
        
        # #print(bboxes)
        # draw_img = show_bboxes(img=image, bounding_boxes=bboxes, facial_landmarks=landmarks)
        # frame = np.asarray(draw_img)
        
        try:
            detections_list = []
            for i, (landmark, bbox) in enumerate(zip(landmarks, bboxes)):
                #print(bbox.shape)
                x_1, y_1, x_2, y_2, prob = bbox
                if prob > 0.8:
                    w = x_2 - x_1
                    h = y_2 - y_1
                    # new_x_1 = max(x_1 - extend * w, 0)
                    # new_x_2 = min(x_2 + extend * w, frame.shape[1] - 1)
                    # new_y_1 = max(y_1 - extend * h, 0)
                    # new_y_2 = min(y_2 + extend * h, frame.shape[0] - 1)
                    # cv2.rectangle(frame, (int(new_x_1), int(new_y_1)), (int(new_x_2), int(new_y_2)), color=(255, 255, 0), thickness=1)
                    #landmark = list(map(lambda x: [x[0] - x_1, x[1] - y_1], landmark))
                    detections_list.append(FaceDetection(bbox=bbox, landmark=landmark, detection_id=i))
                    #print("landmark: {}".format(landmark.shape))
                    roll, pitch, yaw = face_orientation(frame=frame, landmark=landmark)
                    #roll, pitch, yaw = compute_head_pose(landmark=landmark)
                    print("roll: {}, pitch: {}, yaw: {}".format(roll, pitch, yaw))
                    #print(y_1, y_2, x_1, x_2)
                    # face_img = frame[int(y_1):int(y_2), int(x_1):int(x_2)]
                    # right_eye = landmark[45]
                    # left_eye = landmark[36]
                    # nose = landmark[30]
                    # face_img = alignment_procedure(img=face_img, left_eye=left_eye, right_eye=right_eye, nose=nose)
                    # face_id += 1
                    # cv2.imwrite(os.path.join(save_dir, "face_{}.jpg".format(face_id)), face_img)
                    # for j, point in enumerate(landmark):
                    #     if j in [8, 30, 36, 45, 48, 54]:
                    #         # 8: Cằm
                    #         # 30: Mũi
                    #         # 36: Mắt trái
                    #         # 45: Mắt phải
                    #         # 48: Miệng trái
                    #         # 54: Miệng phải
                    #         cv2.circle(frame, (int(point[0]), int(point[1])), 2, (0, 0, 255), -1)
                    #         cv2.putText(frame, str(j), (int(point[0]) + 3, int(point[1]) - 3),  cv2.FONT_HERSHEY_COMPLEX, 0.3, (255, 255, 0), 1)            
            tracker.step(face_detections=detections_list)
            face_tracks = tracker.get_result()
            frame_copy = frame.copy()
            for face_track in face_tracks:
                bbox = face_track.bbox
                landmark = face_track.landmark
                #print(bbox)
                track_id = face_track.track_id
                x_min, y_min, x_max, y_max, prob = bbox
                x_min = int(x_min)
                x_max = int(x_max)
                y_min = int(y_min)
                y_max = int(y_max)
                c_x = int((x_min + x_max)/2)
                c_y = int((y_min + y_max)/2)
                if h > w:
                    h_max = True
                else:
                    h_max = False
                w = x_max - x_min
                h = y_max - y_min
                
                print(x_min, y_min, x_max, y_max, track_id)
                #ext = 0
                #face_img = frame_copy[max(y_min - int(ext*(y_max - y_min)), 0):min(y_max + int(ext*(y_max - y_min)), frame.shape[0]), max(x_min - int(ext * (x_max - x_min)), 0):min(x_max + int(ext * (x_max - x_min)), frame.shape[1])]
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
                face_id += 1
                cv2.imwrite(os.path.join(save_dir, "face_{}.jpg".format(face_id)), face_img)
                result = "Track id: {}".format(track_id)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 255, 0), 1)
                cv2.putText(frame, result, (x_min + int(0.05*(x_max - x_min)), y_min - int(0.05*(y_max - y_min))), cv2.FONT_HERSHEY_COMPLEX, 0.3, (255, 255, 0), 1)

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
        except Exception as e:
            print("Exception: {}".format(e))
            continue
        
        cv2.imshow("", frame)
        cv2.waitKey(1)

        #print(stream.fps)