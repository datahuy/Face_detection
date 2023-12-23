from collections import OrderedDict
import time
import numpy as np
from scipy.optimize import linear_sum_assignment
import cv2

from face_tracking.face_track import TentativeTrack, FaceTrack
from face_tracking.utils import iou_mat


class FaceTracker:

    def __init__(self,
                 inactive_steps_before_removed,
                 reid_iou_threshold,
                 max_traject_steps,
                 tentative_steps_before_accepted=30):

        #self.face_detector = face_detector
        #self.regression_face_thresh = regression_face_thresh
        #self.detection_nms_thresh = detection_nms_thresh
        #self.regression_nms_thresh = regression_nms_thresh
        self.inactive_steps_before_removed = inactive_steps_before_removed
        self.reid_iou_threshold = reid_iou_threshold
        #self.do_align = do_align
        self.max_traject_steps = max_traject_steps
        self.tentative_steps_before_accepted = tentative_steps_before_accepted
        #self.shape = shape
        self.det_id_to_track_id_match_dict = OrderedDict()

        self.tentative_tracks = []
        self.active_tracks = []
        self.inactive_tracks = []
        self.tentative_track_num = 0
        self.track_num = 0
        #self.im_index = 0
        #self.results = {}

    def reset(self, hard=True):
        self.active_tracks = []
        self.inactive_tracks = []

        if hard:
            self.tentative_track_num = 0
            self.track_num = 0
            #self.results = {}
            #self.im_index = 0


    def reset_match_dict(self):
        self.det_id_to_track_id_match_dict = OrderedDict()

    
    def tracks_to_inactive(self, tracks):
        self.active_tracks = [t for t in self.active_tracks if t not in tracks]
        for t in tracks:
            t.bbox = t.traject_pos[-1]
        self.inactive_tracks += tracks

    
    def tentative_tracks_to_active(self, tracks):
        """Initializes new Track objects and save them"""
        num_new = len(tracks)
        for i in range(num_new):
            self.active_tracks.append(FaceTrack(bbox=tracks[i].bbox,
                                                landmark=tracks[i].landmark,
                                                track_id=self.track_num + i,
                                                inactive_steps_before_removed=self.inactive_steps_before_removed,
                                                traject_pos=tracks[i].traject_pos,
                                                traject_vel=tracks[i].traject_vel,
                                                time_stamp=tracks[i].time_stamp))
        self.track_num += num_new


    def add_new_tentative_tracks(self, face_detections_list):

        num_new = len(face_detections_list)

        for i in range(num_new):
            self.tentative_tracks.append(TentativeTrack(bbox=face_detections_list[i].bbox,
                                                        landmark=face_detections_list[i].landmark,
                                                        tentative_id=self.tentative_track_num + 1,
                                                        tentative_steps_before_accepted=self.tentative_steps_before_accepted,
                                                        max_traject_steps=self.max_traject_steps))

        
        self.tentative_track_num += num_new


    def get_tentative_boxes(self):
        if len(self.tentative_tracks) >= 1:
            bboxes = [t.bbox for t in self.tentative_tracks]
        else:
            bboxes = []
        return bboxes


    def get_active_boxes(self):
        if len(self.active_tracks) >= 1:
            bboxes = [t.bbox for t in self.active_tracks]
        else:
            bboxes = []
        return bboxes


    def get_inactive_boxes(self):
        if len(self.inactive_tracks) >= 1:
            bboxes = [t.bbox for t in self.inactive_tracks]
        else:
            bboxes = []
        return bboxes


    def match_reid_iou_tentative(self, face_detections):
        tentative_tracks_bboxes = self.get_tentative_boxes()
        detections_bboxes = [detection.bbox for detection in face_detections]

        iou_matrix = iou_mat(tentative_tracks_bboxes, detections_bboxes)
        iou_matrix = np.where(iou_matrix > self.reid_iou_threshold, iou_matrix, 0)

        row_ind, col_ind = linear_sum_assignment(iou_matrix, maximize=True)

        matched_detection_indx = []
        matched_track_indx = []

        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] >= self.reid_iou_threshold:
                matched_track_indx.append(r)
                matched_detection_indx.append(c)
                t = self.tentative_tracks[r]
                t.bbox = face_detections[c].bbox
                t.landmark = face_detections[c].landmark
                t.time_stamp.append(time.time())
        unmatched_tracks = [self.tentative_tracks[i] for i in range(len(self.tentative_tracks)) if i not in matched_track_indx]
        unmatched_detections = [face_detections[j] for j in range(len(face_detections)) if j not in matched_detection_indx]

        #self.tentative_tracks = [self.tentative_tracks[k] for k in matched_track_indx]
        for t in unmatched_tracks:
            self.tentative_tracks.remove(t)

        return unmatched_detections

    
    def match_reid_iou_active(self, face_detections):
        active_tracks_bboxes = self.get_active_boxes()
        detections_bboxes = [detection.bbox for detection in face_detections]

        iou_matrix = iou_mat(active_tracks_bboxes, detections_bboxes)
        iou_matrix = np.where(iou_matrix > self.reid_iou_threshold, iou_matrix, 0)

        row_ind, col_ind = linear_sum_assignment(iou_matrix, maximize=True)

        matched_detection_indx = []
        matched_track_indx = []

        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] >= self.reid_iou_threshold:
                matched_track_indx.append(r)
                matched_detection_indx.append(c)
                t = self.active_tracks[r]
                t.bbox = face_detections[c].bbox
                t.landmark = face_detections[c].landmark
                t.time_stamp.append(time.time())
                self.det_id_to_track_id_match_dict[face_detections[c].detection_id] = t.track_id
        unmatched_tracks = [self.active_tracks[i] for i in range(len(self.active_tracks)) if i not in matched_track_indx]
        unmatched_detections = [face_detections[j] for j in range(len(face_detections)) if j not in matched_detection_indx] # Huhu, đáng ra phải là matched_detection_indx nhưng viết nhầm thành matched_track_indx, giờ mới phát hiện ra

        self.active_tracks = [self.active_tracks[k] for k in matched_track_indx]

        for track in unmatched_tracks:
            track.alive_time.append(time.time() - track.birth_time[-1])

        return unmatched_tracks, unmatched_detections


    def match_reid_iou_inactive(self, face_detections):
        inactivate_tracks_bboxes = self.get_inactive_boxes()
        detections_bboxes = [detection.bbox for detection in face_detections]

        iou_matrix = iou_mat(inactivate_tracks_bboxes, detections_bboxes)
        iou_matrix = np.where(iou_matrix > self.reid_iou_threshold, iou_matrix, 0)

        row_ind, col_ind = linear_sum_assignment(iou_matrix, maximize=True)

        matched_detection_idx = []
        matched_track_idx = []

        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] >= self.reid_iou_threshold: # Huhu đáng ra phải là self.inactive_tracks[r].class_id nhưng viết nhầm thành self.active_tracks[r].class_id
                matched_track_idx.append(r)
                matched_detection_idx.append(c)
                t = self.inactive_tracks[r]
                t.bbox = face_detections[c].bbox
                t.landmark = face_detections[c].landmark
                t.reset_trajectory()
                t.inactive_steps = 0
                t.birth_time.append(time.time())
                self.active_tracks.append(t)
                self.det_id_to_track_id_match_dict[face_detections[c].detection_id] = t.track_id

        matched_tracks = [self.inactive_tracks[i] for i in range(len(self.inactive_tracks)) if i in matched_track_idx]
        unmatched_detections = [face_detections[j] for j in range(len(face_detections)) if j not in matched_detection_idx]

        for t in matched_tracks:
            self.inactive_tracks.remove(t)

        return unmatched_detections

    
    def motion_step(self, track):
        track.bbox = track.bbox + track.traject_vel[-1] * (time.time() - track.time_stamp[-1])


    def motion(self):
        for t in self.active_tracks:
            if len(t.traject_pos) == 1:
                continue
            last_bboxes = t.traject_pos
            moments = t.time_stamp

            vs = np.asarray([(p2 - p1) / (t2 - t1) for p1, p2, t1, t2 in zip(list(last_bboxes), list(last_bboxes)[1:], list(moments), list(moments)[1:])], dtype=np.float)
            vs = np.mean(vs)

            t.traject_vel.append(vs)

            self.motion_step(t)

        for t in self.tentative_tracks:
            if len(t.traject_pos) == 1:
                continue
            last_bboxes = t.traject_pos
            moments = t.time_stamp

            vs = np.asarray([(p2 - p1) / (t2 - t1) for p1, p2, t1, t2 in zip(list(last_bboxes), list(last_bboxes)[1:], list(moments), list(moments)[1:])], dtype=np.float)
            vs = np.mean(vs)

            t.traject_vel.append(vs)

            self.motion_step(t)


    def step(self, face_detections):
        self.reset_match_dict()

        for t in self.active_tracks:
            if len(t.traject_pos) > 1:
                t.traject_pos.append(t.bbox.copy())
            else: # If len of t.traject_pos is 1 means that track just been created and t.bbox was already appended to t.traject_ps
                pass

        for t in self.tentative_tracks:
            if len(t.traject_pos) > 1:
                t.traject_pos.append(t.bbox.copy())
            else:
                pass

        self.motion()

        if len(self.tentative_tracks) > 0 and len(face_detections) > 0:
            #print("14, {}, {}".format(len(self.tentative_tracks), len(face_detections)), end=", ")
            face_detections = self.match_reid_iou_tentative(face_detections=face_detections)
            #print("{}, {}".format(len(self.tentative_tracks), len(face_detections)))
        elif len(self.tentative_tracks) > 0 and len(face_detections) == 0:
            #print("15, {}, {}".format(len(self.tentative_tracks), len(face_detections)))
            self.tentative_tracks.clear()

        if len(self.active_tracks) > 0 and len(face_detections) > 0:
            unmatched_active_tracks, unmatched_detections = self.match_reid_iou_active(face_detections=face_detections)

            if len(unmatched_detections) > 0:
                if len(self.inactive_tracks) > 0: # Phải sửa từ active thành inactive
                    #print("1")
                    unmatched_detections = self.match_reid_iou_inactive(face_detections=unmatched_detections)
                else:
                    #print("2")
                    pass
            else:
                #print("3")
                pass

            if len(unmatched_active_tracks) > 0:
                #print("4")
                self.tracks_to_inactive(unmatched_active_tracks)
            else:
                #print("5")
                pass

            if len(unmatched_detections) > 0:
                #print("6")
                self.add_new_tentative_tracks(face_detections_list=unmatched_detections)
            else:
                #print("7")
                pass

        elif len(self.active_tracks) > 0 and len(face_detections) == 0:
            #print("8")
            self.tracks_to_inactive(self.active_tracks)

        elif len(self.active_tracks) == 0 and len(face_detections) > 0:
            if len(self.inactive_tracks) > 0:
                #print("9")
                unmatched_detections = self.match_reid_iou_inactive(face_detections=face_detections)
                if len(unmatched_detections) > 0:
                    #print("10")
                    self.add_new_tentative_tracks(face_detections_list=unmatched_detections)
                else:
                    #print("11")
                    pass
            else:
                #print("12")
                self.add_new_tentative_tracks(face_detections_list=face_detections)
        else:
            #print("13")
            pass

        tentative_to_active = []
        for t in self.tentative_tracks:
            t.tentative_steps += 1
            if t.tentative_steps > t.tentative_steps_before_accepted:
                tentative_to_active.append(t)
        self.tentative_tracks_to_active(tentative_to_active)
        for track in tentative_to_active:
            self.tentative_tracks.remove(track)

        remove_inactive = []
        for t in self.inactive_tracks:
            t.inactive_steps += 1
            if t.inactive_steps > t.inactive_steps_before_removed:
                remove_inactive.append(t)

        for track in remove_inactive:
            self.inactive_tracks.remove(track)


    def get_result(self):
        return self.active_tracks

    def get_dict_convert_col_to_track_id(self):
        return dict(zip(list(range(len(self.square_of_mask.keys()))), list(self.square_of_mask.keys())))