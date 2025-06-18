import cv2
import numpy as np
import math
from ultralytics import YOLO
import logging
import mediapipe as mp
from django.conf import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChokingAnalyzer:
    """Analyzes choking signs based on pose keypoints and features"""

    def __init__(self):
        self.KEYPOINTS = {
            'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
            'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8,
            'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11, 'right_hip': 12,
            'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16
        }
        self.WRIST_NECK_THRESHOLD = 0.12
        self.CONFIDENCE_THRESHOLD = 0.3
        self.eye_closed_threshold = 15  # pixels

    def calculate_neck_center(self, keypoints, confidences):
        left_shoulder = keypoints[self.KEYPOINTS['left_shoulder']]
        right_shoulder = keypoints[self.KEYPOINTS['right_shoulder']]
        left_shoulder_conf = confidences[self.KEYPOINTS['left_shoulder']]
        right_shoulder_conf = confidences[self.KEYPOINTS['right_shoulder']]
        if left_shoulder_conf > self.CONFIDENCE_THRESHOLD and right_shoulder_conf > self.CONFIDENCE_THRESHOLD:
            neck_x = (left_shoulder[0] + right_shoulder[0]) / 2
            neck_y = (left_shoulder[1] + right_shoulder[1]) / 2 + 10  # offset down
            return np.array([neck_x, neck_y]), True
        return None, False

    def calculate_wrist_to_neck_distance(self, wrist_point, neck_center, image_shape):
        if neck_center is None:
            return float('inf')
        distance = np.linalg.norm(np.array(wrist_point) - np.array(neck_center))
        image_diagonal = math.sqrt(image_shape[0] ** 2 + image_shape[1] ** 2)
        return distance / image_diagonal

    def detect_eye_closed(self, keypoints, confidences):
        left_eye = keypoints[self.KEYPOINTS['left_eye']]
        right_eye = keypoints[self.KEYPOINTS['right_eye']]
        left_ear = keypoints[self.KEYPOINTS['left_ear']]
        right_ear = keypoints[self.KEYPOINTS['right_ear']]
        left_eye_conf = confidences[self.KEYPOINTS['left_eye']]
        right_eye_conf = confidences[self.KEYPOINTS['right_eye']]
        left_ear_conf = confidences[self.KEYPOINTS['left_ear']]
        right_ear_conf = confidences[self.KEYPOINTS['right_ear']]
        if (left_eye_conf > self.CONFIDENCE_THRESHOLD and right_eye_conf > self.CONFIDENCE_THRESHOLD and
            left_ear_conf > self.CONFIDENCE_THRESHOLD and right_ear_conf > self.CONFIDENCE_THRESHOLD):
            left_eye_ear_dist = np.linalg.norm(np.array(left_eye) - np.array(left_ear))
            right_eye_ear_dist = np.linalg.norm(np.array(right_eye) - np.array(right_ear))
            if left_eye_ear_dist < self.eye_closed_threshold or right_eye_ear_dist < self.eye_closed_threshold:
                return True
        return False

    def analyze(self, keypoints, confidences, image_shape, finger_tips=None):
        features = {
            'left_wrist_at_neck': False,
            'right_wrist_at_neck': False,
            'both_wrists_at_neck': False,
            'finger_at_neck': False,
            'eyes_closed': False,
            'choking_state': "NORMAL",
            'choking_probability': 0.0
        }
        neck_center, neck_detected = self.calculate_neck_center(keypoints, confidences)
        if not neck_detected:
            return features

        left_wrist = keypoints[self.KEYPOINTS['left_wrist']]
        right_wrist = keypoints[self.KEYPOINTS['right_wrist']]
        left_wrist_conf = confidences[self.KEYPOINTS['left_wrist']]
        right_wrist_conf = confidences[self.KEYPOINTS['right_wrist']]

        if left_wrist_conf > self.CONFIDENCE_THRESHOLD:
            left_dist = self.calculate_wrist_to_neck_distance(left_wrist, neck_center, image_shape)
            if left_dist < self.WRIST_NECK_THRESHOLD:
                features['left_wrist_at_neck'] = True

        if right_wrist_conf > self.CONFIDENCE_THRESHOLD:
            right_dist = self.calculate_wrist_to_neck_distance(right_wrist, neck_center, image_shape)
            if right_dist < self.WRIST_NECK_THRESHOLD:
                features['right_wrist_at_neck'] = True

        features['both_wrists_at_neck'] = features['left_wrist_at_neck'] and features['right_wrist_at_neck']
        features['eyes_closed'] = self.detect_eye_closed(keypoints, confidences)
        # Finger-to-neck logic
        FINGER_NECK_THRESHOLD = 0.10  # You can adjust this threshold
        if finger_tips:
            for fx, fy in finger_tips:
                finger_dist = np.linalg.norm(np.array([fx, fy]) - np.array(neck_center))
                image_diagonal = math.sqrt(image_shape[0] ** 2 + image_shape[1] ** 2)
                norm_finger_dist = finger_dist / image_diagonal
                if norm_finger_dist < FINGER_NECK_THRESHOLD:
                    features['finger_at_neck'] = True
                    break

        # Choking logic
        if features['eyes_closed']:
            features['choking_state'] = "CHOKING (EYES CLOSED)"
            features['choking_probability'] = 1.0
        elif features['both_wrists_at_neck'] or features['finger_at_neck']:
            features['choking_state'] = "CHOKING (HANDS/FINGERS)"
            features['choking_probability'] = 0.9
        elif features['left_wrist_at_neck'] or features['right_wrist_at_neck']:
            features['choking_state'] = "CHOKING (ONE WRIST)"
            features['choking_probability'] = 0.7
        else:
            features['choking_state'] = "NORMAL"
            features['choking_probability'] = 0.0

        return features

class ChokingDetector:
    def __init__(self):
        self.model_path = settings.MODEL_PATHS['choking']
        self.model = YOLO(self.model_path)
        self.analyzer = ChokingAnalyzer()
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
        self.confidence_threshold = 0.5
        self.alert_threshold = 0.7

    def process_frame(self, frame):
        """Process a single frame and return annotated frame with choking detection"""
        try:
            results = self.model(frame, conf=self.confidence_threshold, verbose=False)
            annotated_frame = frame.copy()
            choking_detected = False
            
            for result in results:
                if result.keypoints is not None and result.keypoints.xy is not None and result.keypoints.conf is not None:
                    keypoints = result.keypoints.xy.cpu().numpy()
                    confidences = result.keypoints.conf.cpu().numpy()
                    boxes = result.boxes.xyxy.cpu().numpy() if (result.boxes is not None and len(result.boxes) > 0) else None
                    
                    for i, (kpts, confs) in enumerate(zip(keypoints, confidences)):
                        finger_tips = self.detect_fingers(frame, kpts, confs)
                        features = self.analyzer.analyze(kpts, confs, frame.shape[:2], finger_tips=finger_tips)
                        
                        # Check if choking is detected based on updated logic
                        is_choking = (
                            features['both_wrists_at_neck'] or 
                            features['finger_at_neck'] or 
                            features['left_wrist_at_neck'] or 
                            features['right_wrist_at_neck']
                        )
                        
                        if is_choking:
                            choking_detected = True
                            # Draw bounding box around person if choking detected
                            if boxes is not None and i < len(boxes):
                                x1, y1, x2, y2 = boxes[i].astype(int)
                                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                                cv2.putText(annotated_frame, "CHOKING", (x1, y1-10), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        
                        annotated_frame = self.draw_annotations(
                            annotated_frame, kpts, confs, features
                        )
            
            return annotated_frame
        except Exception as e:
            logger.error(f"Error in process_frame: {e}")
            return frame
    
    def detect_fingers(self, frame, keypoints, confidences):
        """Detect fingers using MediaPipe Hands, returns list of finger tip positions."""
        finger_tips = []
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for idx in [4, 8, 12, 16, 20]:  # Thumb_tip, Index_tip, Middle_tip, Ring_tip, Pinky_tip
                    lm = hand_landmarks.landmark[idx]
                    h, w, _ = frame.shape
                    x, y = int(lm.x * w), int(lm.y * h)
                    finger_tips.append((x, y))
        return finger_tips

    def draw_annotations(self, frame, keypoints, confidences, features):
        for i, (x, y) in enumerate(keypoints):
            if confidences[i] > self.confidence_threshold:
                cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
        
        # Draw neck center
        neck_center, neck_detected = self.analyzer.calculate_neck_center(keypoints, confidences)
        if neck_detected:
            cv2.circle(frame, (int(neck_center[0]), int(neck_center[1])), 8, (255, 0, 255), -1)
        
        # Draw wrist-to-neck lines
        if features['left_wrist_at_neck']:
            lw = keypoints[self.analyzer.KEYPOINTS['left_wrist']]
            cv2.line(frame, (int(lw[0]), int(lw[1])), (int(neck_center[0]), int(neck_center[1])), (0, 0, 255), 2)
        if features['right_wrist_at_neck']:
            rw = keypoints[self.analyzer.KEYPOINTS['right_wrist']]
            cv2.line(frame, (int(rw[0]), int(rw[1])), (int(neck_center[0]), int(neck_center[1])), (0, 0, 255), 2)
        
        # Draw finger tips
        finger_tips = self.detect_fingers(frame, keypoints, confidences)
        for x, y in finger_tips:
            cv2.circle(frame, (x, y), 5, (255, 255, 0), -1)
        
        return frame
