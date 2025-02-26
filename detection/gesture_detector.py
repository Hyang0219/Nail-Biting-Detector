import cv2
import mediapipe as mp
import numpy as np
from enum import Enum
from datetime import datetime, timedelta
import math
import tensorflow as tf
import os
import logging

class GestureState(Enum):
    IDLE = 0
    HAND_NEAR_MOUTH = 1
    POTENTIAL_BITING = 2
    BITING = 3
    COOLDOWN = 4

class GestureDetector:
    def __init__(self, model_path=None, sensitivity=0.5):
        self.mp_hands = mp.solutions.hands
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize MediaPipe solutions
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Mouth landmarks indices (upper and lower lip)
        self.MOUTH_LANDMARKS = [13, 14, 78, 308, 415, 318, 324, 402, 317, 14, 87, 178, 88, 95]
        
        # Fingertip indices in MediaPipe hand landmarks
        self.FINGERTIPS = [4, 8, 12, 16, 20]  # thumb to pinky
        self.FINGER_MIDS = [3, 7, 11, 15, 19]  # mid points of fingers
        self.FINGER_BASES = [2, 6, 10, 14, 18]  # base points of fingers
        
        # State management
        self.current_state = GestureState.IDLE
        self.last_detection_time = None
        self.cooldown_period = timedelta(seconds=2)  # Reduced from 3 to 2
        
        # Make thresholds adjustable based on sensitivity
        self.update_sensitivity(sensitivity)
        
        # Detection parameters
        self.consecutive_frames_threshold = 2  # Reduced from 3 to 2
        self.consecutive_detections = 0
        
        # Load ML model if available
        self.model = None
        if model_path is None:
            # Try to find the latest model in the models directory
            models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models')
            if os.path.exists(models_dir):
                model_files = [f for f in os.listdir(models_dir) if f.endswith('.keras')]
                if model_files:
                    latest_model = max(model_files)
                    model_path = os.path.join(models_dir, latest_model)
        
        if model_path and os.path.exists(model_path):
            try:
                self.model = tf.keras.models.load_model(model_path)
                logging.info(f"Loaded ML model from {model_path}")
            except Exception as e:
                logging.error(f"Failed to load ML model: {e}")
        
        self.input_size = (224, 224)  # Model input size
    
    def update_sensitivity(self, sensitivity):
        """Update detection thresholds based on sensitivity value (0.0 to 1.0)."""
        # Interpolate thresholds based on sensitivity
        # Higher sensitivity = larger thresholds = easier to trigger
        self.distance_threshold = 0.15 + (sensitivity * 0.15)  # 0.15 to 0.30
        self.angle_threshold = 45 + (sensitivity * 45)  # 45 to 90 degrees
        self.cluster_threshold = 0.05 + (sensitivity * 0.10)  # 0.05 to 0.15
        self.ml_confidence_threshold = max(0.3, 0.8 - (sensitivity * 0.5))  # 0.8 to 0.3
    
    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points."""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def calculate_angle(self, point1, point2, point3):
        """Calculate angle between three points (in degrees)."""
        a = np.array(point1)
        b = np.array(point2)
        c = np.array(point3)
        
        ba = a - b
        bc = c - b
        
        # Calculate angle using dot product
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        # Convert to degrees
        return np.degrees(angle)
    
    def is_finger_bent(self, tip, mid, base):
        """Check if finger is bent (indicating potential biting)."""
        angle = self.calculate_angle(tip, mid, base)
        # Consider finger bent if angle between segments is less than 160 degrees
        return angle < 160  # Bent if angle is less than 160 degrees
    
    def are_fingertips_clustered(self, fingertips, frame_height):
        """Check if fingertips are clustered together near mouth."""
        if len(fingertips) < 2:
            return False
            
        # Calculate pairwise distances between fingertips
        distances = []
        for i in range(len(fingertips)):
            for j in range(i + 1, len(fingertips)):
                dist = self.calculate_distance(fingertips[i], fingertips[j])
                distances.append(dist / frame_height)
        
        # Check if maximum distance between any two fingertips is small enough
        return max(distances) if distances else False < self.cluster_threshold
    
    def is_hand_near_mouth(self, hand_landmarks, mouth_center, frame_height):
        """Check if any fingertip is near the mouth."""
        near_fingers = []
        for tip_idx in self.FINGERTIPS:
            tip = (hand_landmarks.landmark[tip_idx].x * frame_height * 4/3,  # Adjust for aspect ratio
                  hand_landmarks.landmark[tip_idx].y * frame_height)
            distance = self.calculate_distance(tip, mouth_center) / frame_height
            if distance < self.distance_threshold:
                near_fingers.append(tip)
        return near_fingers
    
    def is_finger_pointing_to_mouth(self, tip, mid, mouth_center):
        """Check if finger is oriented towards mouth."""
        angle = self.calculate_angle(tip, mid, mouth_center)
        return angle < self.angle_threshold
    
    def update_state(self, near_fingers, pointing_fingers, bent_fingers, ml_prediction=None):
        """Update the gesture state based on current detection and history."""
        now = datetime.now()
        
        # Handle cooldown period
        if self.current_state == GestureState.COOLDOWN:
            if self.last_detection_time and now - self.last_detection_time > self.cooldown_period:
                self.current_state = GestureState.IDLE
                self.consecutive_detections = 0
            return False
        
        # State machine logic with ML integration
        if len(near_fingers) > 0:
            if len(pointing_fingers) > 0 and len(bent_fingers) > 0:
                self.consecutive_detections += 1
                if self.consecutive_detections >= self.consecutive_frames_threshold:
                    # If ML model is available, use it to confirm the detection
                    if ml_prediction is not None and ml_prediction > self.ml_confidence_threshold:
                        if self.current_state in [GestureState.POTENTIAL_BITING, GestureState.HAND_NEAR_MOUTH]:
                            self.current_state = GestureState.BITING
                            self.last_detection_time = now
                            self.current_state = GestureState.COOLDOWN
                            self.consecutive_detections = 0
                            return True
                    elif ml_prediction is None:  # Fall back to geometric approach if no ML model
                        if self.current_state in [GestureState.POTENTIAL_BITING, GestureState.HAND_NEAR_MOUTH]:
                            self.current_state = GestureState.BITING
                            self.last_detection_time = now
                            self.current_state = GestureState.COOLDOWN
                            self.consecutive_detections = 0
                            return True
                self.current_state = GestureState.POTENTIAL_BITING
            else:
                self.current_state = GestureState.HAND_NEAR_MOUTH
                self.consecutive_detections = max(0, self.consecutive_detections - 1)
        else:
            self.current_state = GestureState.IDLE
            self.consecutive_detections = 0
        
        return False
    
    def get_roi_for_model(self, frame, hand_landmarks, mouth_bbox):
        """Extract and preprocess ROI for ML model."""
        try:
            # Convert landmarks to numpy array
            points = np.array([[lm.x * frame.shape[1], lm.y * frame.shape[0]] 
                             for lm in hand_landmarks.landmark], dtype=np.int32)
            
            # Get bounding box around hand
            x, y, w, h = cv2.boundingRect(points)
            
            # Expand box to include mouth area if close
            mx, my, mw, mh = mouth_bbox
            combined_x = min(x, mx)
            combined_y = min(y, my)
            combined_w = max(x + w, mx + mw) - combined_x
            combined_h = max(y + h, my + mh) - combined_y
            
            # Add padding
            pad = 20
            combined_x = max(0, combined_x - pad)
            combined_y = max(0, combined_y - pad)
            combined_w = min(frame.shape[1] - combined_x, combined_w + 2*pad)
            combined_h = min(frame.shape[0] - combined_y, combined_h + 2*pad)
            
            # Crop region
            roi = frame[combined_y:combined_y+combined_h, 
                      combined_x:combined_x+combined_w]
            
            # Resize and preprocess for model
            if roi.size > 0:
                roi = cv2.resize(roi, self.input_size)
                roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                roi = roi.astype(np.float32) / 255.0
                return roi
            
            return None
        except Exception as e:
            logging.error(f"Error preparing ROI: {e}")
            return None
    
    def process_frame(self, frame):
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process hands
        hand_results = self.hands.process(rgb_frame)
        
        # Process face
        face_results = self.face_mesh.process(rgb_frame)
        
        # Draw initial rectangles
        frame_with_detections = frame.copy()
        is_biting = False
        ml_prediction = None
        
        # Get mouth position if face is detected
        mouth_center = None
        mouth_bbox = None
        if face_results.multi_face_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0]
            mouth_points = np.array([
                [face_landmarks.landmark[idx].x * frame.shape[1],
                 face_landmarks.landmark[idx].y * frame.shape[0]]
                for idx in self.MOUTH_LANDMARKS
            ], dtype=np.int32)
            
            mouth_x, mouth_y, mouth_w, mouth_h = cv2.boundingRect(mouth_points)
            mouth_center = (mouth_x + mouth_w//2, mouth_y + mouth_h//2)
            mouth_bbox = (mouth_x, mouth_y, mouth_w, mouth_h)
            
            # Draw mouth rectangle
            cv2.rectangle(frame_with_detections, 
                        (mouth_x, mouth_y), 
                        (mouth_x + mouth_w, mouth_y + mouth_h),
                        (0, 255, 0), 2)
        
        # Process hands and check for nail biting
        if hand_results.multi_hand_landmarks and mouth_center:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                # Get hand bounding box for visualization
                hand_points = np.array([
                    [hand_landmarks.landmark[i].x * frame.shape[1],
                     hand_landmarks.landmark[i].y * frame.shape[0]]
                    for i in range(21)
                ], dtype=np.int32)
                
                hand_x, hand_y, hand_w, hand_h = cv2.boundingRect(hand_points)
                
                # Check for nail biting gestures
                near_fingers = self.is_hand_near_mouth(hand_landmarks, mouth_center, frame.shape[0])
                
                pointing_fingers = []
                bent_fingers = []
                
                for tip_idx, mid_idx, base_idx in zip(self.FINGERTIPS, self.FINGER_MIDS, self.FINGER_BASES):
                    tip = (hand_landmarks.landmark[tip_idx].x * frame.shape[1],
                          hand_landmarks.landmark[tip_idx].y * frame.shape[0])
                    mid = (hand_landmarks.landmark[mid_idx].x * frame.shape[1],
                          hand_landmarks.landmark[mid_idx].y * frame.shape[0])
                    base = (hand_landmarks.landmark[base_idx].x * frame.shape[1],
                           hand_landmarks.landmark[base_idx].y * frame.shape[0])
                    
                    if self.is_finger_pointing_to_mouth(tip, mid, mouth_center):
                        pointing_fingers.append(tip)
                    
                    if self.is_finger_bent(tip, mid, base):
                        bent_fingers.append(tip)
                
                # Check if detected fingers are clustered
                clustered = self.are_fingertips_clustered(near_fingers, frame.shape[0])
                
                # Use ML model if available
                if self.model is not None and len(near_fingers) > 0:
                    roi = self.get_roi_for_model(frame, hand_landmarks, mouth_bbox)
                    if roi is not None:
                        # Add batch dimension
                        roi = np.expand_dims(roi, 0)
                        # Get model prediction
                        ml_prediction = float(self.model.predict(roi, verbose=0)[0])
                
                # Update detection state
                is_biting = self.update_state(near_fingers, pointing_fingers, bent_fingers, ml_prediction)
                
                # Draw hand rectangle with appropriate color
                color = (0, 0, 255) if is_biting else (255, 0, 0)
                cv2.rectangle(frame_with_detections,
                            (hand_x, hand_y),
                            (hand_x + hand_w, hand_y + hand_h),
                            color, 2)
                
                # Draw detection status
                status_text = f"State: {self.current_state.name}"
                cv2.putText(frame_with_detections,
                          status_text,
                          (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX,
                          1,
                          color,
                          2)
                
                # Draw detection metrics
                metrics_text = f"Near: {len(near_fingers)}, Pointing: {len(pointing_fingers)}, Bent: {len(bent_fingers)}"
                if ml_prediction is not None:
                    metrics_text += f", ML: {ml_prediction:.2f}"
                cv2.putText(frame_with_detections,
                          metrics_text,
                          (10, 60),
                          cv2.FONT_HERSHEY_SIMPLEX,
                          1,
                          color,
                          2)
        
        return frame_with_detections, is_biting 