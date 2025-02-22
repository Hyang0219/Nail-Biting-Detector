import cv2
import mediapipe as mp
import numpy as np
from enum import Enum
from datetime import datetime, timedelta
import math

class GestureState(Enum):
    IDLE = 0
    HAND_NEAR_MOUTH = 1
    POTENTIAL_BITING = 2
    BITING = 3
    COOLDOWN = 4

class GestureDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize MediaPipe solutions
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Mouth landmarks indices (upper and lower lip)
        self.MOUTH_LANDMARKS = [13, 14, 78, 308, 415, 318, 324, 402, 317, 14, 87, 178, 88, 95]
        
        # Fingertip indices in MediaPipe hand landmarks
        self.FINGERTIPS = [4, 8, 12, 16, 20]  # thumb to pinky
        self.FINGER_MIDS = [3, 7, 11, 15, 19]  # mid points of fingers
        
        # State management
        self.current_state = GestureState.IDLE
        self.last_detection_time = None
        self.cooldown_period = timedelta(seconds=3)
        
        # Detection parameters
        self.distance_threshold = 0.15  # relative to frame height
        self.angle_threshold = 60  # degrees
        self.consecutive_frames_threshold = 3
        self.consecutive_detections = 0
    
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
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        return np.degrees(angle)
    
    def is_finger_pointing_to_mouth(self, fingertip, finger_mid, mouth_center):
        """Check if finger is pointing towards mouth."""
        angle = self.calculate_angle(fingertip, finger_mid, mouth_center)
        return angle < self.angle_threshold
    
    def update_state(self, close_fingers, pointing_fingers):
        """Update the gesture state based on current detection and history."""
        now = datetime.now()
        
        # Handle cooldown period
        if self.current_state == GestureState.COOLDOWN:
            if self.last_detection_time and now - self.last_detection_time > self.cooldown_period:
                self.current_state = GestureState.IDLE
                self.consecutive_detections = 0
            return False
        
        # State machine logic
        if close_fingers > 0 and pointing_fingers > 0:
            self.consecutive_detections += 1
            if self.consecutive_detections >= self.consecutive_frames_threshold:
                if self.current_state in [GestureState.POTENTIAL_BITING, GestureState.HAND_NEAR_MOUTH]:
                    self.current_state = GestureState.BITING
                    self.last_detection_time = now
                    self.current_state = GestureState.COOLDOWN
                    self.consecutive_detections = 0
                    return True
                elif self.current_state == GestureState.IDLE:
                    self.current_state = GestureState.POTENTIAL_BITING
        else:
            self.consecutive_detections = max(0, self.consecutive_detections - 1)
            if close_fingers > 0:
                self.current_state = GestureState.HAND_NEAR_MOUTH
            else:
                self.current_state = GestureState.IDLE
        
        return False
    
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
        
        # Get mouth position if face is detected
        mouth_center = None
        if face_results.multi_face_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0]
            mouth_points = np.array([
                [face_landmarks.landmark[idx].x * frame.shape[1],
                 face_landmarks.landmark[idx].y * frame.shape[0]]
                for idx in self.MOUTH_LANDMARKS
            ], dtype=np.int32)
            
            mouth_x, mouth_y, mouth_w, mouth_h = cv2.boundingRect(mouth_points)
            mouth_center = (mouth_x + mouth_w//2, mouth_y + mouth_h//2)
            
            # Draw mouth rectangle
            cv2.rectangle(frame_with_detections, 
                        (mouth_x, mouth_y), 
                        (mouth_x + mouth_w, mouth_y + mouth_h),
                        (0, 255, 0), 2)
        
        close_fingers = 0
        pointing_fingers = 0
        
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
                
                # Check each fingertip
                for tip_idx, mid_idx in zip(self.FINGERTIPS, self.FINGER_MIDS):
                    fingertip = (hand_landmarks.landmark[tip_idx].x * frame.shape[1],
                               hand_landmarks.landmark[tip_idx].y * frame.shape[0])
                    finger_mid = (hand_landmarks.landmark[mid_idx].x * frame.shape[1],
                                hand_landmarks.landmark[mid_idx].y * frame.shape[0])
                    
                    # Calculate distance to mouth (relative to frame height)
                    distance = self.calculate_distance(fingertip, mouth_center) / frame.shape[0]
                    
                    if distance < self.distance_threshold:
                        close_fingers += 1
                        if self.is_finger_pointing_to_mouth(fingertip, finger_mid, mouth_center):
                            pointing_fingers += 1
                
                # Update detection state
                is_biting = self.update_state(close_fingers, pointing_fingers)
                
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
                metrics_text = f"Close: {close_fingers}, Pointing: {pointing_fingers}"
                cv2.putText(frame_with_detections,
                          metrics_text,
                          (10, 60),
                          cv2.FONT_HERSHEY_SIMPLEX,
                          1,
                          color,
                          2)
        
        return frame_with_detections, is_biting 