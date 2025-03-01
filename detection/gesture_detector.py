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
        
        # Initialize MediaPipe solutions with more conservative settings
        try:
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,  # Reduced from 2 to 1 for stability
                min_detection_confidence=0.6,  # Increased from 0.5 for more reliable detection
                min_tracking_confidence=0.6,   # Increased from 0.5 for more stable tracking
                model_complexity=0  # Use the lightest model for better performance
            )
            
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                min_detection_confidence=0.6,  # Increased from 0.5
                min_tracking_confidence=0.6,   # Increased from 0.5
                refine_landmarks=False  # Disable landmark refinement for better performance
            )
            logging.info("MediaPipe models initialized successfully")
        except Exception as e:
            logging.error(f"Error initializing MediaPipe models: {e}")
            # Create dummy objects that will safely return empty results
            self.hands = None
            self.face_mesh = None
        
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
        self.input_size = (224, 224)  # Model input size
        self.load_model(model_path)
    
    def load_model(self, model_path=None):
        """Load model for prediction."""
        try:
            if model_path is None:
                # Try to find the latest model in the models directory
                models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
                if os.path.exists(models_dir):
                    model_files = [f for f in os.listdir(models_dir) if f.endswith('.keras') or f.endswith('.h5')]
                    if model_files:
                        latest_model = max(model_files)
                        model_path = os.path.join(models_dir, latest_model)
            
            if model_path and os.path.exists(model_path):
                logging.info(f"Loading model from {model_path}")
                self.model = tf.keras.models.load_model(model_path)
                self.model_loaded = True
                logging.info(f"Loaded ML model from {model_path}")
            else:
                logging.warning("No model specified or found. Using geometric approach only.")
                self.model_loaded = False
                self.model = None
        except Exception as e:
            logging.error(f"Failed to load ML model: {e}")
            self.model_loaded = False
            self.model = None
    
    def update_sensitivity(self, sensitivity):
        """Update detection thresholds based on sensitivity value (0.0 to 1.0)."""
        # Interpolate thresholds based on sensitivity
        # Higher sensitivity = larger thresholds = easier to trigger
        self.distance_threshold = 0.15 + (sensitivity * 0.15)  # 0.15 to 0.30
        self.angle_threshold = 45 + (sensitivity * 45)  # 45 to 90 degrees
        self.cluster_threshold = 0.05 + (sensitivity * 0.10)  # 0.05 to 0.15
        
        # Adjust ML threshold to match the model's output range (around 0.35-0.36)
        # Lower values = more sensitive detection
        self.ml_confidence_threshold = max(0.2, 0.4 - (sensitivity * 0.2))  # 0.4 to 0.2
    
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
        max_distance = max(distances) if distances else float('inf')
        return max_distance < self.cluster_threshold
    
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
        """
        Update the gesture state based on finger positions and ML prediction.
        
        Args:
            near_fingers: List of fingers that are near the mouth
            pointing_fingers: List of fingers that are pointing to the mouth
            bent_fingers: List of fingers that are bent
            ml_prediction: Optional ML model prediction (0-1 value where higher means more likely nail biting)
        """
        # Get current time for timing calculations
        current_time = datetime.now()
        
        # If we're in cooldown, check if we can exit
        if self.current_state == GestureState.COOLDOWN:
            if current_time - self.last_detection_time > self.cooldown_period:
                self.current_state = GestureState.IDLE
                self.consecutive_detections = 0
                logging.debug("Exiting cooldown state")
            return False
        
        # If no fingers are near the mouth, reset to IDLE
        if not near_fingers:
            self.current_state = GestureState.IDLE
            self.last_detection_time = None
            return False
        
        # Check if we have an ML prediction and it's above threshold
        ml_confidence = 0.0
        if ml_prediction is not None:
            ml_confidence = float(ml_prediction)
            logging.debug(f"ML prediction: {ml_confidence:.3f}, threshold: {self.ml_confidence_threshold:.3f}")
        
        # Calculate geometric confidence based on finger positions
        # More near fingers, pointing fingers, and bent fingers increase confidence
        geometric_confidence = (
            (len(near_fingers) / 5.0) * 0.4 +  # Weight for near fingers
            (len(pointing_fingers) / 5.0) * 0.3 +  # Weight for pointing fingers
            (len(bent_fingers) / 5.0) * 0.3  # Weight for bent fingers
        )
        
        # Combine geometric and ML confidence
        # If ML prediction is available, use it as a boost
        combined_confidence = geometric_confidence
        if ml_prediction is not None:
            # Normalize ML prediction to be more sensitive
            # Map the typical range (0.35-0.37) to a wider range (0-0.5)
            normalized_ml = max(0, min(0.5, (ml_confidence - 0.35) * 25))
            combined_confidence = geometric_confidence + normalized_ml
            logging.debug(f"Combined confidence: {combined_confidence:.3f} (Geometric: {geometric_confidence:.3f}, ML: {normalized_ml:.3f})")
        
        # State machine logic
        if self.current_state == GestureState.IDLE:
            if len(near_fingers) >= 1:
                self.current_state = GestureState.HAND_NEAR_MOUTH
                self.last_detection_time = current_time
                logging.debug(f"State change: IDLE -> HAND_NEAR_MOUTH (near fingers: {near_fingers})")
        
        elif self.current_state == GestureState.HAND_NEAR_MOUTH:
            # Check for potential biting
            if combined_confidence >= 0.6:  # Threshold for potential biting
                self.current_state = GestureState.POTENTIAL_BITING
                self.last_detection_time = current_time
                logging.debug(f"State change: HAND_NEAR_MOUTH -> POTENTIAL_BITING (confidence: {combined_confidence:.3f})")
            elif len(near_fingers) == 0:
                self.current_state = GestureState.IDLE
                self.last_detection_time = current_time
                logging.debug("State change: HAND_NEAR_MOUTH -> IDLE (no near fingers)")
        
        elif self.current_state == GestureState.POTENTIAL_BITING:
            # If confidence drops, go back to hand near mouth
            if combined_confidence < 0.5:  # Lower threshold to exit potential biting
                self.current_state = GestureState.HAND_NEAR_MOUTH
                self.last_detection_time = current_time
                logging.debug(f"State change: POTENTIAL_BITING -> HAND_NEAR_MOUTH (confidence dropped: {combined_confidence:.3f})")
            # If no fingers are near, go back to idle
            elif len(near_fingers) == 0:
                self.current_state = GestureState.IDLE
                self.last_detection_time = current_time
                logging.debug("State change: POTENTIAL_BITING -> IDLE (no near fingers)")
            # If we've been in potential biting state long enough, transition to biting
            elif (self.last_detection_time is not None and 
                  current_time - self.last_detection_time > timedelta(seconds=self.cooldown_period.total_seconds())):
                self.current_state = GestureState.BITING
                self.last_detection_time = current_time
                logging.info(f"NAIL BITING DETECTED! Confidence: {combined_confidence:.3f}")
        
        elif self.current_state == GestureState.BITING:
            # After detection, go to cooldown state
            self.current_state = GestureState.COOLDOWN
            self.last_detection_time = current_time
            logging.debug("State change: BITING -> COOLDOWN")
        
        return False
    
    def get_roi_for_model(self, frame, hand_landmarks, mouth_bbox):
        """Extract and preprocess ROI for ML model with improved focus on hand-mouth interaction."""
        try:
            # Check if inputs are valid
            if frame is None or hand_landmarks is None or mouth_bbox is None:
                logging.warning("Invalid inputs for ROI extraction")
                return None, None
                
            # Convert landmarks to numpy array
            points = np.array([[lm.x * frame.shape[1], lm.y * frame.shape[0]] 
                             for lm in hand_landmarks.landmark], dtype=np.int32)
            
            # Get bounding box around hand
            x, y, w, h = cv2.boundingRect(points)
            
            # Expand box to include mouth area if close
            mx, my, mw, mh = mouth_bbox
            
            # Validate mouth bbox
            if mx < 0 or my < 0 or mw <= 0 or mh <= 0:
                logging.warning("Invalid mouth bounding box")
                return None, None
                
            # Calculate center points
            hand_center = (x + w//2, y + h//2)
            mouth_center = (mx + mw//2, my + mh//2)
            
            # Calculate distance between hand and mouth centers
            distance = self.calculate_distance(hand_center, mouth_center)
            
            # Calculate dynamic padding based on distance and frame size
            # More padding when hand and mouth are further apart
            frame_diagonal = np.sqrt(frame.shape[0]**2 + frame.shape[1]**2)
            distance_factor = min(1.0, distance / (frame_diagonal * 0.2))  # Normalize by 20% of frame diagonal
            
            # Base padding as percentage of frame size for better scaling
            base_padding_percent = 0.05  # 5% of frame size
            base_padding = int(min(frame.shape[0], frame.shape[1]) * base_padding_percent)
            
            # Dynamic padding increases with distance but caps at reasonable value
            max_padding_percent = 0.15  # 15% of frame size
            max_padding = int(min(frame.shape[0], frame.shape[1]) * max_padding_percent)
            dynamic_padding = int(base_padding + (distance_factor * (max_padding - base_padding)))
            
            # Create a bounding box that encompasses both hand and mouth with intelligent padding
            combined_x = min(x, mx) - dynamic_padding
            combined_y = min(y, my) - dynamic_padding
            combined_w = max(x + w, mx + mw) - combined_x + dynamic_padding
            combined_h = max(y + h, my + mh) - combined_y + dynamic_padding
            
            # Ensure the box stays within the frame
            combined_x = max(0, combined_x)
            combined_y = max(0, combined_y)
            combined_w = min(frame.shape[1] - combined_x, combined_w)
            combined_h = min(frame.shape[0] - combined_y, combined_h)
            
            # Make the ROI square to avoid distortion during resizing
            # Find the center of the ROI
            center_x = combined_x + combined_w // 2
            center_y = combined_y + combined_h // 2
            
            # Make the ROI square based on the larger dimension
            square_size = max(combined_w, combined_h)
            
            # Recalculate the ROI coordinates to be centered and square
            square_x = center_x - square_size // 2
            square_y = center_y - square_size // 2
            
            # Ensure the square ROI stays within the frame
            square_x = max(0, min(square_x, frame.shape[1] - square_size))
            square_y = max(0, min(square_y, frame.shape[0] - square_size))
            
            # If square extends beyond frame, adjust size
            if square_x + square_size > frame.shape[1]:
                square_size = frame.shape[1] - square_x
            if square_y + square_size > frame.shape[0]:
                square_size = frame.shape[0] - square_y
            
            # Ensure we have a valid ROI size
            if square_size <= 0:
                logging.warning("Invalid ROI size calculated")
                return None, None
                
            # Crop region
            roi = frame[square_y:square_y+square_size, square_x:square_x+square_size]
            
            # Verify ROI is not empty
            if roi.size == 0 or roi.shape[0] == 0 or roi.shape[1] == 0:
                logging.warning("Empty ROI extracted")
                return None, None
                
            # Resize and preprocess for model
            # Apply contrast normalization to enhance features
            roi_yuv = cv2.cvtColor(roi, cv2.COLOR_BGR2YUV)
            roi_yuv[:,:,0] = cv2.equalizeHist(roi_yuv[:,:,0])
            roi_enhanced = cv2.cvtColor(roi_yuv, cv2.COLOR_YUV2BGR)
            
            # Resize to model input size
            roi_resized = cv2.resize(roi_enhanced, self.input_size)
            
            # Convert to RGB and normalize
            roi_rgb = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)
            roi_normalized = roi_rgb.astype(np.float32) / 255.0
            
            # Log ROI extraction details for debugging
            logging.debug(f"ROI extracted: hand-mouth distance={distance:.2f}, " 
                         f"padding={dynamic_padding}, size={square_size}x{square_size}")
            
            return roi_normalized, (square_x, square_y, square_size, square_size)  # Return ROI and its coordinates
            
        except Exception as e:
            logging.error(f"Error preparing ROI: {e}")
            return None, None
    
    def process_frame(self, frame):
        # Check if MediaPipe models are available
        if self.hands is None or self.face_mesh is None:
            logging.warning("MediaPipe models not available, returning original frame")
            # Draw a warning message on the frame
            cv2.putText(frame,
                      "MediaPipe initialization failed",
                      (10, 30),
                      cv2.FONT_HERSHEY_SIMPLEX,
                      0.7,
                      (0, 0, 255),
                      2)
            return frame, False
            
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        try:
            # Process hands
            hand_results = self.hands.process(rgb_frame)
            
            # Process face
            face_results = self.face_mesh.process(rgb_frame)
        except Exception as e:
            logging.error(f"Error processing frame with MediaPipe: {e}")
            # Draw error message on the frame
            cv2.putText(frame,
                      f"MediaPipe error: {str(e)[:30]}",
                      (10, 30),
                      cv2.FONT_HERSHEY_SIMPLEX,
                      0.7,
                      (0, 0, 255),
                      2)
            return frame, False
        
        # Draw initial rectangles
        frame_with_detections = frame.copy()
        is_biting = False
        ml_prediction = None
        roi_bbox = None
        
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
        
        # Process hands even if no face detected (for tracking purposes)
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                # Get hand bounding box for visualization
                hand_points = np.array([
                    [hand_landmarks.landmark[i].x * frame.shape[1],
                     hand_landmarks.landmark[i].y * frame.shape[0]]
                    for i in range(21)
                ], dtype=np.int32)
                
                hand_x, hand_y, hand_w, hand_h = cv2.boundingRect(hand_points)
                
                # Initialize finger analysis variables
                near_fingers = []
                pointing_fingers = []
                bent_fingers = []
                
                # Only perform full analysis if face/mouth is detected
                if mouth_center:
                    # Check for nail biting gestures
                    near_fingers = self.is_hand_near_mouth(hand_landmarks, mouth_center, frame.shape[0])
                    
                    # Only analyze finger positions if hand is near mouth
                    if near_fingers:
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
                        
                        # Only use ML model if hand is near mouth
                        if self.model is not None and len(near_fingers) > 0:
                            roi, roi_coords = self.get_roi_for_model(frame, hand_landmarks, mouth_bbox)
                            if roi is not None and roi_coords is not None:
                                try:
                                    # Add batch dimension
                                    roi = np.expand_dims(roi, 0)
                                    # Get model prediction
                                    ml_prediction = float(self.model.predict(roi, verbose=0)[0][0])
                                    roi_bbox = roi_coords
                                    
                                    # Draw ML confidence on frame
                                    confidence_text = f"ML: {ml_prediction:.2f}"
                                    confidence_color = (0, 255, 255) if ml_prediction > self.ml_confidence_threshold else (255, 255, 255)
                                    cv2.putText(frame_with_detections, confidence_text, 
                                              (roi_coords[0], roi_coords[1] - 10), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, confidence_color, 2)
                                    
                                    # Draw ROI box
                                    cv2.rectangle(frame_with_detections, 
                                                (roi_coords[0], roi_coords[1]), 
                                                (roi_coords[0] + roi_coords[2], roi_coords[1] + roi_coords[3]), 
                                                confidence_color, 2)
                                except Exception as e:
                                    logging.error(f"Error during model prediction: {e}")
                                    ml_prediction = None
                                    roi_bbox = None
                
                    # Update detection state only if face/mouth is detected
                    is_biting = self.update_state(near_fingers, pointing_fingers, bent_fingers, ml_prediction)
                
                # Draw hand rectangle with appropriate color
                color = (0, 0, 255) if is_biting else (255, 0, 0)
                if mouth_center is None:
                    color = (255, 255, 0)  # Yellow when no face detected
                
                cv2.rectangle(frame_with_detections,
                            (hand_x, hand_y),
                            (hand_x + hand_w, hand_y + hand_h),
                            color, 2)
                
                # Draw ROI rectangle if available
                if roi_bbox is not None:
                    rx, ry, rw, rh = roi_bbox
                    cv2.rectangle(frame_with_detections,
                                (rx, ry),
                                (rx + rw, ry + rh),
                                (255, 165, 0), 2)  # Orange for ROI, thicker line
                
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
                          0.7,
                          color,
                          2)
                
                # Add threshold info to visualize sensitivity settings
                threshold_text = f"Dist: {self.distance_threshold:.2f}, Angle: {int(self.angle_threshold)}°, ML: {self.ml_confidence_threshold:.2f}"
                cv2.putText(frame_with_detections,
                          threshold_text,
                          (10, 90),
                          cv2.FONT_HERSHEY_SIMPLEX,
                          0.7,
                          (255, 255, 255),
                          2)
        
        return frame_with_detections, is_biting 