import cv2
import mediapipe as mp
import numpy as np
from enum import Enum
from datetime import datetime, timedelta
import math
import tensorflow as tf
import os
import logging
import random

class GestureState(Enum):
    IDLE = 0
    HAND_NEAR_MOUTH = 1
    POTENTIAL_BITING = 2
    BITING = 3
    COOLDOWN = 4

class GestureDetector:
    def __init__(self, model_path=None, sensitivity=0.5):
        """Initialize the gesture detector with optional ML model."""
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
        
        # Initialize detection parameters
        self.sensitivity = sensitivity
        self.update_sensitivity(sensitivity)
        
        # Initialize state tracking
        self.current_state = GestureState.IDLE
        self.consecutive_detections = 0
        self.detection_threshold = 3  # Number of consecutive detections to trigger an alert
        self.cooldown_period = timedelta(seconds=5)
        self.last_detection_time = None
        self.hand_near_mouth_start_time = None
        self.min_finger_mouth_distance = float('inf')
        
        # Initialize model
        self.model = None
        self.model_path = model_path
        self.model_used = "None"  # Track which detection method was used
        self.load_model()
        
        # Initialize sticker display
        self.preferred_sticker = "apt.webp"  # Preferred sticker to display
        self.load_stickers()
        self.sticker_start_time = None
        self.sticker_display_duration = 4.0  # Increased to 4 seconds
        self.current_sticker_index = 0
        self.current_frame_index = 0
        self.last_sticker_update = None
        self.sticker_frame_duration = 0.05  # seconds between frames for animated stickers (original WebP speed)
        
        # Initialize distance threshold and model_used
        self.distance_threshold = 0.08  # Default value, will be updated dynamically
        
        # Detection parameters
        self.consecutive_frames_threshold = 2
        self.min_finger_mouth_distance = float('inf')
        
        # Log initialization
        if self.model:
            logging.info("GestureDetector initialized with ML model")
        else:
            logging.info("GestureDetector initialized without ML model")
        
        self.input_size = (224, 224)  # Model input size
        
        # For ROI visualization
        self.current_roi_coords = None
        self.show_roi_debug = True
    
    def load_stickers(self):
        """Load stickers/GIFs for display when nail biting is detected."""
        self.stickers = []
        stickers_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'assets', 'stickers')
        
        if not os.path.exists(stickers_dir):
            logging.warning(f"Stickers directory not found at {stickers_dir}")
            return
            
        # Look for image files in the stickers directory
        sticker_files = [f for f in os.listdir(stickers_dir) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp'))]
        
        if not sticker_files:
            logging.warning(f"No sticker files found in {stickers_dir}")
            return
            
        logging.info(f"Found {len(sticker_files)} sticker files")
        
        # Import PIL for image handling
        try:
            from PIL import Image
            import numpy as np
            has_pil = True
        except ImportError:
            logging.warning("PIL not available, some image formats may not load correctly")
            has_pil = False
        
        # Check if imageio is available for GIF handling
        try:
            import imageio
            has_imageio = True
        except ImportError:
            logging.warning("imageio not available, will use PIL for GIF handling if possible")
            has_imageio = False
        
        # Load each sticker
        preferred_sticker_index = -1
        for i, sticker_file in enumerate(sticker_files):
            sticker_path = os.path.join(stickers_dir, sticker_file)
            
            # Check if this is our preferred sticker
            if sticker_file.lower() == self.preferred_sticker.lower():
                preferred_sticker_index = i
                logging.info(f"Found preferred sticker: {sticker_file}")
            
            # Check if file is an animated GIF or WebP
            is_animated = False
            if has_pil:
                try:
                    with Image.open(sticker_path) as img:
                        # Check if image is animated (has multiple frames)
                        is_animated = hasattr(img, 'n_frames') and img.n_frames > 1
                        if is_animated:
                            logging.info(f"Detected animated file: {sticker_file} with {img.n_frames} frames")
                except Exception as e:
                    logging.error(f"Error checking animation status: {e}")
            
            if sticker_file.lower().endswith(('.gif', '.webp')) and is_animated:
                # For animated files (GIF or WebP), we need to extract all frames
                try:
                    if has_imageio and sticker_file.lower().endswith('.gif'):
                        # Try with imageio first for GIFs
                        import imageio
                        gif = imageio.mimread(sticker_path)
                        # Convert from RGB to BGR for OpenCV
                        frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in gif]
                    elif has_pil:
                        # Use PIL for animated WebP and as fallback for GIF
                        img = Image.open(sticker_path)
                        frames = []
                        try:
                            for frame_idx in range(0, img.n_frames):
                                img.seek(frame_idx)
                                # Convert PIL image to numpy array
                                frame = np.array(img.convert('RGBA'))
                                # Convert from RGBA to BGRA (OpenCV uses BGR)
                                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGRA)
                                frames.append(frame)
                        except Exception as e:
                            logging.error(f"Error extracting frames from animated file: {e}")
                    else:
                        # Skip if neither imageio nor PIL is available
                        logging.warning(f"Skipping animated file {sticker_file} - no suitable library available")
                        continue
                        
                    self.stickers.append({
                        'type': 'animated',
                        'frames': frames,
                        'path': sticker_path
                    })
                    logging.info(f"Loaded animated sticker with {len(frames)} frames: {sticker_file}")
                except Exception as e:
                    logging.error(f"Error loading animated file {sticker_file}: {e}")
            else:
                # For static images
                try:
                    # Try loading with OpenCV first
                    sticker = cv2.imread(sticker_path, cv2.IMREAD_UNCHANGED)
                    
                    # If loading failed or no alpha channel for webp, try with PIL
                    if (sticker is None or 
                        (sticker_file.lower().endswith('.webp') and 
                         (sticker.shape[2] < 4 if sticker is not None and len(sticker.shape) > 2 else True))):
                        
                        if has_pil:
                            # Open the image with PIL which has better WebP support
                            pil_img = Image.open(sticker_path)
                            
                            # Convert to RGBA if it doesn't have an alpha channel
                            if pil_img.mode != 'RGBA':
                                pil_img = pil_img.convert('RGBA')
                                
                            # Convert PIL image to numpy array
                            sticker = np.array(pil_img)
                            
                            # Convert from RGBA to BGRA (OpenCV uses BGR)
                            if sticker.shape[2] == 4:  # If it has an alpha channel
                                sticker = cv2.cvtColor(sticker, cv2.COLOR_RGBA2BGRA)
                            else:
                                sticker = cv2.cvtColor(sticker, cv2.COLOR_RGB2BGR)
                        else:
                            # If PIL is not available and OpenCV failed, skip this sticker
                            if sticker is None:
                                logging.warning(f"Skipping {sticker_file} - failed to load with OpenCV and PIL not available")
                                continue
                    
                    self.stickers.append({
                        'type': 'static',
                        'image': sticker,
                        'path': sticker_path
                    })
                    logging.info(f"Loaded static sticker: {sticker_file} with shape {sticker.shape}")
                except Exception as e:
                    logging.error(f"Error loading sticker {sticker_file}: {e}")
        
        # Set the preferred sticker as the default if found
        if preferred_sticker_index >= 0:
            logging.info(f"Using preferred sticker: {self.preferred_sticker}")
            self.current_sticker_index = preferred_sticker_index
        
        if not self.stickers:
            logging.warning("No stickers were successfully loaded")
        else:
            logging.info(f"Successfully loaded {len(self.stickers)} stickers")
    
    def update_sensitivity(self, sensitivity):
        """Update detection thresholds based on sensitivity value (0.0 to 1.0)."""
        # Interpolate thresholds based on sensitivity
        # Higher sensitivity = larger thresholds = easier to trigger
        self.base_distance_multiplier = 0.8 + (sensitivity * 0.7)  # 0.8 to 1.5 multiplier (was 0.5 to 1.0)
        self.angle_threshold = 45 + (sensitivity * 45)  # 45 to 90 degrees
        self.cluster_threshold = 0.05 + (sensitivity * 0.10)  # 0.05 to 0.15
        self.ml_confidence_threshold = max(0.3, 0.8 - (sensitivity * 0.5))  # 0.8 to 0.3
        
        # Update duration based on sensitivity (lower sensitivity = longer duration)
        duration_seconds = max(0.8, 2.0 - (sensitivity * 1.0))  # 2.0 to 0.8 seconds
        self.hand_near_duration = timedelta(seconds=duration_seconds)
    
    def toggle_roi_debug(self, enabled=None):
        """Toggle the ROI debug visualization.
        
        Args:
            enabled: If None, toggles the current state. Otherwise, sets to the specified boolean value.
        
        Returns:
            bool: The new state of the ROI debug visualization
        """
        if enabled is None:
            self.show_roi_debug = not self.show_roi_debug
        else:
            self.show_roi_debug = bool(enabled)
        
        logging.debug(f"ROI debug visualization {'enabled' if self.show_roi_debug else 'disabled'}")
        return self.show_roi_debug
    
    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points."""
        return math.sqrt(((point1[0] - point2[0]) ** 2) + ((point1[1] - point2[1]) ** 2))
    
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
    
    def calculate_box_center_distance(self, hand_bbox, mouth_bbox, frame_height):
        """Calculate the normalized distance between the centers of two bounding boxes.
        Also calculate a dynamic threshold based on the sizes of the boxes."""
        # Extract box dimensions
        hand_x, hand_y, hand_w, hand_h = hand_bbox
        mouth_x, mouth_y, mouth_w, mouth_h = mouth_bbox
        
        # Calculate centers
        hand_center = (hand_x + hand_w//2, hand_y + hand_h//2)
        mouth_center = (mouth_x + mouth_w//2, mouth_y + mouth_h//2)
        
        # Calculate Euclidean distance
        pixel_distance = math.sqrt((hand_center[0] - mouth_center[0])**2 + 
                                  (hand_center[1] - mouth_center[1])**2)
        
        # Normalize by frame height
        normalized_distance = pixel_distance / frame_height
        
        # Calculate dynamic threshold based on box sizes
        # Average the normalized dimensions of hand and mouth
        hand_size = (hand_w + hand_h) / (2 * frame_height)
        mouth_size = (mouth_w + mouth_h) / (2 * frame_height)
        avg_size = (hand_size + mouth_size) / 2
        
        # Set threshold proportional to the average size, adjusted by sensitivity
        dynamic_threshold = avg_size * self.base_distance_multiplier
        
        # Ensure threshold is within reasonable bounds (0.08 to 0.20)
        dynamic_threshold = max(0.08, min(0.20, dynamic_threshold))
        
        return normalized_distance, hand_center, mouth_center, dynamic_threshold
    
    def is_hand_near_mouth(self, hand_landmarks, mouth_center, frame_height):
        """Check if any fingertip is near the mouth and track the closest one."""
        near_fingers = []
        
        # We'll still calculate fingertip distances for gesture detection
        for i, tip_idx in enumerate(self.FINGERTIPS):
            tip_x = int(hand_landmarks.landmark[tip_idx].x * frame_height * 4/3)
            tip_y = int(hand_landmarks.landmark[tip_idx].y * frame_height)
            tip_point = (tip_x, tip_y)
            
            # Calculate distance
            pixel_distance = self.calculate_distance(tip_point, mouth_center)
            normalized_distance = pixel_distance / frame_height
            
            if normalized_distance < self.distance_threshold:
                near_fingers.append(tip_point)
        
        return near_fingers
    
    def is_finger_pointing_to_mouth(self, tip, mid, mouth_center):
        """Check if finger is oriented towards mouth."""
        angle = self.calculate_angle(tip, mid, mouth_center)
        return angle < self.angle_threshold
    
    def overlay_sticker(self, frame):
        """Overlay a sticker on the frame if nail biting was detected."""
        if not self.stickers:
            return frame
            
        now = datetime.now()
        
        # Check if we should display a sticker
        if self.sticker_start_time is not None:
            elapsed = (now - self.sticker_start_time).total_seconds()
            
            # If we're still within the display duration
            if elapsed < self.sticker_display_duration:
                sticker_data = self.stickers[self.current_sticker_index]
                
                # Get frame dimensions
                frame_height, frame_width = frame.shape[:2]
                
                # For animated stickers, we need to select the current frame
                if sticker_data['type'] == 'animated':
                    frames = sticker_data['frames']
                    
                    if not frames:
                        logging.warning("Animated sticker has no frames")
                        return frame
                    
                    # Update the frame index based on time
                    if self.last_sticker_update is None or (now - self.last_sticker_update).total_seconds() >= self.sticker_frame_duration:
                        self.current_frame_index = int(elapsed / self.sticker_frame_duration) % len(frames)
                        self.last_sticker_update = now
                        logging.debug(f"Displaying frame {self.current_frame_index+1}/{len(frames)} of animated sticker")
                        
                    sticker = frames[self.current_frame_index]
                else:
                    sticker = sticker_data['image']
                
                # Resize sticker to a reasonable size (1/3 of the frame width)
                target_width = frame_width // 3
                sticker_height, sticker_width = sticker.shape[:2]
                scale = target_width / sticker_width
                sticker = cv2.resize(sticker, (0, 0), fx=scale, fy=scale)
                
                # Get new dimensions after resize
                sticker_height, sticker_width = sticker.shape[:2]
                
                # Position the sticker in the center of the screen
                x_offset = (frame_width - sticker_width) // 2
                y_offset = (frame_height - sticker_height) // 2
                
                # Add a semi-transparent background behind the sticker for better visibility
                overlay = frame.copy()
                bg_padding = 20
                cv2.rectangle(overlay,
                             (x_offset - bg_padding, y_offset - bg_padding),
                             (x_offset + sticker_width + bg_padding, y_offset + sticker_height + bg_padding),
                             (40, 40, 40), -1)
                cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
                
                # Check if sticker has alpha channel
                if sticker.shape[2] == 4:
                    # Extract RGB and alpha channels
                    sticker_rgb = sticker[:, :, :3]
                    sticker_alpha = sticker[:, :, 3] / 255.0
                    
                    # Create alpha masks
                    alpha = np.stack([sticker_alpha] * 3, axis=2)
                    inv_alpha = 1.0 - alpha
                    
                    # Define the region of interest (ROI)
                    roi = frame[y_offset:y_offset+sticker_height, x_offset:x_offset+sticker_width]
                    
                    # Blend the sticker with the ROI
                    blended = (sticker_rgb * alpha + roi * inv_alpha).astype(np.uint8)
                    
                    # Place the blended image back into the frame
                    frame[y_offset:y_offset+sticker_height, x_offset:x_offset+sticker_width] = blended
                else:
                    # If no alpha channel, just overlay the sticker
                    frame[y_offset:y_offset+sticker_height, x_offset:x_offset+sticker_width] = sticker
            else:
                # Reset sticker display
                self.sticker_start_time = None
                self.current_frame_index = 0
                self.last_sticker_update = None
                
        return frame
    
    def update_state(self, near_fingers, pointing_fingers, bent_fingers, ml_prediction=None):
        """Update the gesture state based on current detection and history."""
        now = datetime.now()
        self.model_used = "None"  # Reset model used for this detection
        
        # Handle cooldown period
        if self.current_state == GestureState.COOLDOWN:
            if self.last_detection_time and now - self.last_detection_time > self.cooldown_period:
                self.current_state = GestureState.IDLE
                self.consecutive_detections = 0
                self.hand_near_mouth_start_time = None
            return False
        
        # Reset to IDLE if hand is too far from mouth (add a buffer to prevent oscillation)
        if self.min_finger_mouth_distance > (self.distance_threshold * 1.5) and self.current_state != GestureState.IDLE:
            self.current_state = GestureState.IDLE
            self.consecutive_detections = 0
            self.hand_near_mouth_start_time = None
            return False
        
        # State machine logic with ML integration and duration check
        if len(near_fingers) > 0 or self.min_finger_mouth_distance < self.distance_threshold:
            # Start tracking time when hand first comes near mouth
            if self.current_state == GestureState.IDLE:
                self.current_state = GestureState.HAND_NEAR_MOUTH
            
            # Check if hand has been near mouth for the required duration
            hand_near_long_enough = False
            if self.hand_near_mouth_start_time:
                hand_near_long_enough = (now - self.hand_near_mouth_start_time) >= self.hand_near_duration
            
            # Only proceed with detection if hand has been near mouth long enough
            if hand_near_long_enough:
                if len(pointing_fingers) > 0 and len(bent_fingers) > 0:
                    self.consecutive_detections += 1
                    if self.consecutive_detections >= self.consecutive_frames_threshold:
                        # If ML model is available, use it to confirm the detection
                        if ml_prediction is not None and ml_prediction > self.ml_confidence_threshold:
                            if self.current_state in [GestureState.POTENTIAL_BITING, GestureState.HAND_NEAR_MOUTH]:
                                self.model_used = "ML"  # ML model confirmed the detection
                                self.current_state = GestureState.BITING
                                self.last_detection_time = now
                                self.current_state = GestureState.COOLDOWN
                                self.consecutive_detections = 0
                                self.hand_near_mouth_start_time = None
                                
                                # Start displaying a sticker
                                if hasattr(self, 'stickers') and self.stickers:
                                    self.sticker_start_time = now
                                    # Use preferred sticker if available, otherwise random
                                    preferred_index = -1
                                    for i, sticker in enumerate(self.stickers):
                                        if os.path.basename(sticker['path']).lower() == self.preferred_sticker.lower():
                                            preferred_index = i
                                            break
                                    
                                    if preferred_index >= 0:
                                        self.current_sticker_index = preferred_index
                                    else:
                                        # Fallback to random selection
                                        self.current_sticker_index = random.randint(0, len(self.stickers) - 1)
                                        
                                    if self.stickers[self.current_sticker_index]['type'] == 'animated':
                                        self.current_frame_index = 0
                                        self.last_sticker_update = now
                                
                                return True
                        elif ml_prediction is None:  # Fall back to geometric approach if no ML model
                            if self.current_state in [GestureState.POTENTIAL_BITING, GestureState.HAND_NEAR_MOUTH]:
                                self.model_used = "MediaPipe"  # Using MediaPipe for detection
                                self.current_state = GestureState.BITING
                                self.last_detection_time = now
                                self.current_state = GestureState.COOLDOWN
                                self.consecutive_detections = 0
                                self.hand_near_mouth_start_time = None
                                
                                # Start displaying a sticker
                                if hasattr(self, 'stickers') and self.stickers:
                                    self.sticker_start_time = now
                                    # Use preferred sticker if available, otherwise random
                                    preferred_index = -1
                                    for i, sticker in enumerate(self.stickers):
                                        if os.path.basename(sticker['path']).lower() == self.preferred_sticker.lower():
                                            preferred_index = i
                                            break
                                    
                                    if preferred_index >= 0:
                                        self.current_sticker_index = preferred_index
                                    else:
                                        # Fallback to random selection
                                        self.current_sticker_index = random.randint(0, len(self.stickers) - 1)
                                        
                                    if self.stickers[self.current_sticker_index]['type'] == 'animated':
                                        self.current_frame_index = 0
                                        self.last_sticker_update = now
                                
                                return True
                    self.current_state = GestureState.POTENTIAL_BITING
                else:
                    self.current_state = GestureState.HAND_NEAR_MOUTH
                    self.consecutive_detections = max(0, self.consecutive_detections - 1)
            else:
                # Hand is near mouth but not long enough yet
                self.current_state = GestureState.HAND_NEAR_MOUTH
        else:
            self.current_state = GestureState.IDLE
            self.consecutive_detections = 0
            self.hand_near_mouth_start_time = None
        
        return False
    
    def get_roi_for_model(self, frame, hand_landmarks, mouth_bbox):
        """Extract and preprocess ROI for ML model."""
        try:
            # Extract fingertip landmarks only (we're most interested in these for nail biting)
            fingertip_points = np.array([
                [hand_landmarks.landmark[idx].x * frame.shape[1], 
                 hand_landmarks.landmark[idx].y * frame.shape[0]] 
                for idx in self.FINGERTIPS
            ], dtype=np.int32)
            
            # Get bounding box around fingertips
            x, y, w, h = cv2.boundingRect(fingertip_points)
            
            # Get mouth dimensions
            mx, my, mw, mh = mouth_bbox
            
            # Create a more focused ROI that prioritizes the interaction area between fingertips and mouth
            # Calculate the center points
            fingertip_center_x = x + w//2
            fingertip_center_y = y + h//2
            mouth_center_x = mx + mw//2
            mouth_center_y = my + mh//2
            
            # Calculate center of interaction area
            interaction_center_x = (fingertip_center_x + mouth_center_x) // 2
            interaction_center_y = (fingertip_center_y + mouth_center_y) // 2
            
            # Calculate ROI dimensions (make it large enough to capture the interaction, but focused)
            # Use the distance between fingertips and mouth to determine size
            distance = math.sqrt((fingertip_center_x - mouth_center_x)**2 + 
                               (fingertip_center_y - mouth_center_y)**2)
            
            # Size the ROI relative to the distance, but with a minimum size
            roi_size = max(int(distance * 1.5), 150)
            
            # Calculate ROI coordinates
            roi_x = max(0, interaction_center_x - roi_size//2)
            roi_y = max(0, interaction_center_y - roi_size//2)
            roi_w = min(frame.shape[1] - roi_x, roi_size)
            roi_h = min(frame.shape[0] - roi_y, roi_size)
            
            # Store ROI coordinates for visualization
            self.current_roi_coords = (roi_x, roi_y, roi_w, roi_h)
            
            # Crop region
            roi = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
            
            # Resize and preprocess for model
            if roi.size > 0:
                roi = cv2.resize(roi, self.input_size)
                roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                roi = roi.astype(np.float32) / 255.0
                return roi
            
            return None
        except Exception as e:
            logging.error(f"Error preparing ROI: {e}")
            self.current_roi_coords = None
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
        if hand_results.multi_hand_landmarks and mouth_center and mouth_bbox:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                # Get hand bounding box for visualization
                hand_points = np.array([
                    [hand_landmarks.landmark[i].x * frame.shape[1],
                     hand_landmarks.landmark[i].y * frame.shape[0]]
                    for i in range(21)
                ], dtype=np.int32)
                
                hand_x, hand_y, hand_w, hand_h = cv2.boundingRect(hand_points)
                hand_bbox = (hand_x, hand_y, hand_w, hand_h)
                
                # Calculate distance between hand and mouth centers with dynamic threshold
                self.min_finger_mouth_distance, hand_center, _, self.distance_threshold = self.calculate_box_center_distance(
                    hand_bbox, mouth_bbox, frame.shape[0])
                
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
                
                # Track time when hand is near mouth
                now = datetime.now()
                if self.min_finger_mouth_distance < self.distance_threshold:
                    # Start tracking time when hand first comes near mouth
                    if self.hand_near_mouth_start_time is None:
                        self.hand_near_mouth_start_time = now
                        logging.debug("Hand near mouth - starting timer")
                else:
                    # Reset timer if hand moves away
                    if self.hand_near_mouth_start_time is not None:
                        logging.debug("Hand moved away - resetting timer")
                    self.hand_near_mouth_start_time = None
                
                # Check if hand has been near mouth for the required duration
                hand_near_long_enough = False
                if self.hand_near_mouth_start_time:
                    elapsed = now - self.hand_near_mouth_start_time
                    hand_near_long_enough = elapsed >= self.hand_near_duration
                    if hand_near_long_enough and self.model is not None:
                        logging.debug(f"Hand near mouth for {elapsed.total_seconds():.1f}s - triggering ML model")
                
                # Use ML model if available and hand has been near mouth for the required duration
                if self.model is not None and hand_near_long_enough:
                    roi = self.get_roi_for_model(frame, hand_landmarks, mouth_bbox)
                    if roi is not None:
                        # Add batch dimension
                        roi = np.expand_dims(roi, 0)
                        # Get model prediction
                        ml_prediction = float(self.model.predict(roi, verbose=0)[0])
                        logging.debug(f"ML prediction: {ml_prediction:.3f} (threshold: {self.ml_confidence_threshold:.3f})")
                
                # Update detection state
                is_biting = self.update_state(near_fingers, pointing_fingers, bent_fingers, ml_prediction)
                
                # Draw hand rectangle with appropriate color
                color = (0, 0, 255) if is_biting else (255, 0, 0)
                cv2.rectangle(frame_with_detections,
                            (hand_x, hand_y),
                            (hand_x + hand_w, hand_y + hand_h),
                            color, 2)
                
                # Draw line between hand center and mouth center
                line_color = (0, 255, 0) if self.min_finger_mouth_distance < self.distance_threshold else (0, 0, 255)
                cv2.line(frame_with_detections, mouth_center, hand_center, line_color, 2)
                
                # Draw ROI box for debugging if available
                if self.show_roi_debug and self.current_roi_coords is not None:
                    roi_x, roi_y, roi_w, roi_h = self.current_roi_coords
                    # Draw ROI as semi-transparent gray box
                    overlay = frame_with_detections.copy()
                    cv2.rectangle(overlay, 
                                (roi_x, roi_y), 
                                (roi_x + roi_w, roi_y + roi_h), 
                                (128, 128, 128), -1)
                    # Add text label
                    cv2.putText(overlay, "ML ROI", 
                              (roi_x + 5, roi_y + 25),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                              (255, 255, 255), 2)
                    # Blend with transparency
                    cv2.addWeighted(overlay, 0.4, frame_with_detections, 0.6, 0, frame_with_detections)
                    # Add border
                    cv2.rectangle(frame_with_detections, 
                                (roi_x, roi_y), 
                                (roi_x + roi_w, roi_y + roi_h), 
                                (128, 128, 128), 2)
                
                # Calculate positions for status panel in the middle-left
                frame_height, frame_width = frame.shape[:2]
                panel_width = 350
                panel_height = 180
                panel_x = 20
                panel_y = (frame_height - panel_height) // 2
                
                # Create a semi-transparent rounded panel for status information
                overlay = frame_with_detections.copy()
                
                # Draw a filled rectangle with rounded corners
                cv2.rectangle(overlay, 
                             (panel_x, panel_y), 
                             (panel_x + panel_width, panel_y + panel_height), 
                             (40, 40, 40), -1)
                
                # Add a border
                cv2.rectangle(overlay, 
                             (panel_x, panel_y), 
                             (panel_x + panel_width, panel_y + panel_height), 
                             (100, 100, 100), 2)
                
                # Blend the overlay with the original frame
                cv2.addWeighted(overlay, 0.7, frame_with_detections, 0.3, 0, frame_with_detections)
                
                # Add a title bar
                cv2.rectangle(frame_with_detections,
                             (panel_x, panel_y),
                             (panel_x + panel_width, panel_y + 30),
                             (60, 60, 60), -1)
                
                # Add title text
                cv2.putText(frame_with_detections,
                          "Detection Status",
                          (panel_x + 10, panel_y + 22),
                          cv2.FONT_HERSHEY_SIMPLEX,
                          0.7,
                          (255, 255, 255),
                          2)
                
                # Draw status information in the panel
                text_x = panel_x + 15
                text_y_start = panel_y + 55
                line_height = 30
                
                # State with colored indicator
                state_color = (0, 255, 0) if self.current_state == GestureState.IDLE else \
                             (255, 255, 0) if self.current_state == GestureState.HAND_NEAR_MOUTH else \
                             (255, 165, 0) if self.current_state == GestureState.POTENTIAL_BITING else \
                             (255, 0, 0) if self.current_state == GestureState.COOLDOWN else \
                             (255, 0, 0)
                
                # Draw colored circle indicator
                cv2.circle(frame_with_detections, 
                          (text_x, text_y_start - 5), 
                          8, 
                          state_color, 
                          -1)
                
                # State text
                cv2.putText(frame_with_detections,
                          f"State: {self.current_state.name}",
                          (text_x + 20, text_y_start),
                          cv2.FONT_HERSHEY_SIMPLEX,
                          0.6,
                          (255, 255, 255),
                          1)
                
                # Distance with threshold
                distance_color = (0, 255, 0) if self.min_finger_mouth_distance < self.distance_threshold else (255, 100, 100)
                cv2.putText(frame_with_detections,
                          f"Distance: {self.min_finger_mouth_distance:.3f} / {self.distance_threshold:.3f}",
                          (text_x, text_y_start + line_height),
                          cv2.FONT_HERSHEY_SIMPLEX,
                          0.6,
                          distance_color,
                          1)
                
                # Duration info
                if self.current_state == GestureState.COOLDOWN and self.last_detection_time:
                    elapsed = now - self.last_detection_time
                    remaining = max(0, (self.cooldown_period - elapsed).total_seconds())
                    duration_text = f"Cooldown: {remaining:.1f}s"
                    duration_color = (255, 100, 100)
                elif self.hand_near_mouth_start_time:
                    elapsed = now - self.hand_near_mouth_start_time
                    duration_text = f"Hand near: {elapsed.total_seconds():.1f}s / {self.hand_near_duration.total_seconds():.1f}s"
                    duration_color = (255, 255, 0)
                else:
                    duration_text = "Waiting for hand near mouth"
                    duration_color = (200, 200, 200)
                
                cv2.putText(frame_with_detections,
                          duration_text,
                          (text_x, text_y_start + line_height * 2),
                          cv2.FONT_HERSHEY_SIMPLEX,
                          0.6,
                          duration_color,
                          1)
                
                # Model info
                model_text = f"Model: {self.model_used}" if self.model_used != "None" else "Model: Not triggered"
                model_color = (0, 255, 0) if self.model_used == "ML" else \
                             (100, 100, 255) if self.model_used == "MediaPipe" else \
                             (200, 200, 200)
                
                cv2.putText(frame_with_detections,
                          model_text,
                          (text_x, text_y_start + line_height * 3),
                          cv2.FONT_HERSHEY_SIMPLEX,
                          0.6,
                          model_color,
                          1)
                
                # Finger metrics
                metrics_text = f"Fingers: {len(near_fingers)} near, {len(pointing_fingers)} pointing, {len(bent_fingers)} bent"
                cv2.putText(frame_with_detections,
                          metrics_text,
                          (text_x, text_y_start + line_height * 4),
                          cv2.FONT_HERSHEY_SIMPLEX,
                          0.6,
                          (200, 200, 200),
                          1)
        
        # Overlay sticker if needed
        frame_with_detections = self.overlay_sticker(frame_with_detections)
        
        return frame_with_detections, is_biting
    
    def process_frame_with_model_info(self, frame):
        """Process frame and return detection results along with which model was used."""
        # Reset model used
        self.model_used = "None"
        
        # Process the frame using the existing method
        frame_with_detections, is_biting = self.process_frame(frame)
        
        # Add model info to the frame
        model_text = f"Model: {self.model_used}"
        cv2.putText(frame_with_detections,
                  model_text,
                  (frame.shape[1] - 200, 30),  # Position in top right
                  cv2.FONT_HERSHEY_SIMPLEX,
                  0.7,
                  (255, 255, 0),  # Yellow text
                  2)
        
        return frame_with_detections, is_biting, self.model_used
    
    def load_model(self):
        """Load the ML model for classification if available."""
        if self.model_path and os.path.exists(self.model_path):
            try:
                self.model = tf.keras.models.load_model(self.model_path)
                logging.info(f"Loaded model from {self.model_path}")
            except Exception as e:
                logging.error(f"Error loading model: {e}")
        elif self.model_path:
            # If specific path was provided but doesn't exist
            logging.warning(f"Model path {self.model_path} not found")
            
            # Try to find the latest model in the models directory
            models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
            if os.path.exists(models_dir):
                model_files = [f for f in os.listdir(models_dir) if f.endswith('.keras') or f.endswith('.h5')]
                if model_files:
                    # Sort by modification time (newest first)
                    model_files.sort(key=lambda x: os.path.getmtime(os.path.join(models_dir, x)), reverse=True)
                    latest_model = os.path.join(models_dir, model_files[0])
                    try:
                        self.model = tf.keras.models.load_model(latest_model)
                        logging.info(f"Loaded latest model from {latest_model}")
                    except Exception as e:
                        logging.error(f"Error loading latest model: {e}") 