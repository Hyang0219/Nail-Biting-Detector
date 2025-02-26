import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
from PIL import Image
import logging

class GestureModel:
    def __init__(self):
        # Load MobileNet V2 model from TF Hub
        self.model = hub.load('https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/5')
        self.input_size = (224, 224)  # MobileNet V2 input size
        self.temporal_window = []
        self.window_size = 5  # Number of frames to consider for temporal smoothing
        
    def preprocess_frame(self, frame, hand_landmarks, mouth_bbox):
        """
        Process frame to create input for the model.
        Args:
            frame: Original frame
            hand_landmarks: MediaPipe hand landmarks
            mouth_bbox: Bounding box of mouth (x, y, w, h)
        Returns:
            Processed image tensor
        """
        try:
            # Get hand region
            if hand_landmarks and mouth_bbox:
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
                
                # Resize to model input size
                if roi.size > 0:
                    roi = cv2.resize(roi, self.input_size)
                    # Convert to RGB
                    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                    # Convert to float and normalize
                    roi = roi.astype(np.float32) / 255.0
                    return roi
            
            return None
        except Exception as e:
            logging.error(f"Error preprocessing frame: {e}")
            return None
    
    def predict(self, frame, hand_landmarks, mouth_bbox):
        """
        Make prediction on processed frame.
        Returns:
            confidence: Float between 0 and 1
            is_biting: Boolean indicating if nail biting is detected
        """
        try:
            # Preprocess frame
            processed = self.preprocess_frame(frame, hand_landmarks, mouth_bbox)
            if processed is None:
                return 0.0, False
            
            # Add batch dimension
            input_tensor = tf.expand_dims(processed, 0)
            
            # Get model prediction
            predictions = self.model(input_tensor)
            # Get confidence for relevant class (we'll use "biting" class from ImageNet)
            # ImageNet class 927 is 'nail' - we'll use it as a proxy
            confidence = float(predictions[0, 927])
            
            # Add to temporal window
            self.temporal_window.append(confidence)
            if len(self.temporal_window) > self.window_size:
                self.temporal_window.pop(0)
            
            # Get smoothed confidence
            smoothed_confidence = np.mean(self.temporal_window)
            
            # Threshold for detection
            is_biting = bool(smoothed_confidence > 0.6)
            
            return smoothed_confidence, is_biting
        
        except Exception as e:
            logging.error(f"Error making prediction: {e}")
            return 0.0, False 