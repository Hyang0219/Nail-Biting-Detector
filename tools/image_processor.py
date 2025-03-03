import mediapipe as mp
import cv2
import numpy as np
import logging
import os
from pathlib import Path
import sys
from datetime import datetime
import gc
import time
import signal
from contextlib import contextmanager
import threading
import psutil

# Add the project root to the path for relative imports
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(src_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# Configure logging at the module level
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/nail_biting_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class TimeoutException(Exception):
    """Exception raised when an operation times out."""
    pass

@contextmanager
def timeout(seconds, operation_name="Operation"):
    """Context manager for timing out operations after specified seconds."""
    def signal_handler(signum, frame):
        raise TimeoutException(f"{operation_name} timed out after {seconds} seconds")

    # Register a function to raise a TimeoutException on the signal
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)

    try:
        yield
    finally:
        # Disable the alarm
        signal.alarm(0)

def log_memory_usage():
    """Log current memory usage."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    logger.info(f"Memory usage - RSS: {mem_info.rss / 1024 / 1024:.1f}MB, VMS: {mem_info.vms / 1024 / 1024:.1f}MB")

def force_cleanup():
    """Force cleanup of memory."""
    # Perform a full garbage collection
    gc.collect()

class FaceDetector:
    def __init__(self):
        """Initialize face detector using MediaPipe Face Mesh."""
        try:
            # Set environment variable to limit threads
            os.environ['OMP_NUM_THREADS'] = '1'
            os.environ['MEDIAPIPE_NUM_THREADS'] = '1'
            
            self.face_mesh = None
            self.max_retries = 3
            self.retry_delay = 1  # seconds
        except Exception as e:
            logger.error(f"Failed to initialize FaceDetector: {str(e)}")
            raise
            
    def __enter__(self):
        """Initialize face mesh when entering context."""
        for attempt in range(self.max_retries):
            try:
                self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=1,
                    min_detection_confidence=0.3,
                    refine_landmarks=False,
                    min_tracking_confidence=0.3
                )
                return self
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{self.max_retries} to initialize face_mesh failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    raise
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up resources when exiting context."""
        if self.face_mesh:
            try:
                self.face_mesh.close()
            except Exception as e:
                logger.error(f"Error closing face_mesh: {str(e)}")
            finally:
                self.face_mesh = None
                gc.collect()

    def detect_faces(self, image):
        """Detect faces in the image and return list of face detections."""
        if image is None:
            return []
            
        # Convert to RGB for MediaPipe
        try:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(image_rgb)
            
            if results.multi_face_landmarks:
                return results.multi_face_landmarks
        except Exception as e:
            logger.error(f"Error detecting faces: {str(e)}")
        return []

    def get_mouth_landmarks(self, image):
        """Get mouth landmarks for the first detected face."""
        if image is None:
            return None
            
        try:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(image_rgb)
            
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0].landmark
                # Convert landmarks to pixel coordinates
                h, w = image.shape[:2]
                
                # Get mouth landmarks (indices for the lips region in MediaPipe Face Mesh)
                # This is a more precise way to get lips landmarks
                lips_indices = list(range(61, 69)) + list(range(291, 299))
                mouth_points = [(int(face_landmarks[idx].x * w), int(face_landmarks[idx].y * h)) 
                             for idx in lips_indices]
                return mouth_points
        except Exception as e:
            logger.error(f"Error getting mouth landmarks: {str(e)}")
        return None

class HandDetector:
    def __init__(self):
        """Initialize hand detector using MediaPipe Hands."""
        try:
            os.environ['OMP_NUM_THREADS'] = '1'
            os.environ['MEDIAPIPE_NUM_THREADS'] = '1'
            
            self.hands = None
            self.max_retries = 3
            self.retry_delay = 1  # seconds
        except Exception as e:
            logger.error(f"Failed to initialize HandDetector: {str(e)}")
            raise
            
    def __enter__(self):
        """Initialize hands when entering context."""
        for attempt in range(self.max_retries):
            try:
                self.hands = mp.solutions.hands.Hands(
                    static_image_mode=True,
                    max_num_hands=2,
                    min_detection_confidence=0.3,
                    min_tracking_confidence=0.3,
                    model_complexity=0
                )
                return self
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{self.max_retries} to initialize hands failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    raise
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up resources when exiting context."""
        if self.hands:
            try:
                self.hands.close()
            except Exception as e:
                logger.error(f"Error closing hands: {str(e)}")
            finally:
                self.hands = None
                gc.collect()

    def detect_hands(self, image):
        """Detect hands in the image and return list of hand landmarks."""
        if image is None:
            return []
            
        try:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.hands.process(image_rgb)
            
            if results.multi_hand_landmarks:
                h, w = image.shape[:2]
                hands_landmarks = []
                for hand_landmarks in results.multi_hand_landmarks:
                    # Convert landmarks to pixel coordinates
                    points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]
                    hands_landmarks.append(points)
                return hands_landmarks
        except Exception as e:
            logger.error(f"Error detecting hands: {str(e)}")
        return []

class ImageValidator:
    def __init__(self, category):
        """Initialize validator for specific image category."""
        self.category = category
        self.validation_timeout = 15  # reduced timeout for efficiency
        
    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points."""
        if point1 is None or point2 is None:
            return float('inf')
        
        try:
            # Convert points to numpy arrays if they aren't already
            point1 = np.array(point1)
            
            # If point2 is a list of points, find the minimum distance to any point
            if isinstance(point2, list):
                if not point2:  # Empty list
                    return float('inf')
                min_dist = float('inf')
                for p2 in point2:
                    if p2 is not None:
                        p2 = np.array(p2)
                        if not np.any(np.isnan(p2)) and not np.any(np.isinf(p2)):
                            dist = np.linalg.norm(point1 - p2)
                            min_dist = min(min_dist, dist)
                return min_dist
            # If point2 is a tuple or array, treat it as a single point
            else:
                point2 = np.array(point2)
                if not np.any(np.isnan(point2)) and not np.any(np.isinf(point2)):
                    return np.linalg.norm(point1 - point2)
        except Exception as e:
            logger.error(f"Error calculating distance: {str(e)}")
        return float('inf')

    def calculate_finger_angle(self, tip, mid, base):
        """Calculate angle between finger segments, handling errors properly."""
        try:
            if None in (tip, mid, base):
                return None
                
            v1 = np.array(tip) - np.array(mid)
            v2 = np.array(base) - np.array(mid)
            
            # Skip if vectors are zero length
            if np.all(v1 == 0) or np.all(v2 == 0):
                return None
                
            # Calculate dot product and norms
            dot_product = np.dot(v1, v2)
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)
            
            # Avoid division by zero
            if norm_v1 == 0 or norm_v2 == 0:
                return None
                
            # Get angle in degrees
            cos_angle = np.clip(dot_product / (norm_v1 * norm_v2), -1.0, 1.0)
            angle = np.degrees(np.arccos(cos_angle))
            
            return angle
        except Exception as e:
            logger.debug(f"Error calculating finger angle: {str(e)}")
            return None

    def validate_image(self, image, face_detector, hand_detector):
        """Validate image based on category-specific criteria with more permissive rules."""
        logger.info("\nValidating image...")
        
        try:
            with timeout(self.validation_timeout, "Image validation"):
                # Resize image if too large (max dimension 1280px)
                max_dim = 1280
                height, width = image.shape[:2]
                if max(height, width) > max_dim:
                    scale = max_dim / max(height, width)
                    image = cv2.resize(image, (int(width * scale), int(height * scale)))
                    height, width = image.shape[:2]
                    logger.debug(f"Resized image to {width}x{height}")
                
                # Detect faces and hands
                faces = face_detector.detect_faces(image)
                hands = hand_detector.detect_hands(image)
                
                has_face = len(faces) > 0
                has_hands = len(hands) > 0
                
                # More permissive validation:
                # For nail_biting: Require both face and hands
                # For non_nail_biting: Require at least hands
                if self.category == "nail_biting":
                    if not has_face:
                        logger.info(f"Rejecting nail_biting image: No face detected")
                        return False
                    
                    if not has_hands:
                        logger.info(f"Rejecting nail_biting image: No hands detected")
                        return False
                else:  # non_nail_biting
                    # At minimum, we need hands for non-nail-biting
                    if not has_hands:
                        logger.info(f"Rejecting non_nail_biting image: No hands detected")
                        return False
                
                # Get mouth landmarks if face is detected
                mouth_landmarks = None
                if has_face:
                    mouth_landmarks = face_detector.get_mouth_landmarks(image)
                
                # For nail-biting, we need mouth landmarks
                if self.category == "nail_biting" and (mouth_landmarks is None or len(mouth_landmarks) == 0):
                    logger.info(f"Rejecting nail_biting image: No mouth landmarks detected")
                    return False
                
                # Calculate hand-mouth distance
                hand_near_mouth = False
                min_distance = float('inf')
                
                if has_hands and mouth_landmarks:
                    # Calculate center of mouth
                    mouth_center = np.mean(mouth_landmarks, axis=0).astype(int)
                    
                    # Make sure mouth center is valid
                    if not np.any(np.isnan(mouth_center)) and not np.any(np.isinf(mouth_center)):
                        # Calculate minimum distance from any fingertip to mouth
                        for hand in hands:
                            # Get fingertips (indices 4,8,12,16,20)
                            fingertips = [hand[i] for i in [4, 8, 12, 16, 20] if i < len(hand)]
                            
                            for tip in fingertips:
                                if tip is not None:
                                    dist = self.calculate_distance(tip, mouth_center)
                                    min_distance = min(min_distance, dist)
                        
                        # Define hand near mouth as within 40% of image height
                        hand_near_mouth = min_distance < height * 0.4
                
                # For nail-biting, hands must be near mouth
                if self.category == "nail_biting" and not hand_near_mouth:
                    logger.info(f"Rejecting nail_biting image: Hand not near mouth (distance: {min_distance/height*100:.1f}% of height)")
                    return False
                
                # Get finger angles for detecting bent fingers
                bent_fingers = []
                for hand in hands:
                    # Check each finger
                    for i in range(5):  # 5 fingers
                        # Make sure we have enough landmarks
                        if 4 + i*4 >= len(hand) or 3 + i*4 >= len(hand) or 2 + i*4 >= len(hand):
                            continue
                            
                        tip = hand[4 + i*4]  # Fingertip
                        mid = hand[3 + i*4]  # Middle joint
                        base = hand[2 + i*4]  # Base joint
                        
                        angle = self.calculate_finger_angle(tip, mid, base)
                        if angle is not None:
                            # For both categories, consider finger bent if angle < 160°
                            # This threshold can be adjusted based on specific requirements
                            if angle < 160:
                                bent_fingers.append((i, angle, tip))
                
                has_bent_fingers = len(bent_fingers) > 0
                
                # Log validation summary
                logger.info("Validation summary:")
                logger.info(f"- Face detected: {has_face}")
                logger.info(f"- Hands detected: {has_hands} (count: {len(hands)})")
                logger.info(f"- Hand near mouth: {hand_near_mouth}")
                logger.info(f"- Bent fingers detected: {has_bent_fingers} (count: {len(bent_fingers)})")
                if mouth_landmarks:
                    logger.info(f"- Min hand-mouth distance: {min_distance:.2f}px ({min_distance/height*100:.1f}% of height)")
                
                # Apply category-specific validation rules:
                if self.category == "nail_biting":
                    # For nail-biting: require hands near mouth and bent fingers
                    # We've already validated face, hands, and hand-mouth proximity
                    valid = has_bent_fingers
                    logger.info(f"Nail-biting validation result: {valid}")
                    return valid
                else:  # non_nail_biting
                    # More permissive non-nail-biting validation:
                    # 1. If hands near mouth, require NO bent fingers
                    # 2. If hands not near mouth, it's valid regardless of bent fingers
                    if hand_near_mouth:
                        valid = not has_bent_fingers
                    else:
                        valid = True
                    
                    logger.info(f"Non-nail-biting validation result: {valid}")
                    return valid
                
        except TimeoutException as e:
            logger.error(f"Validation timed out: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Error during validation: {str(e)}")
            return False

def process_image_batch(batch, category, stats):
    """Process a batch of images."""
    raw_dir = Path('data/raw') / category
    processed_dir = Path('data/processed') / category
    
    with FaceDetector() as face_detector, HandDetector() as hand_detector:
        # Create validator
        validator = ImageValidator(category)
        
        for image_path in batch:
            try:
                input_path = image_path
                output_path = processed_dir / image_path.name
                
                # Skip if already processed
                if output_path.exists():
                    logger.info(f"Skipping already processed image: {output_path}")
                    stats['skipped'] += 1
                    continue
                
                logger.info(f"\nProcessing image: {input_path}")
                
                # Read image with error handling
                try:
                    image = cv2.imread(str(input_path))
                    if image is None:
                        logger.error(f"Failed to read image: {input_path}")
                        stats['errors'] += 1
                        continue
                except Exception as e:
                    logger.error(f"Error reading image {input_path}: {str(e)}")
                    stats['errors'] += 1
                    continue
                
                # Validate the image
                if validator.validate_image(image, face_detector, hand_detector):
                    # Save the processed image
                    cv2.imwrite(str(output_path), image)
                    stats['processed'] += 1
                    logger.info(f"Successfully processed and saved: {output_path}")
                else:
                    stats['failed_validation'] += 1
                
                # Clean up memory
                del image
                gc.collect()
                
            except TimeoutException as e:
                logger.error(f"Timeout processing {image_path}: {str(e)}")
                stats['timeouts'] += 1
            except Exception as e:
                logger.error(f"Error processing {input_path}: {str(e)}")
                stats['errors'] += 1
    
    # Force cleanup after batch
    gc.collect()
    return stats

def main():
    """Process images from raw to processed directories."""
    logger.info("Starting image processing...")
    
    # Process images in smaller batches to manage memory
    BATCH_SIZE = 5  # Increased from 3 to 5 due to more efficient processing
    
    # Initialize statistics
    total_stats = {
        'processed': 0,
        'failed_validation': 0,
        'skipped': 0,
        'errors': 0,
        'timeouts': 0
    }
    
    # Create base directories
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    try:
        # Process each category
        categories = ['nail_biting', 'non_nail_biting']
        for category in categories:
            # Reset statistics for each category
            category_stats = {
                'processed': 0,
                'failed_validation': 0,
                'skipped': 0,
                'errors': 0,
                'timeouts': 0
            }
            
            logger.info(f"\nProcessing {category} images...")
            
            # Create proper directory paths
            raw_dir = Path('data/raw') / category
            processed_dir = Path('data/processed') / category
            
            # Create category directories if they don't exist
            os.makedirs(raw_dir, exist_ok=True)
            os.makedirs(processed_dir, exist_ok=True)
            
            # Get list of files to process
            image_files = list(raw_dir.glob('*.jpg')) + list(raw_dir.glob('*.jpeg')) + list(raw_dir.glob('*.png'))
            total_images = len(image_files)
            
            logger.info(f"Found {total_images} images in {category}")
            
            # Process in batches
            for i in range(0, total_images, BATCH_SIZE):
                batch = image_files[i:i + BATCH_SIZE]
                batch_num = i//BATCH_SIZE + 1
                total_batches = (total_images + BATCH_SIZE - 1)//BATCH_SIZE
                logger.info(f"\nProcessing batch {batch_num}/{total_batches} for {category}")
                logger.info(f"Processing images {i+1} to {min(i+BATCH_SIZE, total_images)} of {total_images}")
                
                # Log memory usage before batch
                log_memory_usage()
                
                # Process batch
                process_image_batch(batch, category, category_stats)
                
                # Give system time to reclaim memory
                time.sleep(0.5)
                
                # Log memory usage after batch
                log_memory_usage()
                
                # Log batch statistics
                logger.info(f"\nBatch {batch_num}/{total_batches} Statistics for {category}:")
                for key, value in category_stats.items():
                    logger.info(f"{key}: {value}")
                
            # Log category completion
            logger.info(f"\nCompleted processing {category} images")
            logger.info("Category Statistics:")
            for key, value in category_stats.items():
                logger.info(f"{key}: {value}")
                total_stats[key] += value
            
    except KeyboardInterrupt:
        logger.info("\nProcessing interrupted by user. Saving final statistics...")
    except Exception as e:
        logger.error(f"Critical error in main processing loop: {str(e)}")
    
    finally:
        # Log final statistics
        logger.info("\nFinal Processing Statistics:")
        for key, value in total_stats.items():
            logger.info(f"{key}: {value}")
        
        # Final cleanup
        force_cleanup()

if __name__ == "__main__":
    main() 