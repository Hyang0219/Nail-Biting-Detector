import cv2
import sys
import os
import numpy as np
import logging
from detection.gesture_detector import GestureDetector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    logging.info("Starting test script")
    
    # Initialize the detector
    try:
        detector = GestureDetector(sensitivity=0.7)
        logging.info("Gesture detector initialized successfully!")
    except Exception as e:
        logging.error(f"Failed to initialize detector: {e}")
        return
    
    # Test with a blank image
    logging.info("Attempting to load placeholder image")
    blank_image = cv2.imread('resources/placeholder.jpg') if os.path.exists('resources/placeholder.jpg') else None
    
    if blank_image is None:
        # Create a blank image if placeholder doesn't exist
        logging.info("Trying absolute path for placeholder")
        blank_image = cv2.imread('/workspaces/nail-biting-detection/resources/placeholder.jpg') if os.path.exists('/workspaces/nail-biting-detection/resources/placeholder.jpg') else None
    
    if blank_image is None:
        # Create a blank image if we still don't have one
        logging.info("Creating blank test image")
        blank_image = np.ones((480, 640, 3), dtype=np.uint8) * 255
    
    # Try to process the blank image
    try:
        logging.info("Processing blank image")
        result, is_biting = detector.process_frame(blank_image)
        logging.info(f"Frame processed successfully! Biting detected: {is_biting}")
    except Exception as e:
        logging.error(f"Error processing frame: {e}")
        import traceback
        logging.error(traceback.format_exc())
    
    logging.info("Detector test completed.")

if __name__ == "__main__":
    main()
    logging.info("Script execution finished.") 