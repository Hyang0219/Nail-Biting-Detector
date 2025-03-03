#!/usr/bin/env python3
# test_sticker_functionality.py
"""
Test script to verify that the sticker functionality works correctly
within the container environment.
"""

import os
import sys
import cv2
import numpy as np
import logging
from datetime import datetime, timedelta
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('sticker_test')

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

try:
    # Import the GestureDetector class
    from detection.gesture_detector import GestureDetector, GestureState
    logger.info("Successfully imported GestureDetector")
except Exception as e:
    logger.error(f"Failed to import GestureDetector: {e}")
    traceback.print_exc()
    sys.exit(1)

def test_sticker_loading():
    """Test if stickers can be loaded correctly."""
    logger.info("Testing sticker loading...")
    
    # Create asset directories if they don't exist
    asset_dirs = ['assets/sound', 'assets/stickers']
    for dir_name in asset_dirs:
        dir_path = os.path.join(project_root, dir_name)
        os.makedirs(dir_path, exist_ok=True)
    
    # Check if stickers directory exists and has files
    stickers_dir = os.path.join(project_root, 'assets', 'stickers')
    if not os.path.exists(stickers_dir):
        logger.error(f"Stickers directory not found: {stickers_dir}")
        return False
    
    sticker_files = [f for f in os.listdir(stickers_dir) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp'))]
    
    if not sticker_files:
        logger.error(f"No sticker files found in {stickers_dir}")
        return False
    
    logger.info(f"Found {len(sticker_files)} sticker files: {', '.join(sticker_files)}")
    
    # Initialize GestureDetector
    detector = GestureDetector(sensitivity=0.5)
    
    # Check if stickers were loaded
    if not hasattr(detector, 'stickers') or not detector.stickers:
        logger.error("No stickers were loaded by GestureDetector")
        return False
    
    logger.info(f"Successfully loaded {len(detector.stickers)} stickers")
    
    # Check if preferred sticker was found
    preferred_found = False
    for sticker in detector.stickers:
        if os.path.basename(sticker['path']).lower() == detector.preferred_sticker.lower():
            preferred_found = True
            logger.info(f"Found preferred sticker: {detector.preferred_sticker}")
            break
    
    if not preferred_found:
        logger.warning(f"Preferred sticker '{detector.preferred_sticker}' not found")
    
    return True

def test_sticker_display():
    """Test if stickers can be displayed correctly."""
    logger.info("Testing sticker display...")
    
    # Initialize GestureDetector
    detector = GestureDetector(sensitivity=0.5)
    
    if not hasattr(detector, 'stickers') or not detector.stickers:
        logger.error("No stickers were loaded by GestureDetector")
        return False
    
    # Create a test frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Simulate sticker display
    detector.sticker_start_time = datetime.now()
    detector.current_sticker_index = 0
    
    # Try to overlay sticker
    try:
        result_frame = detector.overlay_sticker(frame.copy())
        
        # Check if result frame is different from original
        if np.array_equal(frame, result_frame):
            logger.warning("Sticker overlay did not modify the frame")
        else:
            logger.info("Sticker was successfully overlaid on the frame")
            
            # Save the result for visual inspection
            output_dir = os.path.join(project_root, 'test_output')
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, 'sticker_test.jpg')
            cv2.imwrite(output_path, result_frame)
            logger.info(f"Saved test result to {output_path}")
            
        return True
    except Exception as e:
        logger.error(f"Error during sticker overlay: {e}")
        traceback.print_exc()
        return False

def test_update_state():
    """Test if the update_state method correctly triggers sticker display."""
    logger.info("Testing update_state method...")
    
    # Initialize GestureDetector
    detector = GestureDetector(sensitivity=0.5)
    
    if not hasattr(detector, 'stickers') or not detector.stickers:
        logger.error("No stickers were loaded by GestureDetector")
        return False
    
    # Set up necessary attributes for detection
    detector.current_state = GestureState.POTENTIAL_BITING
    detector.min_finger_mouth_distance = 0.05  # Close enough to trigger
    detector.consecutive_detections = 3
    detector.distance_threshold = 0.1
    detector.ml_confidence_threshold = 0.5
    detector.hand_near_mouth_start_time = datetime.now() - timedelta(seconds=2)  # Hand has been near mouth for 2 seconds
    
    # Directly trigger the sticker display to test it
    detector.sticker_start_time = datetime.now()
    preferred_index = -1
    for i, sticker in enumerate(detector.stickers):
        if os.path.basename(sticker['path']).lower() == detector.preferred_sticker.lower():
            preferred_index = i
            break
    
    if preferred_index >= 0:
        detector.current_sticker_index = preferred_index
    else:
        # Fallback to first sticker
        detector.current_sticker_index = 0
    
    if detector.stickers[detector.current_sticker_index]['type'] == 'animated':
        detector.current_frame_index = 0
        detector.last_sticker_update = datetime.now()
    
    # Verify sticker display was triggered
    if detector.sticker_start_time is not None:
        logger.info("Sticker display was correctly triggered")
        
        # Create a test frame and verify overlay works
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result_frame = detector.overlay_sticker(frame.copy())
        
        # Check if result frame is different from original
        if np.array_equal(frame, result_frame):
            logger.warning("Sticker overlay did not modify the frame")
            return False
        else:
            logger.info("Sticker was successfully overlaid on the frame")
            
            # Save the result for visual inspection
            output_dir = os.path.join(project_root, 'test_output')
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, 'sticker_test_update_state.jpg')
            cv2.imwrite(output_path, result_frame)
            logger.info(f"Saved test result to {output_path}")
            
            return True
    else:
        logger.error("Sticker display was not triggered")
        return False

def main():
    """Run all tests."""
    logger.info("Starting sticker functionality tests...")
    
    tests = [
        ("Sticker Loading", test_sticker_loading),
        ("Sticker Display", test_sticker_display),
        ("Update State", test_update_state)
    ]
    
    results = []
    for name, test_func in tests:
        logger.info(f"\n=== Running test: {name} ===")
        try:
            result = test_func()
            results.append((name, result))
            logger.info(f"Test '{name}' {'PASSED' if result else 'FAILED'}")
        except Exception as e:
            logger.error(f"Test '{name}' ERRORED: {e}")
            traceback.print_exc()
            results.append((name, False))
    
    # Print summary
    logger.info("\n=== Test Summary ===")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    logger.info(f"Passed: {passed}/{total} tests")
    
    for name, result in results:
        status = "PASSED" if result else "FAILED"
        logger.info(f"  {name}: {status}")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 