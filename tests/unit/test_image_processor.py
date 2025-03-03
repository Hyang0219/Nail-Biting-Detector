#!/usr/bin/env python3
import pytest
import numpy as np
import cv2
import os
from pathlib import Path
import sys
import logging
import time
from tqdm import tqdm
from tools.image_processor import FaceDetector, HandDetector, ImageValidator

# Add the root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@pytest.fixture
def sample_image():
    """Create a sample test image."""
    # Create a 300x300 RGB image
    image = np.zeros((300, 300, 3), dtype=np.uint8)
    # Add some features to simulate face and hands
    cv2.circle(image, (150, 100), 30, (200, 200, 200), -1)  # Face
    cv2.circle(image, (150, 150), 10, (150, 150, 150), -1)  # Mouth
    cv2.rectangle(image, (130, 180), (170, 220), (180, 180, 180), -1)  # Hand
    return image

@pytest.fixture
def validator_nail_biting():
    """Create an ImageValidator instance for nail-biting."""
    return ImageValidator(category="nail_biting")

@pytest.fixture
def validator_non_nail_biting():
    """Create an ImageValidator instance for non-nail-biting."""
    return ImageValidator(category="non_nail_biting")

class TestImageValidator:
    def test_initialization(self, validator_nail_biting, validator_non_nail_biting):
        """Test if ImageValidator initializes correctly."""
        assert validator_nail_biting.category == "nail_biting"
        assert validator_non_nail_biting.category == "non_nail_biting"
        assert validator_nail_biting.validation_timeout == 15
        assert validator_non_nail_biting.validation_timeout == 15

    def test_calculate_distance(self, validator_nail_biting):
        """Test distance calculation between points."""
        point1 = (0, 0)
        point2 = (3, 4)
        distance = validator_nail_biting.calculate_distance(point1, point2)
        assert distance == 5.0  # Pythagorean theorem: 3-4-5 triangle

        # Test with None points
        assert validator_nail_biting.calculate_distance(None, point2) == float('inf')
        assert validator_nail_biting.calculate_distance(point1, None) == float('inf')
        
    def test_calculate_finger_angle(self, validator_nail_biting):
        """Test angle calculation between finger segments."""
        tip = (0, 0)
        mid = (1, 1)
        base = (2, 0)
        
        # This should form a right angle (90 degrees)
        angle = validator_nail_biting.calculate_finger_angle(tip, mid, base)
        assert 89.0 < angle < 91.0  # Allow for floating point precision
        
        # Test with None values
        assert validator_nail_biting.calculate_finger_angle(None, mid, base) is None
        assert validator_nail_biting.calculate_finger_angle(tip, None, base) is None
        assert validator_nail_biting.calculate_finger_angle(tip, mid, None) is None

class TestFaceDetector:
    @pytest.fixture
    def face_detector(self):
        with FaceDetector() as detector:
            yield detector

    def test_detect_faces(self, face_detector, sample_image):
        """Test face detection."""
        faces = face_detector.detect_faces(sample_image)
        assert isinstance(faces, list)
        
    def test_get_mouth_landmarks(self, face_detector, sample_image):
        """Test mouth landmark detection."""
        landmarks = face_detector.get_mouth_landmarks(sample_image)
        # May return None if no face is detected in the sample image
        if landmarks:
            assert isinstance(landmarks, list)

class TestHandDetector:
    @pytest.fixture
    def hand_detector(self):
        with HandDetector() as detector:
            yield detector

    def test_detect_hands(self, hand_detector, sample_image):
        """Test hand detection."""
        hands = hand_detector.detect_hands(sample_image)
        assert isinstance(hands, list)

# This is a helper function, not a pytest test
def _test_classification(input_path, category, expected_result, debug=False):
    """Helper function to test classification of a single image.
    
    This is NOT a pytest test - it's called by run_real_image_tests()
    """
    try:
        # Load image
        image = cv2.imread(str(input_path))
        if image is None:
            logger.error(f"Failed to load image: {input_path}")
            return False
            
        # Initialize detectors and validator
        with FaceDetector() as face_detector, HandDetector() as hand_detector:
            validator = ImageValidator(category)
            
            # Time the validation
            start_time = time.time()
            result = validator.validate_image(image, face_detector, hand_detector)
            elapsed_time = time.time() - start_time
            
            # Log results
            status = "PASS" if result == expected_result else "FAIL"
            logger.info(f"{status}: {input_path.name} - Expected: {expected_result}, Got: {result}, Time: {elapsed_time:.2f}s")
            
            # Debug visualization if requested
            if debug and result != expected_result:
                # Save debug image with classification result
                debug_dir = Path('logs/debug')
                os.makedirs(debug_dir, exist_ok=True)
                debug_path = debug_dir / f"{status}_{category}_{input_path.name}"
                cv2.imwrite(str(debug_path), image)
                logger.info(f"Saved debug image to {debug_path}")
                
            return result == expected_result
            
    except Exception as e:
        logger.error(f"Error testing {input_path}: {str(e)}")
        return False

def run_real_image_tests():
    """Run tests on real images from the dataset."""
    logger.info("Starting real image validation tests...")
    
    # Set directories
    raw_dir = Path('data/raw')
    processed_dir = Path('data/processed')
    
    # Verify directories exist
    if not raw_dir.exists() or not processed_dir.exists():
        logger.error("Error: Required data directories do not exist")
        return 0, 0
        
    # Test categories
    categories = ['nail_biting', 'non_nail_biting']
    
    # Initialize test stats
    total_tests = 0
    passed_tests = 0
    
    # Test each category
    for category in categories:
        logger.info(f"\nTesting {category} classification...")
        
        # Get samples for validation
        raw_samples = list((raw_dir / category).glob('*.jpg'))[:5]  # Test 5 raw images
        processed_samples = list((processed_dir / category).glob('*.jpg'))[:5]  # Test 5 processed images
        
        logger.info(f"Testing {len(raw_samples)} raw samples")
        # Test raw samples (should fail validation)
        for sample in tqdm(raw_samples, desc=f"Raw {category} samples"):
            # For raw samples not in processed, we expect them to fail validation
            if not (processed_dir / category / sample.name).exists():
                total_tests += 1
                # We expect these to fail validation (hence False expected result)
                if _test_classification(sample, category, False, debug=True):
                    passed_tests += 1
        
        logger.info(f"Testing {len(processed_samples)} processed samples")
        # Test processed samples (should pass validation)
        for sample in tqdm(processed_samples, desc=f"Processed {category} samples"):
            total_tests += 1
            # We expect these to pass validation (hence True expected result)
            if _test_classification(sample, category, True, debug=True):
                passed_tests += 1
    
    # Print final results
    logger.info("\nTest Results:")
    logger.info(f"Total tests: {total_tests}")
    logger.info(f"Passed tests: {passed_tests}")
    logger.info(f"Failed tests: {total_tests - passed_tests}")
    
    if total_tests > 0:
        success_rate = (passed_tests / total_tests) * 100
        logger.info(f"Success rate: {success_rate:.2f}%")
        return passed_tests, total_tests
    else:
        logger.info("No tests were run")
        return 0, 0

if __name__ == "__main__":
    run_real_image_tests() 