import pytest
import numpy as np
import mediapipe as mp
from detection.gesture_detector import GestureDetector, GestureState
from datetime import datetime, timedelta

class TestGestureDetector:
    @pytest.fixture
    def detector(self):
        return GestureDetector()
    
    @pytest.fixture
    def sample_frame(self):
        """Create a sample frame for testing."""
        return np.zeros((480, 640, 3), dtype=np.uint8)
    
    @pytest.fixture
    def mock_hand_landmarks(self):
        class MockLandmark:
            def __init__(self, x, y):
                self.x = x
                self.y = y
        
        class MockHandLandmarks:
            def __init__(self):
                self.landmark = [MockLandmark(0.5, 0.5) for _ in range(21)]
        
        return MockHandLandmarks()
    
    def test_initialization(self, detector):
        """Test if GestureDetector initializes correctly."""
        assert detector.hands is not None
        assert detector.face_mesh is not None
        assert len(detector.MOUTH_LANDMARKS) > 0
        assert detector.current_state == GestureState.IDLE
        assert detector.last_detection_time is None
        assert isinstance(detector.cooldown_period, timedelta)
    
    def test_geometric_calculations(self, detector):
        """Test geometric calculation methods."""
        # Test distance calculation
        point1 = (0, 0)
        point2 = (3, 4)
        assert detector.calculate_distance(point1, point2) == 5.0
        
        # Test angle calculation for bent finger
        tip = (1, 0)    # Fingertip
        mid = (0, 0)    # Middle joint
        base = (-1, 0)  # Base joint
        angle = detector.calculate_angle(tip, mid, base)
        assert 170 < angle < 190  # Should be approximately 180 degrees (straight)
    
    def test_finger_bent_detection(self, detector):
        """Test finger bent detection."""
        # Test straight finger (angle = 180 degrees)
        tip = (1, 0)    # Fingertip
        mid = (0, 0)    # Middle joint
        base = (-1, 0)  # Base joint
        assert not detector.is_finger_bent(tip, mid, base)  # Should be False for straight finger
        
        # Test bent finger (angle < 160 degrees)
        tip = (0, 1)    # Fingertip
        mid = (0, 0)    # Middle joint
        base = (1, 0)   # Base joint
        assert detector.is_finger_bent(tip, mid, base)  # Should be True for bent finger
    
    def test_fingertip_clustering(self, detector):
        """Test fingertip clustering detection."""
        # Test clustered fingertips
        fingertips = [(100, 100), (102, 102), (101, 101)]
        frame_height = 480
        assert detector.are_fingertips_clustered(fingertips, frame_height)
        
        # Test spread fingertips
        fingertips = [(100, 100), (200, 200), (300, 300)]
        assert not detector.are_fingertips_clustered(fingertips, frame_height)
        
        # Test single fingertip
        fingertips = [(100, 100)]
        assert not detector.are_fingertips_clustered(fingertips, frame_height)
    
    def test_hand_near_mouth(self, detector, mock_hand_landmarks):
        """Test hand near mouth detection."""
        mouth_center = (320, 240)  # Center of frame
        frame_height = 480
        
        # Test hand near mouth
        near_fingers = detector.is_hand_near_mouth(mock_hand_landmarks, mouth_center, frame_height)
        assert isinstance(near_fingers, list)
        
        # All fingers should be detected as near (due to mock landmarks all being at 0.5, 0.5)
        assert len(near_fingers) == len(detector.FINGERTIPS)
    
    def test_process_frame_no_detection(self, detector, sample_frame):
        """Test processing frame with no hands or face."""
        frame_with_detections, is_biting = detector.process_frame(sample_frame)
        
        # Check output types and values
        assert isinstance(frame_with_detections, np.ndarray)
        assert isinstance(is_biting, bool)
        assert not is_biting  # Should be False when no detection
        
        # Check frame dimensions haven't changed
        assert frame_with_detections.shape == sample_frame.shape
        
        # Check state
        assert detector.current_state == GestureState.IDLE
    
    def test_state_machine_transitions(self, detector):
        """Test state machine transitions."""
        # Start in IDLE state
        assert detector.current_state == GestureState.IDLE
        
        # Test transition to HAND_NEAR_MOUTH
        near_fingers = [(100, 100)]
        pointing_fingers = []
        bent_fingers = []
        detector.update_state(near_fingers, pointing_fingers, bent_fingers)
        assert detector.current_state == GestureState.HAND_NEAR_MOUTH
        
        # Test transition to POTENTIAL_BITING
        pointing_fingers = [(100, 100)]
        bent_fingers = [(100, 100)]
        detector.update_state(near_fingers, pointing_fingers, bent_fingers)
        assert detector.current_state == GestureState.POTENTIAL_BITING
        
        # Test transition to BITING (after consecutive detections)
        for _ in range(detector.consecutive_frames_threshold):
            is_biting = detector.update_state(near_fingers, pointing_fingers, bent_fingers)
            if is_biting:
                break
        
        assert detector.current_state == GestureState.COOLDOWN
        
        # Test cooldown period
        is_biting = detector.update_state(near_fingers, pointing_fingers, bent_fingers)
        assert not is_biting  # Should not detect during cooldown
        assert detector.current_state == GestureState.COOLDOWN
        
        # Test cooldown expiration
        detector.last_detection_time = datetime.now() - timedelta(seconds=4)
        detector.update_state([], [], [])
        assert detector.current_state == GestureState.IDLE 