import pytest
import numpy as np
import mediapipe as mp
from detection.gesture_detector import GestureDetector, GestureState
from datetime import datetime, timedelta

class TestGestureDetector:
    @pytest.fixture
    def detector(self):
        return GestureDetector()
    
    def test_initialization(self, detector):
        """Test if GestureDetector initializes correctly."""
        assert detector.hands is not None
        assert detector.face_mesh is not None
        assert len(detector.MOUTH_LANDMARKS) > 0
        assert detector.current_state == GestureState.IDLE
        assert detector.last_detection_time is None
        assert isinstance(detector.cooldown_period, timedelta)
        assert detector.model is not None
    
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
        
        # Test transition to POTENTIAL_BITING
        detector.update_state(True, 0.9)
        assert detector.current_state == GestureState.POTENTIAL_BITING
        
        # Test transition to BITING
        is_alert = detector.update_state(True, 0.9)
        assert is_alert
        assert detector.current_state == GestureState.COOLDOWN
        
        # Test cooldown period
        is_alert = detector.update_state(True, 0.9)
        assert not is_alert  # Should not alert during cooldown
        assert detector.current_state == GestureState.COOLDOWN
        
        # Test cooldown expiration
        detector.last_detection_time = datetime.now() - timedelta(seconds=4)
        detector.update_state(False, 0.0)
        assert detector.current_state == GestureState.IDLE
    
    def test_process_frame_with_mock_detections(self, detector, sample_frame, monkeypatch):
        """Test processing frame with mocked hand and face detections."""
        # Mock MediaPipe detection results
        class MockHandResults:
            def __init__(self):
                self.multi_hand_landmarks = [MockHandLandmarks()]
        
        class MockFaceResults:
            def __init__(self):
                self.multi_face_landmarks = [MockFaceLandmarks()]
        
        class MockLandmark:
            def __init__(self, x, y):
                self.x = x
                self.y = y
        
        class MockHandLandmarks:
            def __init__(self):
                self.landmark = []
                # Set up landmarks
                for i in range(21):
                    self.landmark.append(MockLandmark(0.5, 0.5))
        
        class MockFaceLandmarks:
            def __init__(self):
                self.landmark = []
                for i in range(468):
                    if i in detector.MOUTH_LANDMARKS:
                        self.landmark.append(MockLandmark(0.5, 0.5))
                    else:
                        self.landmark.append(MockLandmark(0.4, 0.4))
        
        def mock_process_hands(*args, **kwargs):
            return MockHandResults()
        
        def mock_process_face(*args, **kwargs):
            return MockFaceResults()
        
        # Mock ML model predict method
        def mock_predict(*args, **kwargs):
            return 0.9, True
        
        # Apply mocks
        monkeypatch.setattr(detector.hands, 'process', mock_process_hands)
        monkeypatch.setattr(detector.face_mesh, 'process', mock_process_face)
        monkeypatch.setattr(detector.model, 'predict', mock_predict)
        
        # Test first frame - should go to POTENTIAL_BITING
        frame_with_detections, is_biting = detector.process_frame(sample_frame)
        assert not is_biting  # First detection should not trigger alert
        assert detector.current_state == GestureState.POTENTIAL_BITING
        
        # Test second frame - should detect biting
        frame_with_detections, is_biting = detector.process_frame(sample_frame)
        assert is_biting
        assert detector.current_state == GestureState.COOLDOWN 