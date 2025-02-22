import pytest
import numpy as np
import cv2
from detection.ml_utils import GestureModel

class TestGestureModel:
    @pytest.fixture
    def model(self):
        return GestureModel()
    
    @pytest.fixture
    def sample_frame(self):
        # Create a sample frame (300x300 RGB)
        return np.zeros((300, 300, 3), dtype=np.uint8)
    
    @pytest.fixture
    def mock_hand_landmarks(self):
        class MockLandmark:
            def __init__(self, x, y):
                self.x = x
                self.y = y
        
        class MockHandLandmarks:
            def __init__(self):
                self.landmark = [MockLandmark(0.4, 0.4) for _ in range(21)]
        
        return MockHandLandmarks()
    
    def test_initialization(self, model):
        """Test if model initializes correctly."""
        assert model.input_size == (224, 224)
        assert model.window_size == 5
        assert len(model.temporal_window) == 0
    
    def test_preprocess_frame(self, model, sample_frame, mock_hand_landmarks):
        """Test frame preprocessing."""
        mouth_bbox = (100, 100, 50, 30)  # x, y, width, height
        
        # Test with valid inputs
        processed = model.preprocess_frame(sample_frame, mock_hand_landmarks, mouth_bbox)
        assert processed is not None
        assert processed.shape == (*model.input_size, 3)
        assert processed.dtype == np.float32
        assert 0 <= processed.min() <= processed.max() <= 1.0
        
        # Test with no hand landmarks
        processed = model.preprocess_frame(sample_frame, None, mouth_bbox)
        assert processed is None
        
        # Test with no mouth bbox
        processed = model.preprocess_frame(sample_frame, mock_hand_landmarks, None)
        assert processed is None
    
    def test_predict(self, model, sample_frame, mock_hand_landmarks):
        """Test prediction functionality."""
        mouth_bbox = (100, 100, 50, 30)
        
        # Test prediction
        confidence, is_biting = model.predict(sample_frame, mock_hand_landmarks, mouth_bbox)
        assert isinstance(confidence, float)
        assert isinstance(is_biting, bool)
        assert 0 <= confidence <= 1.0
        
        # Test temporal smoothing
        for _ in range(10):
            confidence, is_biting = model.predict(sample_frame, mock_hand_landmarks, mouth_bbox)
        assert len(model.temporal_window) == model.window_size
        
        # Test with invalid inputs
        confidence, is_biting = model.predict(sample_frame, None, None)
        assert confidence == 0.0
        assert not is_biting
    
    def test_temporal_smoothing(self, model):
        """Test temporal smoothing of confidence scores."""
        # Simulate a sequence of confidence scores
        test_scores = [0.1, 0.2, 0.8, 0.9, 0.85]
        for score in test_scores:
            model.temporal_window.append(score)
            if len(model.temporal_window) > model.window_size:
                model.temporal_window.pop(0)
        
        # Check window size
        assert len(model.temporal_window) == min(len(test_scores), model.window_size)
        
        # Check smoothed value
        smoothed = np.mean(model.temporal_window)
        assert smoothed == pytest.approx(np.mean(test_scores[-model.window_size:])) 