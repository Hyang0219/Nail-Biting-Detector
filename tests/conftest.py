import pytest
import cv2
import numpy as np
import os
import sys

# Add src directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

@pytest.fixture
def sample_frame():
    """Create a sample frame for testing."""
    # Create a black frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    return frame

@pytest.fixture
def mock_camera(monkeypatch):
    """Mock camera capture for testing."""
    class MockCamera:
        def __init__(self):
            self.frame = np.zeros((480, 640, 3), dtype=np.uint8)
            self.is_opened = True

        def read(self):
            return True, self.frame

        def release(self):
            self.is_opened = False

        def isOpened(self):
            return self.is_opened

    def mock_videocapture(*args, **kwargs):
        return MockCamera()

    monkeypatch.setattr(cv2, 'VideoCapture', mock_videocapture)
    return MockCamera() 