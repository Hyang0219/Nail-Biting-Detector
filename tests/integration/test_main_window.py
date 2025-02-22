import pytest
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
import sys
import os
from gui.main_window import MainWindow

@pytest.fixture
def app():
    """Create a Qt application."""
    return QApplication(sys.argv)

@pytest.fixture
def main_window(app, mock_camera):
    """Create the main window with a mock camera."""
    window = MainWindow()
    return window

class TestMainWindow:
    def test_window_initialization(self, main_window):
        """Test if the window initializes with correct title and components."""
        assert main_window.windowTitle() == "Nail-Biting Detection"
        assert main_window.camera_label is not None
        assert main_window.toggle_button is not None
        assert main_window.sensitivity_slider is not None
        assert main_window.status_label is not None
    
    def test_toggle_monitoring(self, main_window):
        """Test the monitoring toggle functionality."""
        # Initial state
        assert not main_window.is_monitoring
        assert main_window.toggle_button.text() == "Start Monitoring"
        assert main_window.status_label.text() == "Status: Not Monitoring"
        
        # Toggle on
        main_window.toggle_button.click()
        assert main_window.is_monitoring
        assert main_window.toggle_button.text() == "Stop Monitoring"
        assert main_window.status_label.text() == "Status: Monitoring"
        
        # Toggle off
        main_window.toggle_button.click()
        assert not main_window.is_monitoring
        assert main_window.toggle_button.text() == "Start Monitoring"
        assert main_window.status_label.text() == "Status: Not Monitoring"
    
    def test_sensitivity_slider(self, main_window):
        """Test the sensitivity slider initialization and range."""
        assert main_window.sensitivity_slider.minimum() == 1
        assert main_window.sensitivity_slider.maximum() == 10
        assert main_window.sensitivity_slider.value() == 5  # Default value
    
    def test_camera_setup(self, main_window):
        """Test camera initialization."""
        assert main_window.camera is not None
        assert main_window.timer is not None
        assert not main_window.is_monitoring
    
    def test_logger_setup(self, main_window):
        """Test logger initialization."""
        assert main_window.logger is not None
        # Check if log directory exists
        log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'logs')
        assert os.path.exists(log_dir)
    
    def test_close_event(self, main_window):
        """Test proper cleanup on window close."""
        main_window.close()
        # Camera should be released
        assert not main_window.camera.isOpened() 