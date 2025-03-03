# Standard library imports
import os
import sys
import platform
import subprocess
import threading
import time
import traceback
from datetime import datetime, timedelta
import logging

# Add the project root to the path for relative imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(src_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# GUI imports
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, 
    QPushButton, QLabel, QSlider, QHBoxLayout
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap

# Computer vision imports
import cv2
import mediapipe as mp
import numpy as np

# Local imports
from detection.gesture_detector import GestureDetector
from src.utils.logger import setup_logger
from src.utils.analytics import Analytics

# Import the gesture detector
from detection.gesture_detector import GestureDetector, GestureState

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Nail-Biting Detection")
        self.setup_ui()
        self.setup_camera()
        self.setup_detector()
        self.setup_logger()
        self.setup_analytics()
        self.setup_alert_sound()
        self.last_alert_time = datetime.now()
        self.alert_cooldown = 2  # seconds
        
    def setup_ui(self):
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Create camera feed display
        self.camera_label = QLabel()
        self.camera_label.setMinimumSize(640, 480)
        layout.addWidget(self.camera_label)
        
        # Create controls
        controls_layout = QHBoxLayout()
        
        # Start/Stop button
        self.toggle_button = QPushButton("Start Monitoring")
        self.toggle_button.clicked.connect(self.toggle_monitoring)
        controls_layout.addWidget(self.toggle_button)
        
        # Sensitivity slider
        sensitivity_layout = QVBoxLayout()
        sensitivity_label = QLabel("Detection Sensitivity:")
        self.sensitivity_slider = QSlider(Qt.Orientation.Horizontal)
        self.sensitivity_slider.setMinimum(1)
        self.sensitivity_slider.setMaximum(10)
        self.sensitivity_slider.setValue(5)
        self.sensitivity_slider.valueChanged.connect(self.update_sensitivity)
        sensitivity_layout.addWidget(sensitivity_label)
        sensitivity_layout.addWidget(self.sensitivity_slider)
        controls_layout.addLayout(sensitivity_layout)
        
        # Status label
        self.status_label = QLabel("Status: Not Monitoring")
        controls_layout.addWidget(self.status_label)
        
        # Statistics label
        self.stats_label = QLabel("Today's detections: 0")
        controls_layout.addWidget(self.stats_label)
        
        # Add model type indicator
        self.model_type_label = QLabel("Model: None")
        self.model_type_label.setStyleSheet("background-color: #333; color: white; padding: 3px; border-radius: 3px;")
        controls_layout.addWidget(self.model_type_label)
        
        layout.addLayout(controls_layout)
        
    def setup_camera(self):
        """Setup webcam capture."""
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.is_monitoring = False
        self.last_model_used = "None"
        
    def setup_detector(self):
        """Setup gesture detector."""
        # Use the best performing model
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                'models', 'mobilenet_model_20250302-182824.keras')
        self.detector = GestureDetector(model_path=model_path, 
                                      sensitivity=self.sensitivity_slider.value() / 10.0)
        
    def setup_logger(self):
        self.logger = setup_logger()
        
    def setup_analytics(self):
        self.analytics = Analytics()
        
    def setup_alert_sound(self):
        """Set up platform-specific alert methods."""
        try:
            self.system = platform.system()
            self.logger.info(f"Setting up alert for platform: {self.system}")
            
            # Get the path to the custom alert sound
            self.alert_sound_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                                               'assets', 'sound', 'apt.WAV')
            
            if not os.path.exists(self.alert_sound_path):
                self.logger.warning(f"Custom alert sound not found at {self.alert_sound_path}, will use system sounds")
                self.alert_sound_path = None
            else:
                self.logger.info(f"Found custom alert sound at {self.alert_sound_path}")
            
            # Test the alert system
            if self.system == "Darwin":  # macOS
                try:
                    # Check if we can play sounds via afplay
                    subprocess.run(["afplay", "--help"], 
                                  stdout=subprocess.DEVNULL, 
                                  stderr=subprocess.DEVNULL)
                    self.sound_loaded = True
                except:
                    self.logger.warning("afplay not available on macOS, alert sounds will be disabled")
                    self.sound_loaded = False
            elif self.system == "Windows":
                try:
                    # Check if we can use winsound
                    import winsound
                    self.sound_loaded = True
                except ImportError:
                    self.logger.warning("winsound not available on Windows, alert sounds will be disabled")
                    self.sound_loaded = False
            elif self.system == "Linux":
                try:
                    # Try paplay first (PulseAudio)
                    subprocess.run(["paplay", "--version"], 
                                  stdout=subprocess.DEVNULL, 
                                  stderr=subprocess.DEVNULL)
                    self.sound_loaded = True
                except:
                    try:
                        # Fallback to aplay (ALSA)
                        subprocess.run(["aplay", "--version"], 
                                      stdout=subprocess.DEVNULL, 
                                      stderr=subprocess.DEVNULL)
                        self.sound_loaded = True
                    except:
                        self.logger.warning("Neither paplay nor aplay available on Linux, alert sounds will be disabled")
                        self.sound_loaded = False
            else:
                self.logger.warning(f"Unsupported platform: {self.system}, alert sounds will be disabled")
                self.sound_loaded = False
                
            if self.sound_loaded:
                self.logger.info("Alert sound system initialized successfully")
            else:
                self.logger.warning("Alert sound system could not be initialized")
                
        except Exception as e:
            self.logger.error(f"Error setting up alert sound: {e}")
            self.logger.error(traceback.format_exc())
            self.sound_loaded = False
        
    def update_sensitivity(self):
        """Update detector sensitivity when slider changes."""
        if hasattr(self, 'detector'):
            # Convert slider value (1-10) to sensitivity (0.1-1.0)
            sensitivity = self.sensitivity_slider.value() / 10.0
            self.detector.update_sensitivity(sensitivity)
            
            # Update status label with current sensitivity value
            if self.is_monitoring:
                self.status_label.setText(f"Status: Monitoring Active (Sensitivity: {sensitivity:.1f})")
            else:
                self.status_label.setText(f"Status: Not Monitoring (Sensitivity: {sensitivity:.1f})")
            
            self.logger.info(f"Sensitivity updated to {sensitivity:.1f}")
        
    def toggle_monitoring(self):
        """Toggle webcam monitoring on/off."""
        if not self.is_monitoring:
            self.timer.start(30)  # 30ms = ~33 fps
            self.toggle_button.setText("Stop Monitoring")
            self.status_label.setText("Status: Monitoring Active")
            self.is_monitoring = True
        else:
            self.timer.stop()
            self.toggle_button.setText("Start Monitoring")
            self.status_label.setText("Status: Not Monitoring")
            self.is_monitoring = False
            # End analytics session
            try:
                self.analytics.end_session()
            except Exception as e:
                self.logger.error(f"Error ending analytics session: {e}")
            # Update statistics display
            self.update_statistics()
            
    def update_frame(self):
        """Process and display the next frame."""
        ret, frame = self.cap.read()
        if not ret:
            self.logger.error("Failed to capture frame from camera")
            return
            
        try:
            # Process frame with detector
            frame_with_detections, is_biting, model_used = self.detector.process_frame_with_model_info(frame)
            
            # Update model type indicator
            if model_used != self.last_model_used:
                self.last_model_used = model_used
                self.update_model_indicator(model_used)
            
            # Update status label based on detector state
            if hasattr(self.detector, 'current_state'):
                state = self.detector.current_state
                sensitivity = self.sensitivity_slider.value() / 10.0
                
                if state.name == "COOLDOWN":
                    # Show cooldown timer
                    now = datetime.now()
                    if self.detector.last_detection_time:
                        remaining = (self.detector.cooldown_period - (now - self.detector.last_detection_time)).total_seconds()
                        if remaining > 0:
                            self.status_label.setText(f"Status: Cooldown ({remaining:.1f}s)")
                            self.status_label.setStyleSheet("color: red; font-weight: bold;")
                elif self.is_monitoring:
                    self.status_label.setText(f"Status: Monitoring Active (Sensitivity: {sensitivity:.1f})")
                    self.status_label.setStyleSheet("")
            
            # Check if we should alert (with cooldown protection)
            now = datetime.now()
            if is_biting and (now - self.last_alert_time).total_seconds() > self.alert_cooldown:
                self.last_alert_time = now
                # Separate thread for alert to avoid blocking UI
                QTimer.singleShot(0, self.play_alert)
                QTimer.singleShot(0, self.log_event)
            
            # Convert frame to Qt format and display
            rgb_frame = cv2.cvtColor(frame_with_detections, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self.camera_label.setPixmap(QPixmap.fromImage(qt_image))
        except Exception as e:
            self.logger.error(f"Error processing frame: {e}")
            self.logger.error(traceback.format_exc())
            
    def update_model_indicator(self, model_used):
        """Update the model type indicator."""
        self.model_type_label.setText(f"Model: {model_used}")
        
        # Set background color based on model type
        if model_used == "ML":
            self.model_type_label.setStyleSheet("background-color: #007700; color: white; padding: 3px; border-radius: 3px;")
        elif model_used == "MediaPipe":
            self.model_type_label.setStyleSheet("background-color: #000077; color: white; padding: 3px; border-radius: 3px;")
        else:
            self.model_type_label.setStyleSheet("background-color: #333; color: white; padding: 3px; border-radius: 3px;")
            
    def play_alert(self):
        """Play alert sound using platform-specific methods."""
        if not hasattr(self, 'sound_loaded') or not self.sound_loaded:
            self.logger.warning("Alert sound not enabled, skipping alert")
            return
            
        try:
            self.logger.info("Playing alert sound")
            
            # Try to use the custom alert sound if available
            if hasattr(self, 'alert_sound_path') and self.alert_sound_path and os.path.exists(self.alert_sound_path):
                if self.system == "Darwin":  # macOS
                    subprocess.Popen(["afplay", self.alert_sound_path], 
                                   stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL)
                    return
                elif self.system == "Windows":
                    try:
                        import winsound
                        winsound.PlaySound(self.alert_sound_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
                        return
                    except ImportError:
                        self.logger.warning("winsound not available on Windows")
                elif self.system == "Linux":
                    try:
                        subprocess.Popen(["paplay", self.alert_sound_path],
                                       stdout=subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL)
                        return
                    except:
                        try:
                            subprocess.Popen(["aplay", "-q", self.alert_sound_path],
                                           stdout=subprocess.DEVNULL,
                                           stderr=subprocess.DEVNULL)
                            return
                        except:
                            self.logger.warning("Could not play custom sound on Linux, falling back to system sounds")
            
            # Fallback to system sounds if custom sound failed or is not available
            if self.system == "Darwin":  # macOS
                # Use the built-in macOS 'afplay' command with a system sound
                subprocess.Popen(["afplay", "/System/Library/Sounds/Ping.aiff"], 
                               stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
            elif self.system == "Windows":
                # Use winsound on Windows
                try:
                    import winsound
                    winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
                except ImportError:
                    self.logger.warning("winsound not available on Windows")
            elif self.system == "Linux":
                # Try to use a system sound on Linux
                try:
                    subprocess.Popen(["paplay", "/usr/share/sounds/freedesktop/stereo/bell.oga"],
                                   stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL)
                except:
                    try:
                        # Fallback to aplay with a simple beep
                        subprocess.Popen(["aplay", "-q", "/usr/share/sounds/alsa/Front_Center.wav"],
                                       stdout=subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL)
                    except:
                        self.logger.warning("Could not play sound on Linux")
            
        except Exception as e:
            self.logger.error(f"Failed to play alert sound: {e}")
            self.logger.error(traceback.format_exc())
            # Disable sound for future alerts to prevent further issues
            self.sound_loaded = False
            
    def log_event(self):
        # Log the event
        try:
            self.logger.info("Nail biting detected")
            
            # Record in analytics
            self.analytics.record_detection()
            
            # Update statistics display
            self.update_statistics()
        except Exception as e:
            self.logger.error(f"Error logging nail biting event: {e}")
            self.logger.error(traceback.format_exc())
        
    def update_statistics(self):
        """Update the statistics display."""
        try:
            daily_summary = self.analytics.get_daily_summary(days=1)[0]
            self.stats_label.setText(f"Today's detections: {daily_summary['detections']}")
        except Exception as e:
            self.logger.error(f"Error updating statistics: {e}")
            self.stats_label.setText("Today's detections: --")
            
    def closeEvent(self, event):
        if self.is_monitoring:
            try:
                self.analytics.end_session()
            except Exception as e:
                self.logger.error(f"Error ending analytics session during close: {e}")
        self.cap.release()
        super().closeEvent(event) 