from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, 
                               QPushButton, QLabel, QSlider, QHBoxLayout,
                               QGroupBox, QGridLayout)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
import cv2
import mediapipe as mp
from datetime import datetime
import logging
import simpleaudio as sa
import numpy as np
import os

from detection.gesture_detector import GestureDetector
from utils.logger import setup_logger
from utils.analytics import Analytics

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
        
        # Detection settings group
        detection_group = QGroupBox("Detection Settings")
        detection_layout = QGridLayout()
        detection_group.setLayout(detection_layout)
        
        # Sensitivity slider
        sensitivity_label = QLabel("Detection Sensitivity:")
        self.sensitivity_slider = QSlider(Qt.Orientation.Horizontal)
        self.sensitivity_slider.setMinimum(1)
        self.sensitivity_slider.setMaximum(10)
        self.sensitivity_slider.setValue(5)
        self.sensitivity_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.sensitivity_slider.setTickInterval(1)
        self.sensitivity_slider.valueChanged.connect(self.update_sensitivity)
        detection_layout.addWidget(sensitivity_label, 0, 0)
        detection_layout.addWidget(self.sensitivity_slider, 0, 1)
        
        # ML Confidence threshold slider
        threshold_label = QLabel("ML Confidence:")
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setMinimum(10)
        self.threshold_slider.setMaximum(90)
        self.threshold_slider.setValue(50)  # Default 0.5
        self.threshold_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.threshold_slider.setTickInterval(10)
        self.threshold_slider.valueChanged.connect(self.update_threshold)
        detection_layout.addWidget(threshold_label, 1, 0)
        detection_layout.addWidget(self.threshold_slider, 1, 1)
        
        # Add settings group to controls
        controls_layout.addWidget(detection_group)
        
        # Status label
        self.status_label = QLabel("Status: Not Monitoring")
        controls_layout.addWidget(self.status_label)
        
        # Statistics label
        self.stats_label = QLabel("Today's detections: 0")
        controls_layout.addWidget(self.stats_label)
        
        layout.addLayout(controls_layout)
        
    def setup_camera(self):
        """Setup webcam capture."""
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.is_monitoring = False
        
    def setup_detector(self):
        """Setup gesture detector."""
        # Use the best performing model
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                'models', 'mobilenet_model.keras')
        
        # Get current settings from UI
        sensitivity = self.sensitivity_slider.value() / 10.0
        
        self.detector = GestureDetector(
            model_path=model_path, 
            sensitivity=sensitivity
        )
        
        # Update status label
        if self.detector.model is not None:
            self.status_label.setText("Status: ML model loaded")
        else:
            self.status_label.setText("Status: Geometric detection only")
        
    def setup_logger(self):
        self.logger = setup_logger()
        
    def setup_analytics(self):
        self.analytics = Analytics()
        
    def setup_alert_sound(self):
        """Set up alert sound with multiple fallback options."""
        try:
            # First try to use QSound if available
            try:
                from PySide6.QtMultimedia import QSoundEffect
                from PySide6.QtCore import QUrl
                
                self.sound_system = "qsound"
                self.sound_effect = QSoundEffect()
                
                # Try to load a WAV file if it exists
                sound_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                         'assets', 'alert.wav')
                
                if os.path.exists(sound_file):
                    self.sound_effect.setSource(QUrl.fromLocalFile(sound_file))
                    self.logger.info(f"Loaded alert sound from {sound_file}")
                else:
                    # Create assets directory if it doesn't exist
                    assets_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'assets')
                    os.makedirs(assets_dir, exist_ok=True)
                    
                    # Generate a simple beep sound using simpleaudio as fallback
                    self.generate_wav_file(sound_file)
                    self.sound_effect.setSource(QUrl.fromLocalFile(sound_file))
                    self.logger.info(f"Generated and loaded alert sound to {sound_file}")
                
                self.sound_effect.setVolume(0.5)
                self.logger.info("Alert sound initialized using QSoundEffect")
                return
            except ImportError:
                self.logger.warning("QSoundEffect not available, falling back to simpleaudio")
            
            # Fallback to simpleaudio
            self.sound_system = "simpleaudio"
            frequency = 440  # Hz (A4 note)
            duration = 0.5  # seconds
            sample_rate = 44100  # Hz
            t = np.linspace(0, duration, int(sample_rate * duration))
            samples = (np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)
            self.alert_sound = sa.WaveObject(samples, 1, 2, sample_rate)
            self.logger.info("Alert sound initialized using simpleaudio")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize alert sound: {e}")
            self.sound_system = "none"
            self.alert_sound = None
    
    def generate_wav_file(self, file_path):
        """Generate a WAV file for alert sound."""
        try:
            import wave
            
            # Generate a simple beep sound
            frequency = 440  # Hz (A4 note)
            duration = 0.5  # seconds
            sample_rate = 44100  # Hz
            t = np.linspace(0, duration, int(sample_rate * duration))
            samples = (np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)
            
            # Write to WAV file
            with wave.open(file_path, 'w') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(samples.tobytes())
                
            self.logger.info(f"Generated WAV file at {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to generate WAV file: {e}")
            
    def update_sensitivity(self):
        """Update detector sensitivity when slider changes."""
        if hasattr(self, 'detector'):
            # Convert slider value (1-10) to sensitivity (0.1-1.0)
            sensitivity = self.sensitivity_slider.value() / 10.0
            self.detector.update_sensitivity(sensitivity)
            
    def update_threshold(self):
        """Update ML threshold when slider changes."""
        if hasattr(self, 'detector'):
            # Convert slider value (10-90) to threshold (0.1-0.9)
            threshold = self.threshold_slider.value() / 100.0
            self.detector.ml_confidence_threshold = threshold
        
    def toggle_monitoring(self):
        """Toggle webcam monitoring on/off."""
        if not self.is_monitoring:
            self.timer.start(30)  # 30ms = ~33 fps
            self.toggle_button.setText("Stop Monitoring")
            self.is_monitoring = True
            # Update status label
            current_status = self.status_label.text().split(": ")[1]
            self.status_label.setText(f"Status: Monitoring ({current_status})")
        else:
            self.timer.stop()
            self.toggle_button.setText("Start Monitoring")
            self.is_monitoring = False
            # End analytics session
            self.analytics.end_session()
            # Update statistics display
            self.update_statistics()
            # Update status label
            if self.detector.model is not None:
                self.status_label.setText("Status: ML model loaded")
            else:
                self.status_label.setText("Status: Geometric detection only")
            
    def update_frame(self):
        """Process and display the next frame."""
        try:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                self.logger.warning("Failed to read frame from camera")
                return
                
            # Process frame with detector
            try:
                frame_with_detections, is_biting = self.detector.process_frame(frame)
                
                if is_biting:
                    self.play_alert()
                    self.log_event()
                
                # Convert frame to Qt format and display
                rgb_frame = cv2.cvtColor(frame_with_detections, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_frame.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                self.camera_label.setPixmap(QPixmap.fromImage(qt_image))
            except Exception as e:
                self.logger.error(f"Error processing frame: {e}")
                # Display the original frame if processing fails
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_frame.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                self.camera_label.setPixmap(QPixmap.fromImage(qt_image))
        except Exception as e:
            self.logger.error(f"Critical error in update_frame: {e}")
            
    def play_alert(self):
        """Play alert sound with multiple fallback mechanisms."""
        try:
            if self.sound_system == "qsound":
                if hasattr(self, 'sound_effect'):
                    self.sound_effect.play()
            elif self.sound_system == "simpleaudio":
                if hasattr(self, 'alert_sound') and self.alert_sound is not None:
                    play_obj = self.alert_sound.play()
                    # Don't wait for playback to finish to avoid blocking the UI
            else:
                self.logger.warning("No sound system available, skipping alert sound")
        except Exception as e:
            self.logger.error(f"Failed to play alert sound: {e}")
            # Don't try to reinitialize here to avoid potential crash
            
    def log_event(self):
        # Log the event
        self.logger.info("Nail biting detected")
        
        # Record in analytics
        self.analytics.record_detection()
        
        # Update statistics display
        self.update_statistics()
        
    def update_statistics(self):
        """Update the statistics display."""
        daily_summary = self.analytics.get_daily_summary(days=1)[0]
        self.stats_label.setText(f"Today's detections: {daily_summary['detections']}")
            
    def closeEvent(self, event):
        if self.is_monitoring:
            self.analytics.end_session()
        self.cap.release()
        super().closeEvent(event) 