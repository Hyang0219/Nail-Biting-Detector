from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, 
                               QPushButton, QLabel, QSlider, QHBoxLayout)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
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
        
        # Sensitivity slider
        sensitivity_layout = QVBoxLayout()
        sensitivity_label = QLabel("Detection Sensitivity:")
        self.sensitivity_slider = QSlider(Qt.Orientation.Horizontal)
        self.sensitivity_slider.setMinimum(1)
        self.sensitivity_slider.setMaximum(10)
        self.sensitivity_slider.setValue(5)
        sensitivity_layout.addWidget(sensitivity_label)
        sensitivity_layout.addWidget(self.sensitivity_slider)
        controls_layout.addLayout(sensitivity_layout)
        
        # Status label
        self.status_label = QLabel("Status: Not Monitoring")
        controls_layout.addWidget(self.status_label)
        
        # Statistics label
        self.stats_label = QLabel("Today's detections: 0")
        controls_layout.addWidget(self.stats_label)
        
        layout.addLayout(controls_layout)
        
    def setup_camera(self):
        self.camera = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.is_monitoring = False
        
    def setup_detector(self):
        self.detector = GestureDetector()
        
    def setup_logger(self):
        self.logger = setup_logger()
        
    def setup_analytics(self):
        self.analytics = Analytics()
        
    def setup_alert_sound(self):
        """Generate and prepare alert sound."""
        # Generate a simple beep sound
        frequency = 440  # Hz (A4 note)
        duration = 0.5  # seconds
        sample_rate = 44100  # Hz
        t = np.linspace(0, duration, int(sample_rate * duration))
        samples = (np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)
        self.alert_sound = sa.WaveObject(samples, 1, 2, sample_rate)
        
    def toggle_monitoring(self):
        self.is_monitoring = not self.is_monitoring
        if self.is_monitoring:
            self.toggle_button.setText("Stop Monitoring")
            self.status_label.setText("Status: Monitoring")
            self.timer.start(30)  # 30ms = ~33 fps
        else:
            self.toggle_button.setText("Start Monitoring")
            self.status_label.setText("Status: Not Monitoring")
            self.timer.stop()
            # End analytics session
            self.analytics.end_session()
            # Update statistics display
            self.update_statistics()
            
    def update_frame(self):
        ret, frame = self.camera.read()
        if ret:
            # Process frame for nail-biting detection
            frame, is_biting = self.detector.process_frame(frame)
            
            if is_biting:
                self.on_nail_biting_detected()
            
            # Convert frame to Qt format for display
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self.camera_label.setPixmap(QPixmap.fromImage(qt_image))
            
    def on_nail_biting_detected(self):
        # Log the event
        self.logger.info("Nail biting detected")
        
        # Record in analytics
        self.analytics.record_detection()
        
        # Update statistics display
        self.update_statistics()
        
        # Play alert sound
        try:
            self.alert_sound.play()
        except Exception as e:
            self.logger.error(f"Failed to play alert sound: {e}")
            
    def update_statistics(self):
        """Update the statistics display."""
        daily_summary = self.analytics.get_daily_summary(days=1)[0]
        self.stats_label.setText(f"Today's detections: {daily_summary['detections']}")
            
    def closeEvent(self, event):
        if self.is_monitoring:
            self.analytics.end_session()
        self.camera.release()
        super().closeEvent(event) 