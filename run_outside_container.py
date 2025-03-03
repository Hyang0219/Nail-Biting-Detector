#!/usr/bin/env python3
"""
Run script for nail-biting detection application.
This script ensures the correct Python path is set up and dependencies are met.
"""

import os
import sys
import platform
import argparse
import traceback
from datetime import datetime

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Nail Biting Detection Application')
    parser.add_argument('--headless', action='store_true', help='Run in headless mode (no GUI)')
    parser.add_argument('--test', action='store_true', help='Run tests in headless mode')
    return parser.parse_args()

def ensure_directories_exist():
    """Ensure all necessary directories exist."""
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Create required directories
    dirs = ['logs', 'data', 'models']
    for dir_name in dirs:
        dir_path = os.path.join(project_root, dir_name)
        os.makedirs(dir_path, exist_ok=True)
    
    # Create asset directories
    asset_dirs = ['assets/sound', 'assets/stickers']
    for dir_name in asset_dirs:
        dir_path = os.path.join(project_root, dir_name)
        os.makedirs(dir_path, exist_ok=True)
    
    print("✓ Ensured all necessary directories exist.")

def check_dependencies():
    """Check if required packages are installed."""
    missing_packages = []
    optional_packages = []
    
    try:
        import tensorflow
        print(f"✓ Found tensorflow {tensorflow.__version__}")
    except ImportError as e:
        if platform.system() == "Darwin":  # macOS
            missing_packages.append("tensorflow-macos")
        else:
            missing_packages.append("tensorflow")
        print(f"✗ Failed to import tensorflow: {e}")
    
    try:
        import PySide6
        from PySide6 import __version__
        print(f"✓ Found PySide6 {__version__}")
    except ImportError as e:
        missing_packages.append("PySide6")
        print(f"✗ Failed to import PySide6: {e}")
    
    try:
        import mediapipe
        print(f"✓ Found mediapipe {mediapipe.__version__}")
    except ImportError as e:
        missing_packages.append("mediapipe")
        print(f"✗ Failed to import mediapipe: {e}")
    
    try:
        import cv2
        print(f"✓ Found opencv-python {cv2.__version__}")
    except ImportError as e:
        missing_packages.append("opencv-python")
        print(f"✗ Failed to import cv2: {e}")
    
    # Check for optional packages
    try:
        import imageio
        print(f"✓ Found imageio {imageio.__version__}")
    except ImportError as e:
        optional_packages.append("imageio")
        print(f"ℹ Optional package imageio not found: {e}")
    except Exception as e:
        print(f"ℹ Error with imageio: {e}")
        optional_packages.append("imageio")
    
    if optional_packages:
        print(f"\nℹ Some optional packages are missing: {', '.join(optional_packages)}")
        print("These packages are not required but may enhance functionality.")
        print("You can install them using: pip install " + " ".join(optional_packages))
    
    return missing_packages

def main():
    # Add the project root to the Python path
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_root)
    
    # Parse command line arguments
    args = parse_args()
    
    # Ensure directories exist
    ensure_directories_exist()
    
    # Check for required packages
    missing_required = check_dependencies()
    if missing_required:
        print(f"❌ Missing required packages: {', '.join(missing_required)}")
        print("Please install them using: pip install -r requirements.txt")
        return
    
    # Check if model exists
    model_path = os.path.join(project_root, 'models', 'mobilenet_model_20250302-182824.keras')
    if os.path.exists(model_path):
        print(f"✓ Model found: {os.path.basename(model_path)}")
    else:
        print(f"⚠️ Warning: Model not found at {model_path}")
        print("The application will run without ML-based detection.")
    
    print("\n🚀 Starting nail-biting detection application...")
    
    try:
        if args.headless or args.test:
            # Run in headless mode
            print("Running in headless mode...")
            
            if args.test:
                # Run tests
                print("Running tests...")
                from detection.gesture_detector import GestureDetector
                import cv2
                import numpy as np
                
                # Create a test frame
                test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                
                # Initialize detector
                detector = GestureDetector(model_path=model_path if os.path.exists(model_path) else None)
                
                # Test sticker loading
                print(f"Loaded {len(detector.stickers)} stickers")
                
                # Test overlay_sticker method
                result_frame = detector.overlay_sticker(test_frame.copy())
                
                # Force sticker display for testing
                detector.sticker_start_time = datetime.now()
                detector.current_sticker_index = 0
                
                # Test again with sticker display forced
                result_frame = detector.overlay_sticker(test_frame.copy())
                
                print("✓ Headless tests completed successfully")
            else:
                print("Headless mode active. No GUI will be displayed.")
                print("Use --test flag to run tests in headless mode.")
        else:
            # Run with GUI
            from PySide6.QtWidgets import QApplication
            from src.gui.main_window import MainWindow
            
            app = QApplication(sys.argv)
            window = MainWindow()
            window.show()
            sys.exit(app.exec())
    except Exception as e:
        print(f"❌ Error running the application: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 