#!/usr/bin/env python

"""
Verification script for Python 3.12 migration.
This script checks if all required dependencies are correctly installed
and compatible with the current Python environment.
"""

import sys
import platform
import importlib.util
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_module(module_name):
    """Check if a module is installed and return its version."""
    try:
        spec = importlib.util.find_spec(module_name)
        if spec is None:
            logger.error(f"❌ {module_name} is NOT installed")
            return False, None
        
        # Try to get the version
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, '__version__', 'Unknown version')
            logger.info(f"✅ {module_name} is installed (version: {version})")
            return True, version
        except (AttributeError, ImportError):
            logger.info(f"✅ {module_name} is installed (version unknown)")
            return True, "Unknown"
    except Exception as e:
        logger.error(f"❌ Error checking {module_name}: {str(e)}")
        return False, None

def verify_python_version():
    """Verify the Python version."""
    python_version = platform.python_version()
    logger.info(f"Python version: {python_version}")
    
    major, minor, _ = map(int, python_version.split('.'))
    if major == 3 and minor >= 12:
        logger.info("✅ Python 3.12+ detected")
        return True
    else:
        logger.warning(f"⚠️ Python version is {python_version}. Python 3.12+ recommended for this project.")
        return False

def verify_dependencies():
    """Verify that all necessary dependencies are installed."""
    required_modules = [
        # Core dependencies
        "numpy", "pandas", "scipy",
        
        # Computer Vision and ML
        "cv2", "mediapipe", "tensorflow", "keras",
        
        # GUI
        "PySide6",
        
        # Audio
        "simpleaudio",
        
        # Additional dependencies
        "matplotlib", "seaborn"
    ]
    
    successful = 0
    failed = 0
    
    for module in required_modules:
        success, _ = check_module(module)
        if success:
            successful += 1
        else:
            failed += 1
    
    logger.info(f"Dependency check complete: {successful} modules installed, {failed} modules missing")
    return failed == 0

def verify_tf_keras_compatibility():
    """Verify TensorFlow and Keras compatibility."""
    try:
        import tensorflow as tf
        import keras
        
        tf_version = tf.__version__
        keras_version = keras.__version__
        
        logger.info(f"TensorFlow version: {tf_version}")
        logger.info(f"Keras version: {keras_version}")
        
        # Try to create a simple model to verify functionality
        try:
            logger.info("Testing TensorFlow/Keras model creation...")
            model = keras.Sequential([
                keras.layers.Dense(10, activation='relu', input_shape=(5,)),
                keras.layers.Dense(1, activation='sigmoid')
            ])
            model.compile(optimizer='adam', loss='binary_crossentropy')
            logger.info("✅ TensorFlow/Keras model creation successful")
            return True
        except Exception as e:
            logger.error(f"❌ TensorFlow/Keras model creation failed: {str(e)}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error verifying TensorFlow/Keras: {str(e)}")
        return False

def verify_gui():
    """Verify PySide6 GUI functionality."""
    try:
        from PySide6.QtCore import qVersion
        logger.info(f"Qt version: {qVersion()}")
        
        try:
            # Try to create a QApplication instance with offscreen platform
            os.environ["QT_QPA_PLATFORM"] = "offscreen"
            from PySide6.QtWidgets import QApplication
            app = QApplication.instance() or QApplication([])
            
            # Check if we can create widgets
            from PySide6.QtWidgets import QLabel
            label = QLabel("Test")
            logger.info("✅ PySide6 QApplication and widget creation successful")
            return True
        except Exception as e:
            logger.error(f"❌ PySide6 initialization failed: {str(e)}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error verifying PySide6: {str(e)}")
        return False

def main():
    """Main verification function."""
    logger.info("==== Python 3.12 Migration Verification ====")
    
    # Check Python version
    py_check = verify_python_version()
    
    # Check dependencies
    dep_check = verify_dependencies()
    
    # Check TensorFlow/Keras
    tf_check = verify_tf_keras_compatibility()
    
    # Check GUI
    gui_check = verify_gui()
    
    # Final verdict
    if all([py_check, dep_check, tf_check, gui_check]):
        logger.info("✅ ALL CHECKS PASSED! Your Python 3.12 migration is successful.")
    else:
        logger.warning("⚠️ Some checks failed. Please review the logs above.")
        if not py_check:
            logger.info("  - Python version should be updated to 3.12+")
        if not dep_check:
            logger.info("  - Some dependencies are missing")
        if not tf_check:
            logger.info("  - TensorFlow/Keras compatibility issues")
        if not gui_check:
            logger.info("  - PySide6/GUI compatibility issues")

if __name__ == "__main__":
    main()
