#!/usr/bin/env python3
# verify_app.py - Script to verify the application is ready to run

import os
import sys
import importlib
import platform
from datetime import datetime

# Add the project root to the path for relative imports
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(src_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def check_python_version():
    """Check if Python version is compatible."""
    print(f"Python version: {platform.python_version()}")
    major, minor, _ = platform.python_version_tuple()
    if int(major) < 3 or (int(major) == 3 and int(minor) < 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print("✓ Python version is compatible")
    return True

def check_dependencies():
    """Check if required packages are installed."""
    required_packages = [
        "numpy", "pandas", "PySide6", "opencv-python", "mediapipe", 
        "tensorflow", "keras", "imageio", "PIL", "scikit-learn"
    ]
    
    optional_packages = [
        "matplotlib", "seaborn", "sounddevice", "soundfile"
    ]
    
    missing_required = []
    missing_optional = []
    
    for package in required_packages:
        try:
            if package == "opencv-python":
                # Special case for opencv-python
                import cv2
                print(f"✓ Found {package} {cv2.__version__}")
            elif package == "PIL":
                # Special case for Pillow/PIL
                import PIL
                print(f"✓ Found pillow {PIL.__version__}")
            elif package == "scikit-learn":
                # Special case for scikit-learn
                import sklearn
                print(f"✓ Found {package} {sklearn.__version__}")
            else:
                importlib.import_module(package.replace("-", "_"))
                print(f"✓ Found {package}")
        except ImportError as e:
            missing_required.append(package)
            print(f"❌ Missing required package: {package} ({str(e)})")
    
    for package in optional_packages:
        try:
            if package == "sounddevice":
                # Special case for sounddevice which might raise OSError
                try:
                    import sounddevice
                    print(f"✓ Found optional package: {package}")
                except OSError as e:
                    print(f"⚠️ Optional package {package} found but has issues: {str(e)}")
            else:
                importlib.import_module(package)
                print(f"✓ Found optional package: {package}")
        except ImportError as e:
            missing_optional.append(package)
            print(f"⚠️ Missing optional package: {package}")
    
    if missing_required:
        print(f"\n❌ Missing {len(missing_required)} required packages")
        print("Please install them using: pip install -r requirements.txt")
        return False
    
    if missing_optional:
        print(f"\n⚠️ Missing {len(missing_optional)} optional packages")
        print("These are not required but may enhance functionality")
    
    return True

def check_directories():
    """Check if required directories exist."""
    required_dirs = ['logs', 'data', 'models', 'assets/sound', 'assets/stickers']
    missing_dirs = []
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            missing_dirs.append(dir_path)
            print(f"❌ Missing directory: {dir_path}")
        else:
            print(f"✓ Found directory: {dir_path}")
    
    if missing_dirs:
        print(f"\n❌ Missing {len(missing_dirs)} directories")
        print("Run the application once to create these directories automatically")
        return False
    
    return True

def check_model():
    """Check if model file exists."""
    model_path = os.path.join('models', 'mobilenet_model_20250302-182824.keras')
    if not os.path.exists(model_path):
        print(f"⚠️ Model file not found: {model_path}")
        print("The application will run without ML-based detection")
        return True  # Not critical
    
    print(f"✓ Found model file: {model_path}")
    return True

def check_stickers():
    """Check if sticker files exist."""
    stickers_dir = os.path.join('assets', 'stickers')
    if not os.path.exists(stickers_dir):
        print(f"❌ Stickers directory not found: {stickers_dir}")
        return False
    
    sticker_files = [f for f in os.listdir(stickers_dir) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp'))]
    
    if not sticker_files:
        print(f"❌ No sticker files found in {stickers_dir}")
        return False
    
    print(f"✓ Found {len(sticker_files)} sticker files: {', '.join(sticker_files)}")
    return True

def main():
    """Main verification function."""
    print("=" * 50)
    print("Nail Biting Detection Application Verification")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    
    checks = [
        ("Python Version", check_python_version()),
        ("Dependencies", check_dependencies()),
        ("Directories", check_directories()),
        ("Model", check_model()),
        ("Stickers", check_stickers())
    ]
    
    print("\n" + "=" * 50)
    print("Verification Summary")
    print("=" * 50)
    
    all_passed = True
    for name, result in checks:
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"{name}: {status}")
        if not result:
            all_passed = False
    
    print("\nFinal Result:", "✓ READY TO RUN" if all_passed else "❌ ISSUES DETECTED")
    
    if all_passed:
        print("\nTo run the application:")
        print("  python run_outside_container.py")
        print("\nTo run in headless mode:")
        print("  python run_outside_container.py --headless --test")
    else:
        print("\nPlease fix the issues above before running the application")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main()) 