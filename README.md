# Nail-Biting Detection Application

This application uses computer vision to detect nail-biting behavior and alerts users in real-time. It uses your computer's webcam to monitor hand movements near the mouth and provides immediate feedback when nail-biting is detected.

## Features

- Real-time detection of nail-biting gestures using webcam
- Hybrid detection approach:
  - Geometric detection using MediaPipe hand and face landmarks
  - ML-based detection using MobileNetV3-Large model
- Intelligent ROI (Region of Interest) extraction focused on hand-mouth interaction
- Visual feedback with detection status and confidence levels
- Audible alerts when nail-biting is detected
- Session tracking and daily statistics
- Adjustable sensitivity settings

Version 3.0
- Hybrid Detection System:
  * Combines geometric approach with ML-based detection
  * Uses ML model as a supplementary signal to boost confidence
  * Implements normalized confidence scoring for more accurate detection
  * Optimized for real-time performance with minimal false positives
- Improved ML model training with class balancing and data augmentation
- Enhanced visualization with prediction confidence display
- Fine-tuned detection thresholds based on real-world testing

Version 2.0
- MediaPipe Hand Pose + Geometric Approach (Core Detection):
  * Uses MediaPipe's hand landmarks for reliable detection
  * Calculates the distance between fingertips and mouth landmarks
  * Checks finger orientation relative to the mouth
  * Includes gesture recognition for common nail-biting poses
- Added support for Python 3.12
- Updated to latest compatible libraries (TensorFlow 2.18+, MediaPipe 0.10.18+)

Version 1.0
- MediaPipe Hand Pose + Geometric Approach (Quick Solution):
  * Instead of relying on the pre-trained classification model, the model uses MediaPipe's hand landmarks
  * Calculate the distance between fingertips and mouth landmarks
  * Check finger orientation relative to the mouth
  * Add gesture recognition for common nail-biting poses
  * This would be more reliable and faster than the current ML approach as an MVP



## Requirements

- Python 3.8+
- Webcam
- Required libraries (see requirements.txt)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/nail-biting-detection.git
cd nail-biting-detection
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the application:
```bash
python src/main.py
```

2. Adjust detection settings:
   - Detection Sensitivity: Controls the geometric detection thresholds
   - ML Confidence: Sets the confidence threshold for ML-based detection

3. Click "Start Monitoring" to begin detection

## How It Works

### Detection Approach

The application uses a hybrid approach to detect nail-biting:

1. **Geometric Detection**:
   - Tracks hand and face landmarks using MediaPipe
   - Calculates distance between fingertips and mouth
   - Analyzes finger angles and positions
   - Detects when fingers are clustered near the mouth

2. **ML-Based Detection**:
   - Uses MobileNetV3-Large model trained on nail-biting images
   - Focuses on the hand-mouth interaction area with intelligent ROI extraction
   - Provides confidence score for nail-biting detection

3. **State Machine**:
   - Tracks potential nail-biting gestures over time
   - Requires consistent detection across multiple frames
   - Implements cooldown period to avoid repeated alerts

### Improved ROI Focus

The application uses an intelligent ROI extraction method that:
- Dynamically adjusts padding based on hand-mouth distance
- Creates a square ROI to avoid distortion
- Applies contrast normalization to enhance features
- Centers the ROI on the hand-mouth interaction area

### Advanced Data Augmentation

Training employs enhanced data augmentation techniques:
- **MixUp**: Combines pairs of images and their labels with varying weights
- **Color space transformations**: HSV shifts, brightness/contrast adjustments
- **Channel shifting**: Random modifications to RGB channel intensities
- **Random noise**: Simulates varying lighting and image quality conditions

## Development

### Training the Model

To train the ML model with your own data:

1. Prepare your dataset:
   - Place nail-biting images in `data/raw/nail_biting/`
   - Place non-nail-biting images in `data/raw/non_nail_biting/`

2. Process the dataset:
```bash
python detection/data_processor.py
```

3. Train the model:
```bash
python detection/train_model.py
```

## Version History

### Version 3.0
- Implemented hybrid detection system combining geometric and ML approaches
- Added normalized confidence scoring for more accurate detection
- Optimized ML model integration with geometric detection
- Fine-tuned detection thresholds based on extensive testing
- Improved visualization with prediction confidence display
- Enhanced state machine logic for more reliable detection
- Added detailed logging for better debugging and analysis

### Version 2.0
- MediaPipe Hand Pose + Geometric Approach (Core Detection):
  * Uses MediaPipe's hand landmarks for reliable detection
  * Calculates the distance between fingertips and mouth landmarks
  * Checks finger orientation relative to the mouth
  * Includes gesture recognition for common nail-biting poses
- Added support for Python 3.12
- Updated to latest compatible libraries (TensorFlow 2.18+, MediaPipe 0.10.18+)

### Version 1.0
- MediaPipe Hand Pose + Geometric Approach (Quick Solution):
  * Instead of relying on the pre-trained classification model, the model uses MediaPipe's hand landmarks
  * Calculate the distance between fingertips and mouth landmarks
  * Check finger orientation relative to the mouth
  * Add gesture recognition for common nail-biting poses
  * This would be more reliable and faster than the current ML approach as an MVP

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- MediaPipe for hand and face landmark detection
- TensorFlow for ML model training and inference
- PySide6 for the GUI framework