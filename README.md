# Nail Biting Detection Application
(https://huggingface.co/datasets/alecsharpie/nailbiting_classification)

This application uses computer vision to detect nail-biting behavior and alerts users in real-time. It uses your computer's webcam to monitor hand movements near the mouth and provides immediate feedback when nail-biting is detected.

Version 3.0
- Optimized Hybrid Detection Pipeline:
  * Refined two-stage detection approach for better performance
  * First uses MediaPipe to detect hand-to-mouth proximity
  * Only triggers MobileNet ML model when geometric conditions suggest potential nail biting
  * Reduces computational load while maintaining high accuracy
- Enhanced Visual Feedback System:
  * Added animated sticker alerts with WebP and GIF support
  * Implemented real-time animation playback at native speeds
  * Improved visual alert visibility with semi-transparent backgrounds
  * Added customizable preferred sticker selection
- Improved Robustness:
  * Enhanced error handling for more reliable execution
  * Better management of optional dependencies
  * Fixed import path issues for consistent performance

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

## Features

- **Real-time detection**: Uses a hybrid approach combining MediaPipe hand tracking and ML classification
- **Visual feedback**: Shows which detection model is active (MediaPipe or ML)
- **Alert system**: Plays a sound and displays animated stickers when nail biting is detected
- **Adjustable sensitivity**: Slider to adjust detection sensitivity
- **Analytics tracking**: Records and displays nail-biting incidents
- **Cross-platform**: Works on macOS, Windows, and Linux

## Requirements

- Python 3.8 or higher
- Webcam
- Dependencies listed in `requirements.txt`

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/nail-biting-detection.git
   cd nail-biting-detection
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Make sure you have the trained model in the `models` directory:
   - The application expects to find `mobilenet_model_20250302-182824.keras` in the models directory
   - If you don't have this file, you'll need to train the model first (see Training section)

## Running the Application

Simply run:

```
python run_outside_container.py
```

or make it executable and run directly:

```
chmod +x run_outside_container.py
./run_outside_container.py
```

### Headless Mode

For environments without display capabilities (like containers), you can run the application in headless mode:

```
python run_outside_container.py --headless
```

To run tests in headless mode:

```
python run_outside_container.py --headless --test
```

This will verify that the core functionality works without requiring a GUI.

## Training the Model

If you need to train the model:

1. Download the dataset:
   ```
   python load_hf_dataset.py --convert-to-local
   ```

2. Train the model:
   ```
   python train_improved_model.py
   ```

3. Evaluate the model:
   ```
   python evaluate_model.py --model-path models/your_model_name.keras --data-dir data/hf_dataset --simple-mode
   ```

The model training uses the [Nail Biting Classification dataset](https://huggingface.co/datasets/alecsharpie/nailbiting_classification) from Hugging Face.

## Usage

1. Start the application using the command above
2. Click "Start Monitoring" to begin detection
3. Adjust the sensitivity slider as needed (higher values are more sensitive)
4. The "Model" indicator shows which detection method is active:
   - **MediaPipe**: Using geometric hand pose detection only (blue)
   - **ML**: Using the trained ML model for classification (green)
5. The application will alert you when nail biting is detected with:
   - A sound alert
   - An animated sticker displayed on screen
6. Statistics are displayed and logged for future reference

## Project Structure

- `src/`: Source code for the application
  - `gui/`: GUI components
  - `utils/`: Utility functions (analytics, logging)
- `detection/`: Detection algorithms and model training
- `models/`: Trained ML models
- `data/`: Dataset and analytics data
- `logs/`: Application logs
- `assets/`: Application assets
  - `sound/`: Alert sound files
  - `stickers/`: Visual stickers/GIFs for alerts
- `run_outside_container.py`: Main script to run the application

## License

[MIT License](LICENSE)