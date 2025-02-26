# Nail-Biting Detection Application

This application uses computer vision to detect nail-biting behavior and alerts users in real-time. It uses your computer's webcam to monitor hand movements near the mouth and provides immediate feedback when nail-biting is detected.

Version 2.0
- MediaPipe Hand Pose + Geometric Approach (Core Detection):
  * Uses MediaPipe's hand landmarks for reliable detection
  * Calculates the distance between fingertips and mouth landmarks
  * Checks finger orientation relative to the mouth
  * Includes gesture recognition for common nail-biting poses
- Added support for Python 3.12
- Updated to latest compatible libraries (TensorFlow 2.18+, MediaPipe 0.10.18+)

## Features

- Real-time webcam monitoring
- Hand and face detection using MediaPipe
- Visual feedback with colored rectangles:
  - Blue: Hand detection
  - Green: Mouth detection
  - Red: Nail-biting detected
- Audio alerts when nail-biting is detected
- Adjustable detection sensitivity
- Event logging for tracking behavior

## Requirements

- Python 3.8-3.12 (Python 3.12 recommended)
- Webcam
- The required Python packages are listed in `requirements.txt`

## Installation

### Option 1: Using Dev Container (Recommended)

If you're using VS Code with the Dev Containers extension:

1. Clone the repository:
```bash
git clone <repository-url>
cd nail-biting-detection
```

2. Open the project in VS Code and reopen in container when prompted.
The container will automatically set up Python 3.12 and install all dependencies.

### Option 2: Manual Setup with Python 3.12

1. Clone the repository:
```bash
git clone <repository-url>
cd nail-biting-detection
```

2. Run the setup script:
```bash
./setup_py312.sh
```

3. Activate the virtual environment:
```bash
source venv-py312/bin/activate
```

### Option 3: Traditional Setup (Python 3.8-3.10)

1. Clone the repository:
```bash
git clone <repository-url>
cd nail-biting-detection
```

2. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Generate the alert sound:
```bash
cd src/utils
python generate_alert.py
cd ../..
```

## Usage

1. Start the application:
```bash
cd src
python main.py
```

2. The application window will open with your webcam feed.

3. Click the "Start Monitoring" button to begin detection.

4. Adjust the sensitivity slider if needed:
   - Higher values: More sensitive to hand-mouth proximity
   - Lower values: Less sensitive

5. The application will:
   - Show blue rectangles around detected hands
   - Show a green rectangle around your mouth
   - Change rectangles to red when nail-biting is detected
   - Play an alert sound
   - Log the event

6. Click "Stop Monitoring" to pause detection.

## Logs

The application creates daily log files in the `logs` directory. Each log entry includes:
- Timestamp
- Event type
- Additional information (if any)

## Contributing

Feel free to submit issues and enhancement requests!