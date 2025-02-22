# Nail-Biting Detection Application

This application uses computer vision to detect nail-biting behavior and alerts users in real-time. It uses your computer's webcam to monitor hand movements near the mouth and provides immediate feedback when nail-biting is detected.

Version 1.0
Use MediaPipe Hand Pose + Geometric Approach (Quick Solution):
* Instead of relying on the pre-trained classification model, the model uses MediaPipe's hand landmarks
* Calculate the distance between fingertips and mouth landmarks
* Check finger orientation relative to the mouth
* Add gesture recognition for common nail-biting poses
* This would be more reliable and faster than the current ML approach as an MVP

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

- Python 3.8 or higher
- Webcam
- The required Python packages are listed in `requirements.txt`

## Installation

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