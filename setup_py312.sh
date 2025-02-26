#!/bin/bash

# Setup script for Nail Biting Detection application with Python 3.12
# This script will install Python 3.12 and all necessary dependencies

set -e  # Exit on any error

echo "===== Setting up Nail Biting Detection with Python 3.12 ====="

# Install required system packages
echo "Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install -y python3.12 python3.12-venv python3.12-dev

# Install Qt/PySide dependencies
echo "Installing Qt/PySide dependencies..."
sudo apt-get install -y libxcb-xinerama0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 \
    libxcb-render-util0 libxcb-xkb1 libxkbcommon-x11-0 libdouble-conversion3 libegl1 \
    libxcb-cursor0 libxkbcommon-dev xvfb

# Create and activate Python 3.12 virtual environment
echo "Creating Python 3.12 virtual environment..."
python3.12 -m venv venv-py312
source venv-py312/bin/activate

# Upgrade pip and install essential build tools
echo "Upgrading pip and installing build tools..."
pip install --upgrade pip setuptools wheel

# Install project dependencies
echo "Installing project dependencies..."
pip install -r requirements.txt

# Make run script executable
echo "Setting up run script..."
cat > run_app.sh << 'EOF'
#!/bin/bash

# Set Qt to use the offscreen platform plugin (for headless environments)
export QT_QPA_PLATFORM=offscreen

# Run the application
python src/main.py
EOF

chmod +x run_app.sh

echo "===== Setup complete! ====="
echo "To activate the environment, run: source venv-py312/bin/activate"
echo "To run the application, run: ./run_app.sh"
