#!/bin/bash

# Set Qt to use the offscreen platform plugin
export QT_QPA_PLATFORM=offscreen

# Run the application
python src/main.py

# Clean up
kill $(pgrep Xvfb) 