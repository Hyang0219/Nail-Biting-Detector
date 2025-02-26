import sys
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    logging.info("Testing PySide6 import")
    logging.info(f"Python version: {sys.version}")
    
    try:
        logging.info("Attempting to import PySide6")
        from PySide6.QtCore import qVersion
        from PySide6.QtWidgets import QApplication
        
        logging.info(f"PySide6 imported successfully!")
        logging.info(f"Qt version: {qVersion()}")
        
        # Try to create a QApplication instance
        logging.info("Attempting to create QApplication instance with offscreen platform")
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        app = QApplication.instance() or QApplication([])
        logging.info("QApplication instance created successfully")
        
        # Check if we can create widgets
        from PySide6.QtWidgets import QLabel
        label = QLabel("Test")
        logging.info("Successfully created a QLabel widget")
        
    except ImportError as e:
        logging.error(f"Error importing PySide6: {e}")
    except Exception as e:
        logging.error(f"Error initializing PySide6: {e}")
        import traceback
        logging.error(traceback.format_exc())
    
    logging.info("PySide6 test completed")

if __name__ == "__main__":
    main() 