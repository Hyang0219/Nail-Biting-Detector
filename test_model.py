#!/usr/bin/env python3
import os
import sys
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import random
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def load_model(model_path):
    """Load the trained model."""
    try:
        model = tf.keras.models.load_model(model_path)
        logger.info(f"Loaded model from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None

def preprocess_image(image_path, target_size=(224, 224)):
    """Preprocess an image for model prediction."""
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            logger.error(f"Failed to load image from {image_path}")
            return None
            
        # Resize
        img = cv2.resize(img, target_size)
        
        # Convert from BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize
        img = img.astype(np.float32) / 255.0
        
        return img
    except Exception as e:
        logger.error(f"Error preprocessing image {image_path}: {e}")
        return None

def predict(model, image):
    """Make a prediction on a preprocessed image."""
    # Add batch dimension
    img_batch = np.expand_dims(image, 0)
    
    # Make prediction
    prediction = model.predict(img_batch, verbose=0)
    return prediction[0][0]  # Extract the first value from the batch

def display_prediction(image, prediction, actual_class, image_path, threshold=0.35):
    """Display the image with prediction results."""
    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    
    # Determine prediction class and color using the adjusted threshold
    predicted_class = "Nail Biting" if prediction >= threshold else "Not Nail Biting"
    color = "green" if (predicted_class == "Nail Biting" and actual_class == "nail_biting") or \
                       (predicted_class == "Not Nail Biting" and actual_class == "non_nail_biting") else "red"
    
    # Add prediction text
    plt.title(f"Prediction: {predicted_class} ({prediction:.3f})\nActual: {actual_class}\nThreshold: {threshold}", color=color)
    plt.axis('off')
    
    # Save the figure
    output_dir = "prediction_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get filename from path
    filename = os.path.basename(image_path)
    output_path = os.path.join(output_dir, f"pred_{filename}")
    plt.savefig(output_path)
    logger.info(f"Saved prediction visualization to {output_path}")
    
    # Close the figure to free memory
    plt.close()

def main():
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Test nail-biting detection model')
    parser.add_argument('--threshold', type=float, default=0.35,
                        help='Prediction threshold (default: 0.35)')
    parser.add_argument('--samples', type=int, default=5,
                        help='Number of samples to test from each category (default: 5)')
    args = parser.parse_args()
    
    # Find the best model
    models_dir = "models"
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.keras')]
    
    # Try to find the best model by accuracy in filename
    best_model = None
    best_acc = 0.0
    
    for model_file in model_files:
        try:
            # Extract accuracy from filename (e.g., mobilenet_model_20250301-230908_03_0.688.keras)
            if '_' in model_file and model_file.split('_')[-1].startswith('0.'):
                acc = float(model_file.split('_')[-1].split('.keras')[0])
                if acc > best_acc:
                    best_acc = acc
                    best_model = model_file
        except:
            continue
    
    if best_model:
        model_path = os.path.join(models_dir, best_model)
        logger.info(f"Using best model: {best_model} (accuracy: {best_acc})")
    else:
        # Fallback to a specific model
        model_path = os.path.join(models_dir, 'mobilenet_model_20250301-230908_03_0.688.keras')
        logger.info(f"Using fallback model: {model_path}")
    
    # Load the model
    model = load_model(model_path)
    if model is None:
        logger.error("Failed to load model. Exiting.")
        sys.exit(1)
    
    # Get sample images from each category
    data_dir = Path("data/processed")
    categories = ["nail_biting", "non_nail_biting"]
    
    # Print model summary
    model.summary()
    
    # Collect all predictions to analyze distribution
    all_predictions = []
    
    # Track metrics
    metrics = {
        "nail_biting": {"correct": 0, "total": 0},
        "non_nail_biting": {"correct": 0, "total": 0}
    }
    
    for category in categories:
        category_dir = data_dir / category
        if not category_dir.exists():
            logger.warning(f"Category directory not found: {category_dir}")
            continue
        
        # Get list of image files
        image_files = list(category_dir.glob('*.jpg')) + list(category_dir.glob('*.jpeg')) + list(category_dir.glob('*.png'))
        
        if not image_files:
            logger.warning(f"No images found in {category_dir}")
            continue
        
        # Select random samples
        samples = random.sample(image_files, min(args.samples, len(image_files)))
        
        logger.info(f"Testing {len(samples)} samples from {category}")
        
        for img_path in samples:
            # Preprocess image
            img = preprocess_image(img_path)
            if img is None:
                continue
            
            # Make prediction
            probability = predict(model, img)
            all_predictions.append((probability, category))
            
            # Display prediction
            display_prediction(img, probability, category, img_path, threshold=args.threshold)
            
            # Log result and update metrics
            predicted_class = "nail_biting" if probability >= args.threshold else "non_nail_biting"
            is_correct = predicted_class == category
            result_str = "✓" if is_correct else "✗"
            logger.info(f"{result_str} {img_path.name}: Predicted {predicted_class} ({probability:.3f}), Actual: {category}")
            
            # Update metrics
            metrics[category]["total"] += 1
            if is_correct:
                metrics[category]["correct"] += 1
    
    # Calculate and display metrics
    logger.info("\n===== Test Results =====")
    
    # Calculate accuracy for each class
    nail_biting_acc = metrics["nail_biting"]["correct"] / max(1, metrics["nail_biting"]["total"])
    non_nail_biting_acc = metrics["non_nail_biting"]["correct"] / max(1, metrics["non_nail_biting"]["total"])
    overall_acc = (metrics["nail_biting"]["correct"] + metrics["non_nail_biting"]["correct"]) / \
                 (metrics["nail_biting"]["total"] + metrics["non_nail_biting"]["total"])
    
    logger.info(f"Nail Biting Accuracy: {nail_biting_acc:.2f} ({metrics['nail_biting']['correct']}/{metrics['nail_biting']['total']})")
    logger.info(f"Non-Nail Biting Accuracy: {non_nail_biting_acc:.2f} ({metrics['non_nail_biting']['correct']}/{metrics['non_nail_biting']['total']})")
    logger.info(f"Overall Accuracy: {overall_acc:.2f}")
    
    # Analyze prediction distribution
    nail_biting_preds = [p for p, c in all_predictions if c == "nail_biting"]
    non_nail_biting_preds = [p for p, c in all_predictions if c == "non_nail_biting"]
    
    if nail_biting_preds:
        logger.info(f"Nail Biting predictions - Min: {min(nail_biting_preds):.3f}, Max: {max(nail_biting_preds):.3f}, Avg: {sum(nail_biting_preds)/len(nail_biting_preds):.3f}")
    if non_nail_biting_preds:
        logger.info(f"Non-Nail Biting predictions - Min: {min(non_nail_biting_preds):.3f}, Max: {max(non_nail_biting_preds):.3f}, Avg: {sum(non_nail_biting_preds)/len(non_nail_biting_preds):.3f}")
    
    logger.info("Testing complete! Check the 'prediction_results' directory for visualizations.")

if __name__ == "__main__":
    main() 