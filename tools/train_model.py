#!/usr/bin/env python3

import os
import sys
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add src directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detection.model_trainer import ModelTrainer

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


def plot_training_history(history, save_path):
    """Plot training history metrics."""
    # Set style
    plt.style.use('seaborn-v0_8')
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Train nail-biting detection model')
    parser.add_argument('--epochs', type=int, default=50,
                      help='Number of epochs to train (default: 50)')
    parser.add_argument('--batch-size', type=int, default=32,
                      help='Batch size for training (default: 32)')
    parser.add_argument('--validation-split', type=float, default=0.2,
                      help='Fraction of data to use for validation (default: 0.2)')
    parser.add_argument('--data-dir', type=str, default='data',
                      help='Directory containing the dataset (default: data)')
    parser.add_argument('--model-dir', type=str, default='models',
                      help='Directory to save models (default: models)')
    
    args = parser.parse_args()
    
    # Create directories if they don't exist
    os.makedirs('logs', exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Initialize trainer
    trainer = ModelTrainer(data_dir=args.data_dir, model_dir=args.model_dir)
    
    # Start training
    print(f"\nStarting training with parameters:")
    print(f"- Epochs: {args.epochs}")
    print(f"- Batch size: {args.batch_size}")
    print(f"- Validation split: {args.validation_split}")
    print(f"- Data directory: {args.data_dir}")
    print(f"- Model directory: {args.model_dir}")
    
    model, history = trainer.train(
        epochs=args.epochs,
        validation_split=args.validation_split
    )
    
    # Plot training history
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join('logs', f'training_history_{timestamp}.png')
    plot_training_history(history, plot_path)
    print(f"\nTraining history plot saved to {plot_path}")
    
    # Print final metrics
    print("\nTraining completed. Final metrics:")
    for metric, values in history.history.items():
        print(f"{metric}: {values[-1]:.4f}")

if __name__ == "__main__":
    main() 