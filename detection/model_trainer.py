import os
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from datetime import datetime
import logging

class ModelTrainer:
    def __init__(self, data_dir='data', model_dir='models'):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.input_size = (224, 224)
        self.batch_size = 32
        
        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def prepare_dataset(self, validation_split=0.2):
        """Prepare training and validation datasets with augmentation."""
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=validation_split
        )
        
        # Only rescaling for validation
        val_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=validation_split
        )
        
        # Load training data
        train_ds = train_datagen.flow_from_directory(
            os.path.join(self.data_dir, 'processed'),
            target_size=self.input_size,
            batch_size=self.batch_size,
            class_mode='binary',
            subset='training',
            shuffle=True
        )
        
        # Load validation data
        val_ds = val_datagen.flow_from_directory(
            os.path.join(self.data_dir, 'processed'),
            target_size=self.input_size,
            batch_size=self.batch_size,
            class_mode='binary',
            subset='validation',
            shuffle=True
        )
        
        return train_ds, val_ds
    
    def build_model(self):
        """Build and compile the model."""
        base_model = EfficientNetV2B0(
            include_top=False,
            weights='imagenet',
            input_shape=(*self.input_size, 3)
        )
        
        # Freeze the base model layers
        base_model.trainable = False
        
        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dropout(0.5),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'AUC', 'Precision', 'Recall']
        )
        
        return model
    
    def train(self, epochs=20, validation_split=0.2):
        """Train the model."""
        # Prepare datasets
        train_ds, val_ds = self.prepare_dataset(validation_split)
        
        # Calculate class weights
        total_samples = train_ds.n + val_ds.n
        neg_samples = len(os.listdir(os.path.join(self.data_dir, 'processed/non_nail_biting')))
        pos_samples = len(os.listdir(os.path.join(self.data_dir, 'processed/nail_biting')))
        
        class_weights = {
            0: total_samples / (2 * neg_samples),
            1: total_samples / (2 * pos_samples)
        }
        
        self.logger.info(f"Class weights: {class_weights}")
        
        # Build model
        model = self.build_model()
        
        # Setup callbacks
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(self.model_dir, 'best_model_{epoch:02d}_{val_accuracy:.3f}.keras'),
                monitor='val_accuracy',
                mode='max',
                save_best_only=True
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=3,
                min_lr=1e-6
            )
        ]
        
        # Train the model
        history = model.fit(
            train_ds,
            epochs=epochs,
            validation_data=val_ds,
            class_weight=class_weights,
            callbacks=callbacks
        )
        
        # Save final model
        final_model_path = os.path.join(
            self.model_dir,
            f'nail_biting_detector_{datetime.now().strftime("%Y%m%d_%H%M%S")}.keras'
        )
        model.save(final_model_path)
        self.logger.info(f"Saved final model to {final_model_path}")
        
        return model, history
    
    def evaluate(self, model, val_ds):
        """Evaluate the model on validation dataset."""
        metrics = model.evaluate(val_ds, return_dict=True)
        self.logger.info("Evaluation metrics:")
        for metric, value in metrics.items():
            self.logger.info(f"{metric}: {value:.4f}")
        return metrics 