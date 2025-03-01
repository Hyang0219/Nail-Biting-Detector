import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Large
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
        self.use_mixup = True  # Re-enable MixUp
        self.mixup_alpha = 0.2  # Alpha parameter for beta distribution in MixUp
        self.use_advanced_augmentation = True  # Flag to control augmentation complexity
        
        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Store dataset sizes for class weight calculation
        self.train_samples = 0
        self.val_samples = 0
        self.class_counts = {}
    
    def prepare_dataset(self, validation_split=0.2):
        """Prepare training and validation datasets with augmentation."""
        # Advanced data augmentation for training
        if self.use_advanced_augmentation:
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                # Advanced color augmentations
                brightness_range=[0.8, 1.2],
                channel_shift_range=30.0,  # Random shifts in RGB channels
                # HSV shifts can be simulated with preprocessing function
                preprocessing_function=self.color_augmentation,
                fill_mode='nearest',
                validation_split=validation_split
            )
        else:
            # Simplified augmentation for faster training
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=10,
                width_shift_range=0.1,
                height_shift_range=0.1,
                horizontal_flip=True,
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
        
        # Store dataset sizes before potentially wrapping with MixUp
        self.train_samples = train_ds.n
        self.val_samples = val_ds.n
        self.class_counts = {
            0: len(os.listdir(os.path.join(self.data_dir, 'processed/non_nail_biting'))),
            1: len(os.listdir(os.path.join(self.data_dir, 'processed/nail_biting')))
        }
        
        # If using MixUp, wrap the training dataset
        if self.use_mixup:
            train_ds_with_mixup = self.mixup_data_generator(train_ds)
            self.logger.info("Using MixUp data augmentation")
            return train_ds_with_mixup, val_ds
        
        return train_ds, val_ds
    
    def color_augmentation(self, img):
        """Apply additional color augmentations to an image."""
        # Convert to HSV for better color manipulations
        if len(img.shape) == 3 and img.shape[2] == 3:
            # Randomly decide whether to apply HSV shift
            if np.random.random() < 0.5:
                img = tf.image.rgb_to_hsv(img)
                
                # Random hue shift
                hue_shift = np.random.uniform(-0.1, 0.1)
                img = tf.image.adjust_hue(img, hue_shift)
                
                # Random saturation shift
                saturation_factor = np.random.uniform(0.7, 1.3)
                img = tf.image.adjust_saturation(img, saturation_factor)
                
                # Convert back to RGB
                img = tf.image.hsv_to_rgb(img)
            
            # Randomly adjust contrast
            if np.random.random() < 0.5:
                contrast_factor = np.random.uniform(0.7, 1.3)
                img = tf.image.adjust_contrast(img, contrast_factor)
                
            # Randomly simulate different lighting conditions
            if np.random.random() < 0.3:
                # Gaussian noise addition
                noise = np.random.normal(0, 0.02, img.shape)
                img = img + noise
                img = np.clip(img, 0, 1.0)  # Ensure values stay in valid range
        
        return img
    
    def mixup_data_generator(self, data_generator):
        """Wrapper for data generator that applies MixUp augmentation."""
        while True:
            # Get batch of images and labels
            X1, y1 = next(data_generator)
            
            # MixUp with probability of 0.5
            if np.random.random() < 0.5:
                # Get another batch to mix with
                X2, y2 = next(data_generator)
                
                # Generate mixing ratio from beta distribution
                lam = np.random.beta(self.mixup_alpha, self.mixup_alpha, size=X1.shape[0])
                
                # Reshape lam for proper broadcasting with images
                lam_reshaped = lam.reshape(X1.shape[0], 1, 1, 1)
                
                # Mix images
                X_mixed = X1 * lam_reshaped + X2 * (1 - lam_reshaped)
                
                # Mix labels - ensure proper shape for binary labels
                lam_for_labels = lam.reshape(X1.shape[0])
                
                # For binary classification, y is a 1D array
                if len(y1.shape) == 1:
                    y_mixed = y1 * lam_for_labels + y2 * (1 - lam_for_labels)
                else:
                    # For one-hot encoded labels (multi-class)
                    lam_for_onehot = lam.reshape(X1.shape[0], 1)
                    y_mixed = y1 * lam_for_onehot + y2 * (1 - lam_for_onehot)
                
                yield X_mixed, y_mixed
            else:
                yield X1, y1
    
    def build_model(self):
        """Build and compile the model using MobileNetV3Large."""
        base_model = MobileNetV3Large(
            include_top=False,
            weights='imagenet',
            input_shape=(*self.input_size, 3),
            alpha=1.0,  # Controls the width of the network (1.0 = 100%)
            minimalistic=False  # Use the full model, not the minimalistic version
        )
        
        # Partially unfreeze the base model - freeze early layers, unfreeze later layers
        base_model.trainable = True
        
        # Freeze the first 100 layers, unfreeze the rest for fine-tuning
        for layer in base_model.layers[:100]:
            layer.trainable = False
        
        # Log the number of trainable and non-trainable layers
        trainable_count = sum(1 for layer in base_model.layers if layer.trainable)
        non_trainable_count = sum(1 for layer in base_model.layers if not layer.trainable)
        self.logger.info(f"Base model has {trainable_count} trainable layers and {non_trainable_count} frozen layers")
        
        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dropout(0.5),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])
        
        # Use a lower learning rate for fine-tuning
        model.compile(
            optimizer=Adam(learning_rate=0.0001),  # Reduced learning rate for fine-tuning
            loss='binary_crossentropy',
            metrics=['accuracy', 'AUC', 'Precision', 'Recall']
        )
        
        return model
    
    def train(self, epochs=20, validation_split=0.2):
        """Train the model."""
        # Prepare datasets
        train_ds, val_ds = self.prepare_dataset(validation_split)
        
        # Calculate class weights using stored sample counts
        total_samples = self.train_samples + self.val_samples
        
        class_weights = {
            0: total_samples / (2 * self.class_counts[0]),
            1: total_samples / (2 * self.class_counts[1])
        }
        
        self.logger.info(f"Class weights: {class_weights}")
        
        # Build model
        model = self.build_model()
        
        # Create a timestamp for this training run
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        
        # Create logs directory if it doesn't exist
        logs_dir = 'logs'
        os.makedirs(logs_dir, exist_ok=True)
        
        # Setup callbacks with improved learning rate schedule
        callbacks = [
            # Model checkpoint to save the best model
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(self.model_dir, f'mobilenet_model_{timestamp}_{{epoch:02d}}_{{val_accuracy:.3f}}.keras'),
                monitor='val_accuracy',
                mode='max',
                save_best_only=True,
                verbose=1
            ),
            # Early stopping to prevent overfitting
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=7,  # Increased patience
                restore_best_weights=True,
                verbose=1
            ),
            # Learning rate scheduler with cosine decay
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,  # More gradual reduction
                patience=3,
                min_lr=1e-6,
                verbose=1
            ),
            # TensorBoard logging
            tf.keras.callbacks.TensorBoard(
                log_dir=os.path.join(logs_dir, f'run_{timestamp}'),
                histogram_freq=1,
                write_graph=True,
                update_freq='epoch'
            ),
            # CSV Logger for detailed metrics
            tf.keras.callbacks.CSVLogger(
                os.path.join(logs_dir, f'training_log_{timestamp}.csv'),
                separator=',',
                append=False
            ),
            # Custom callback to save training history plot
            tf.keras.callbacks.LambdaCallback(
                on_epoch_end=None,
                on_train_end=lambda logs: self.save_training_history_plot(model.history, timestamp)
            )
        ]
        
        # Log training parameters
        self.logger.info(f"Starting training with:")
        self.logger.info(f"- Epochs: {epochs}")
        self.logger.info(f"- Batch size: {self.batch_size}")
        self.logger.info(f"- Validation split: {validation_split}")
        self.logger.info(f"- MixUp enabled: {self.use_mixup}")
        self.logger.info(f"- Advanced augmentation: {self.use_advanced_augmentation}")
        self.logger.info(f"- Class weights: {class_weights}")
        self.logger.info(f"- Training samples: {self.train_samples}")
        self.logger.info(f"- Validation samples: {self.val_samples}")
        
        # Calculate steps per epoch to avoid infinite generator issue
        steps_per_epoch = self.train_samples // self.batch_size
        validation_steps = self.val_samples // self.batch_size
        
        # Ensure at least one step
        steps_per_epoch = max(1, steps_per_epoch)
        validation_steps = max(1, validation_steps)
        
        self.logger.info(f"- Steps per epoch: {steps_per_epoch}")
        self.logger.info(f"- Validation steps: {validation_steps}")
        
        # Train the model - don't use class weights with MixUp
        history = model.fit(
            train_ds,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_ds,
            validation_steps=validation_steps,
            class_weight=None if self.use_mixup else class_weights,  # Only use class weights if not using MixUp
            callbacks=callbacks,
            verbose=1
        )
        
        # Save final model
        final_model_path = os.path.join(
            self.model_dir,
            f'mobilenet_model_{timestamp}.keras'
        )
        model.save(final_model_path)
        self.logger.info(f"Saved final model to {final_model_path}")
        
        return model, history
    
    def save_training_history_plot(self, history, timestamp):
        """Save a plot of the training history."""
        import matplotlib.pyplot as plt
        
        # Create logs directory if it doesn't exist
        logs_dir = 'logs'
        os.makedirs(logs_dir, exist_ok=True)
        
        # Plot training & validation accuracy
        plt.figure(figsize=(12, 8))
        
        # Plot accuracy
        plt.subplot(2, 2, 1)
        if 'accuracy' in history.history:
            plt.plot(history.history['accuracy'])
            if 'val_accuracy' in history.history:
                plt.plot(history.history['val_accuracy'])
                plt.legend(['Train', 'Validation'], loc='lower right')
            else:
                plt.legend(['Train'], loc='lower right')
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        
        # Plot loss
        plt.subplot(2, 2, 2)
        if 'loss' in history.history:
            plt.plot(history.history['loss'])
            if 'val_loss' in history.history:
                plt.plot(history.history['val_loss'])
                plt.legend(['Train', 'Validation'], loc='upper right')
            else:
                plt.legend(['Train'], loc='upper right')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        
        # Plot AUC
        plt.subplot(2, 2, 3)
        legend_items = []
        if 'auc' in history.history:
            plt.plot(history.history['auc'])
            legend_items.append('Train')
        if 'val_auc' in history.history:
            plt.plot(history.history['val_auc'])
            legend_items.append('Validation')
        if legend_items:
            plt.legend(legend_items, loc='lower right')
        plt.title('Model AUC')
        plt.ylabel('AUC')
        plt.xlabel('Epoch')
        
        # Plot Precision and Recall
        plt.subplot(2, 2, 4)
        legend_items = []
        if 'precision' in history.history:
            plt.plot(history.history['precision'])
            legend_items.append('Precision')
        if 'recall' in history.history:
            plt.plot(history.history['recall'])
            legend_items.append('Recall')
        if 'val_precision' in history.history:
            plt.plot(history.history['val_precision'])
            legend_items.append('Val Precision')
        if 'val_recall' in history.history:
            plt.plot(history.history['val_recall'])
            legend_items.append('Val Recall')
        if legend_items:
            plt.legend(legend_items, loc='lower right')
        plt.title('Precision and Recall')
        plt.ylabel('Score')
        plt.xlabel('Epoch')
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = os.path.join(logs_dir, f'training_history_{timestamp}.png')
        plt.savefig(plot_path)
        plt.close()
        
        self.logger.info(f"Saved training history plot to {plot_path}")
    
    def evaluate(self, model, val_ds):
        """Evaluate the model on validation dataset."""
        metrics = model.evaluate(val_ds, return_dict=True)
        self.logger.info("Evaluation metrics:")
        for metric, value in metrics.items():
            self.logger.info(f"{metric}: {value:.4f}")
        return metrics 