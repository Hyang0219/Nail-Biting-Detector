import pytest
import os
import json
import tensorflow as tf
import numpy as np
from detection.model_trainer import ModelTrainer

@pytest.fixture
def temp_data_dir(tmp_path):
    """Create a temporary data directory with mock data."""
    # Create directory structure
    data_dir = tmp_path / "data"
    processed_dir = data_dir / "processed"
    for category in ['nail_biting', 'non_nail_biting']:
        (processed_dir / category).mkdir(parents=True)
    
    # Create mock images
    for category in ['nail_biting', 'non_nail_biting']:
        for i in range(5):
            img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            img_path = processed_dir / category / f"test_image_{i}.jpg"
            tf.keras.preprocessing.image.save_img(str(img_path), img)
    
    # Create metadata file
    metadata = {
        'images': {},
        'stats': {'nail_biting': 5, 'non_nail_biting': 5}
    }
    
    # Add entries to metadata
    for category in ['nail_biting', 'non_nail_biting']:
        for i in range(5):
            img_hash = f"test_hash_{category}_{i}"
            img_path = str(processed_dir / category / f"test_image_{i}.jpg")
            metadata['images'][img_hash] = {
                'category': category,
                'processed': True,
                'processed_path': img_path
            }
    
    metadata_file = data_dir / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f)
    
    return str(data_dir)

@pytest.fixture
def temp_model_dir(tmp_path):
    """Create a temporary model directory."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    return str(model_dir)

@pytest.fixture
def trainer(temp_data_dir, temp_model_dir):
    """Create a ModelTrainer instance with temporary directories."""
    return ModelTrainer(data_dir=temp_data_dir, model_dir=temp_model_dir)

class TestModelTrainer:
    def test_initialization(self, trainer, temp_data_dir, temp_model_dir):
        """Test if ModelTrainer initializes correctly."""
        assert trainer.data_dir == temp_data_dir
        assert trainer.model_dir == temp_model_dir
        assert trainer.input_size == (224, 224)
        assert trainer.batch_size == 32
        assert trainer.metadata is not None
    
    def test_prepare_dataset(self, trainer):
        """Test dataset preparation."""
        train_ds, val_ds = trainer.prepare_dataset(validation_split=0.2)
        
        # Check if datasets are TensorFlow datasets
        assert isinstance(train_ds, tf.data.Dataset)
        assert isinstance(val_ds, tf.data.Dataset)
        
        # Check shapes
        for images, labels in train_ds.take(1):
            assert images.shape[1:] == (*trainer.input_size, 3)
            assert labels.shape[1:] == ()
    
    def test_build_model(self, trainer):
        """Test model building."""
        model = trainer.build_model()
        
        # Check model type and structure
        assert isinstance(model, tf.keras.Model)
        assert len(model.layers) > 0
        
        # Check input shape
        assert model.input_shape[1:] == (*trainer.input_size, 3)
        
        # Check output shape
        assert model.output_shape[1:] == (1,)
        
        # Check if model is compiled
        assert model.optimizer is not None
        assert model.loss is not None
    
    def test_train(self, trainer):
        """Test model training."""
        # Train for just 1 epoch to test functionality
        model, history = trainer.train(epochs=1)
        
        # Check if model and history are returned
        assert isinstance(model, tf.keras.Model)
        assert isinstance(history.history, dict)
        
        # Check if history contains expected metrics
        assert 'loss' in history.history
        assert 'accuracy' in history.history
        assert 'val_loss' in history.history
        assert 'val_accuracy' in history.history
        
        # Check if model was saved
        assert len(os.listdir(trainer.model_dir)) > 0
    
    def test_evaluate(self, trainer):
        """Test model evaluation."""
        # Prepare test dataset
        _, val_ds = trainer.prepare_dataset()
        
        # Train model for 1 epoch
        model, _ = trainer.train(epochs=1)
        
        # Evaluate model
        metrics = trainer.evaluate(model, val_ds)
        
        # Check metrics
        assert isinstance(metrics, dict)
        assert 'loss' in metrics
        assert 'accuracy' in metrics 