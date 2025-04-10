import tensorflow as tf
import pandas as pd
import numpy as np
import os
import gc
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Input
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
import pyarrow.parquet as pq

# Set memory growth for GPU to avoid OOM errors
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    print("GPU memory growth set to True")
else:
    print("No GPU found, using CPU")

# Fix random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

class EfficientCategoricalModel:
    def __init__(self, data_path, output_path='models/exports', batch_size=64, model = None):
        self.data_path = data_path
        self.output_path = output_path
        self.batch_size = batch_size
        self.feature_count = None
        self.n_categories = 5  # Categories 0-4
        self.external_model = model  # Store the external model if provided
        os.makedirs(output_path, exist_ok=True)
        self.target_mapping = {0.0: 0, 0.25: 1, 0.5: 2, 0.75: 3, 1.0: 4}  # Map float targets to integers
        self.inverse_target_mapping = {0: 0.0, 1: 0.25, 2: 0.5, 3: 0.75, 4: 1.0}  # For converting back
        
    def _get_dataset_info(self):
        """Get basic information about the dataset"""
        # Open the parquet file
        parquet_file = pq.ParquetFile(self.data_path)
        
        # Read only the first row group for a quick sample
        df_sample = parquet_file.read_row_group(0).to_pandas()
        
        # Identify feature and target columns
        feature_cols = [col for col in df_sample.columns if col.startswith('feature_')]
        target_cols = [col for col in df_sample.columns if col.startswith('target')]
        
        total_rows = parquet_file.metadata.num_rows
        
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self.feature_count = len(feature_cols)
        self.total_rows = total_rows
        
        print(f"Dataset has {total_rows:,} rows")
        print(f"Found {len(feature_cols)} feature columns and {len(target_cols)} target columns")
        
    def _create_dataset_pipeline(self, is_training=True):
        """Create an efficient TF dataset pipeline that processes data in batches"""
        
        # Define dataset batch generator
        def generator():
            parquet_file = pq.ParquetFile(self.data_path)
            
            # Use smaller read batches to reduce memory pressure
            read_batch_size = min(10000, self.total_rows // 50)
            
            for batch in parquet_file.iter_batches(batch_size=read_batch_size):
                df_batch = batch.to_pandas()
                
                # Extract features and target with explicit alignment
                X_batch = df_batch[self.feature_cols].values
                X_batch = np.ascontiguousarray(X_batch, dtype=np.uint8)
                
                # Convert target values to uint8 integers (0-4) with explicit alignment
                y_float = df_batch[self.target_cols[0]].values
                y_batch = np.zeros(y_float.shape, dtype=np.uint8)
                
                # Map float values to integers
                for float_val, int_val in self.target_mapping.items():
                    y_batch[np.isclose(y_float, float_val)] = int_val
                
                y_batch = np.ascontiguousarray(y_batch)
                
                # Yield batches with explicit alignment
                for i in range(0, len(X_batch), self.batch_size):
                    end_idx = min(i + self.batch_size, len(X_batch))
                    # Create properly aligned copies
                    x = np.ascontiguousarray(X_batch[i:end_idx])
                    y = np.ascontiguousarray(y_batch[i:end_idx])
                    yield x, y
                
                # Free memory
                del df_batch, X_batch, y_batch, y_float
                gc.collect()
        
        # Define output shapes and types
        output_signature = (
            tf.TensorSpec(shape=(None, self.feature_count), dtype=tf.uint8),
            tf.TensorSpec(shape=(None,), dtype=tf.uint8)
        )
        
        # Create dataset
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=output_signature
        )
        
        # Configure dataset for performance - reduce shuffle buffer size
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        if is_training:
            # Use a smaller shuffle buffer to avoid memory issues
            dataset = dataset.shuffle(buffer_size=1000)
            
        return dataset
    
    def _create_default_model(self):
        """Create a simple but efficient model for categorical data"""
        model = Sequential([
            # Input layer
            Input(shape=(self.feature_count,), dtype=tf.uint8),
            
            # Convert to float32 for stability
            tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32)),
            
            # Hidden layers
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            # Output layer - 5 classes for our target values
            Dense(5, activation='softmax')
        ])
        
        # Use sparse categorical crossentropy since targets are integers
        optimizer = Adam(learning_rate=0.001)
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, validation_split=0.1, epochs=5):
        """Train the model using memory-efficient batch processing"""
        # Get dataset info
        self._get_dataset_info()
        
        # Create full dataset pipeline
        print("Creating dataset pipeline...")
        full_dataset = self._create_dataset_pipeline(is_training=True)
        
        # Calculate steps for validation data
        train_size = int(self.total_rows * (1 - validation_split))
        steps_per_epoch = train_size // self.batch_size
        validation_steps = max(1, (self.total_rows - train_size) // self.batch_size)
        
        # Manually split the dataset
        val_dataset = full_dataset.take(validation_steps)
        train_dataset = full_dataset.skip(validation_steps)
        
        if self.external_model is not None:
            print("Using provided external model...")
            model = self.external_model
        else:
            print("No model provided, creating default model...")
            model = self._create_default_model()

        model.summary()
        
        # Callbacks for training
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.0001, verbose=1),
        ]
        
        # Train with reduced steps to avoid memory issues
        print(f"Training model for {epochs} epochs with batch size {self.batch_size}...")
        history = model.fit(
            train_dataset,
            epochs=epochs,
            steps_per_epoch=min(steps_per_epoch, 200),  
            validation_data=val_dataset,
            validation_steps=min(validation_steps, 50),
            callbacks=callbacks,
            verbose=1
        )
        
        # Store model as instance variable
        self.model = model
        
        # Simple export
        self.export_model()
        
        return model, history

    def predict(self, X):
        """Convert categorical predictions back to original float values"""
        # Ensure proper alignment
        X = np.ascontiguousarray(X, dtype=np.uint8)
        
        # Get raw predictions
        raw_preds = self.model.predict(X)
        
        # Convert to class indices (0-4)
        class_indices = np.argmax(raw_preds, axis=1)
        
        # Map back to float values (0.0-1.0)
        float_predictions = np.vectorize(self.inverse_target_mapping.get)(class_indices)
        
        return float_predictions
        
    def export_model(self):
        """Simple model export"""
        model_path = os.path.join(self.output_path, 'model.h5')
        self.model.save(model_path)
        print(f"Model saved to: {model_path}")

# Run training if executed directly
if __name__=="__main__":
    try:
        print("Initializing efficient categorical model training...")
        
        # Initialize model with smaller batch size
        efficient_model = EfficientCategoricalModel(
            data_path='data/train.parquet',
            batch_size=64  # Small power-of-2 batch size for memory alignment
        )
        
        # Train with fewer epochs
        model, history = efficient_model.train(epochs=5)
        
        print("Training complete!")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
