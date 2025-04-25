import os
import numpy as np
import tensorflow as tf
from datetime import datetime
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

class ModelManager:
    def __init__(self, 
                feature_count=None,
                output_path='exports', 
                model=None,
                optimizer=None):
        
        self.feature_count = feature_count
        self.output_path = output_path
        self.model = model
        self.optimizer = optimizer
        
        # Target mapping dictionaries for categorical conversion
        self.target_mapping = {0.0: 0, 0.25: 1, 0.5: 2, 0.75: 3, 1.0: 4}  # Map float targets to integers
        self.inverse_target_mapping = {0: 0.0, 1: 0.25, 2: 0.5, 3: 0.75, 4: 1.0}  # For converting back
        
        # Create exports directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
    
    def train(self, train_dataset, val_dataset, steps_per_epoch, validation_steps, epochs=5, callbacks=None):
        """Train the model using memory-efficient batch processing"""
        if self.model is None:
            raise ValueError("No model provided for training")
        
        # Check if model is a factory function and create the actual model instance
        if callable(self.model) and not isinstance(self.model, tf.keras.Model):
            print(f"Creating model instance from factory function with input shape ({self.feature_count},)")
            self.model = self.model((self.feature_count,))
            
        print(f"Model summary:\n{self.model.summary()}")
        self.model.compile(optimizer=self.optimizer, loss='mae', metrics=['mae'])
        
        # Callbacks for training
        default_callbacks = [
            EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.0001, verbose=1),
        ]
        
        # Combine default callbacks with any provided callbacks
        if callbacks:
            all_callbacks = default_callbacks + callbacks
        else:
            all_callbacks = default_callbacks
        
        # Train with reduced steps to avoid memory issues
        print(f"Training model for {epochs} epochs...")
        history = self.model.fit(
            train_dataset,
            epochs=epochs,
            steps_per_epoch=min(steps_per_epoch, 200),  
            validation_data=val_dataset,
            validation_steps=min(validation_steps, 50),
            callbacks=all_callbacks,
            verbose=1
        )
        
        return self.model, history
    
    def predict(self, X):
        """Generate predictions using the trained model"""
        if self.model is None:
            raise ValueError("No trained model available for prediction")
            
        # Ensure proper alignment and data type
        X = np.ascontiguousarray(X, dtype=np.float32)  # Fixed to use float32 instead of uint8
        
        # Get raw predictions
        raw_preds = self.model.predict(X)
        
        if raw_preds.ndim > 1 and raw_preds.shape[1] > 1:
            # For multi-class (categorical) output
            class_indices = np.argmax(raw_preds, axis=1)
            # Map back to float values (0.0-1.0)
            predictions = np.vectorize(self.inverse_target_mapping.get)(class_indices)
        else:
            # For scalar output
            predictions = raw_preds.squeeze()
            
        return predictions
    
    def export_model(self, model_name=None):
        """Export the trained model to disk"""
        if self.model is None:
            raise ValueError("No model to export")
            
        if model_name is None:
            model_name = f'model{datetime.now().strftime("%Y%m%d_%H%M%S")}'
            
        model_path = os.path.join(self.output_path, f'{model_name}.keras')
        self.model.save(model_path)
        print(f"Model saved to: {model_path}")
        return model_path
    
    def load_model(self, model_path):
        """Load a trained model from disk"""
        self.model = tf.keras.models.load_model(model_path)
        return self.model