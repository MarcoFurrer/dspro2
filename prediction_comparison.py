#!/usr/bin/env python
"""
prediction_comparison.py

This script loads a few random samples from the training data,
makes predictions using the EfficientCategoricalModel with the improved model,
and compares the actual values with the predictions.
"""

import os
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from src.EfficientCategoricalModel import EfficientCategoricalModel
from src.models.ImprovedModel import model as improved_model
from src.optimizers.ImprovedAdam import optimizer as improved_optimizer

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

def main():
    print("Loading training data...")
    
    # Paths to data
    data_path = "data/train.parquet"
    val_path = "data/validation.parquet"
    meta_path = "data/features.json"
    meta_model = "data/meta_model.parquet"
    
    # Load a small sample of the training data for display
    try:
        # Use pyarrow to read parquet more efficiently
        data = pd.read_parquet(data_path)
        
        # Get a subset of columns for display (era, data_type, target, and a few features)
        # Determine feature columns (they typically have feature in their name or start with f)
        feature_cols = [col for col in data.columns if col.startswith('feature')]
        if not feature_cols:
            feature_cols = [col for col in data.columns if col.startswith('f')]
        
        display_cols = ['era', 'data_type', 'target'] + feature_cols[:5]  # First 5 features
        
        # Sample 10 random rows
        sample_data = data.sample(10)
        
        print("\nRANDOM SAMPLE OF TRAINING DATA:")
        print("=" * 80)
        with pd.option_context('display.max_columns', None, 'display.width', 1000):
            print(sample_data[display_cols])
        
        print("\nTarget Distribution in Sample:")
        print(sample_data['target'].value_counts())
        
        # Initialize model with the improved architecture
        print("\nInitializing improved model for prediction demonstration...")
        efficient_model = EfficientCategoricalModel(
            data_path_train=data_path,
            data_path_val=val_path,
            data_path_metadata=meta_path,
            data_path_meta_model=meta_model,
            batch_size=64,
            model=improved_model,
            optimizer=improved_optimizer
        )
        
        # Get dataset info and feature set
        efficient_model.data_handler.get_dataset_info()
        feature_set = efficient_model.data_handler.feature_set
        
        # Create training/testing samples
        print("\nPreparing data for training...")
        
        # Use a larger sample for training the improved model
        sample_size = 5000  # Increased from 1000
        training_sample = data.sample(sample_size)
        X_train = training_sample[feature_set].values
        y_train = training_sample['target'].values
        
        # Training step - with mini batches
        print("\nTraining the improved model (this may take a few minutes)...")
        model = efficient_model.model_manager.model
        
        # Compile model directly with specific metrics
        model.compile(
            optimizer=efficient_model.model_manager.optimizer,
            loss='mae',
            metrics=['mae', 'mse']
        )
        
        # Train for more epochs to better learn patterns
        history = model.fit(
            X_train, y_train, 
            epochs=10,
            batch_size=64, 
            validation_split=0.2,
            verbose=1
        )
        
        # Make predictions on the sample data
        print("\nMaking predictions on sample data...")
        X_sample = sample_data[feature_set].values
        raw_predictions = model.predict(X_sample)
        
        # For categorical model, get the class with highest probability
        if raw_predictions.shape[1] > 1:
            # Convert from one-hot to scalar
            predicted_classes = np.argmax(raw_predictions, axis=1)
            # Map back to original values using the inverse mapping
            predictions = np.vectorize(efficient_model.model_manager.inverse_target_mapping.get)(predicted_classes)
        else:
            predictions = raw_predictions.flatten()
            
        sample_data['prediction'] = predictions
        
        # Display actual vs predicted values
        print("\nACTUAL VS PREDICTED VALUES:")
        print("=" * 80)
        result_df = sample_data[['era', 'target', 'prediction']].copy()
        result_df['error'] = np.abs(result_df['target'] - result_df['prediction'])
        with pd.option_context('display.precision', 4):
            print(result_df)
            
        # Calculate overall metrics
        mae = result_df['error'].mean()
        mse = np.mean(np.square(result_df['target'] - result_df['prediction']))
        print(f"\nMean Absolute Error on sample: {mae:.4f}")
        print(f"Mean Squared Error on sample: {mse:.4f}")
        
        # Plot actual vs predicted
        plt.figure(figsize=(10, 6))
        plt.scatter(result_df['target'], result_df['prediction'], alpha=0.6)
        plt.plot([0, 1], [0, 1], 'r--')  # Diagonal line where actual = predicted
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Improved Model: Actual vs Predicted Values')
        plt.tight_layout()
        plt.savefig('improved_predictions.png')
        print(f"\nPlot saved as 'improved_predictions.png'")
        
        # Plot training history
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'])
        plt.plot(history.history['val_mae'])
        plt.title('Model MAE')
        plt.ylabel('MAE')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        
        plt.tight_layout()
        plt.savefig('improved_training_history.png')
        print(f"Training history saved as 'improved_training_history.png'")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()