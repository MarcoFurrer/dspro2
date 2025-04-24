#!/usr/bin/env python
"""
ensemble_predict.py

This script loads the best models trained using k-fold cross-validation
and makes ensemble predictions on the validation or live data.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import argparse
import matplotlib.pyplot as plt
from src.EfficientCategoricalModel import EfficientCategoricalModel
from src.optimizers.ImprovedAdam import optimizer as improved_optimizer

def ensemble_predict(models, X, target_mapping=None):
    """Make predictions using an ensemble of models"""
    all_preds = []
    
    for model in models:
        raw_preds = model.predict(X, verbose=0)
        
        # For categorical model, get the class with highest probability
        if raw_preds.shape[1] > 1:
            # Convert to class indices
            pred_classes = np.argmax(raw_preds, axis=1)
            
            # Use provided mapping if available
            if target_mapping:
                pred_values = np.vectorize(target_mapping.get)(pred_classes)
            else:
                # Default mapping if not provided
                default_mapping = {0: 0.0, 1: 0.25, 2: 0.5, 3: 0.75, 4: 1.0}
                pred_values = np.vectorize(default_mapping.get)(pred_classes)
                
            all_preds.append(pred_values.reshape(-1, 1))
        else:
            # For regression models
            all_preds.append(raw_preds)
    
    # Average the predictions
    ensemble_preds = np.mean(all_preds, axis=0)
    
    return ensemble_preds.flatten()

def load_models(model_dir, fold_prefix="AdvancedModel_fold", n_folds=5):
    """Load all models from the specified directory matching the fold pattern"""
    models = []
    
    for i in range(n_folds):
        model_path = os.path.join(model_dir, f"{fold_prefix}{i}.keras")
        if os.path.exists(model_path):
            print(f"Loading model: {model_path}")
            model = tf.keras.models.load_model(model_path)
            models.append(model)
        else:
            print(f"Warning: Model not found at {model_path}")
    
    print(f"Loaded {len(models)} models for ensemble prediction")
    return models

def main():
    parser = argparse.ArgumentParser(description='Make predictions using ensemble of models')
    parser.add_argument('--data_path', type=str, default='data/validation.parquet',
                      help='Path to data for prediction (validation or live)')
    parser.add_argument('--meta_path', type=str, default='data/features.json',
                      help='Path to metadata JSON file')
    parser.add_argument('--model_dir', type=str, default='exports',
                      help='Directory containing the trained models')
    parser.add_argument('--fold_prefix', type=str, default='AdvancedModel_fold',
                      help='Prefix of the fold model filenames')
    parser.add_argument('--n_folds', type=int, default=5,
                      help='Number of folds (models) to use in ensemble')
    parser.add_argument('--output_path', type=str, default='ensemble_predictions.csv',
                      help='Path to save predictions')
    parser.add_argument('--visualize', action='store_true',
                      help='Create visualization of predictions')
    
    args = parser.parse_args()
    
    try:
        # Initialize an EfficientCategoricalModel to get feature set and other utilities
        print("Initializing EfficientCategoricalModel...")
        ecm = EfficientCategoricalModel(
            data_path_train=None,
            data_path_val=args.data_path,
            data_path_metadata=args.meta_path,
            batch_size=64
        )
        
        # Get feature info
        ecm.data_handler.get_dataset_info()
        feature_set = ecm.data_handler.feature_set
        
        # Load data
        print(f"Loading data from {args.data_path}...")
        data = pd.read_parquet(args.data_path)
        
        # Prepare features for prediction
        X = data[feature_set].values
        
        # Load models for ensemble
        models = load_models(args.model_dir, args.fold_prefix, args.n_folds)
        
        if len(models) == 0:
            raise ValueError("No models found for ensemble prediction")
        
        # Make ensemble predictions
        print("Making ensemble predictions...")
        target_mapping = ecm.model_manager.inverse_target_mapping
        predictions = ensemble_predict(models, X, target_mapping)
        
        # Add predictions to dataframe
        data['ensemble_prediction'] = predictions
        
        # Calculate metrics if target is available
        if 'target' in data.columns:
            print("\nPrediction Metrics:")
            # Calculate MAE
            mae = np.abs(data['target'] - data['ensemble_prediction']).mean()
            print(f"Mean Absolute Error: {mae:.4f}")
            
            # Calculate MSE
            mse = np.mean(np.square(data['target'] - data['ensemble_prediction']))
            print(f"Mean Squared Error: {mse:.4f}")
            
            # Visualize if requested
            if args.visualize:
                plt.figure(figsize=(10, 6))
                plt.scatter(data['target'], data['ensemble_prediction'], alpha=0.5)
                plt.plot([0, 1], [0, 1], 'r--')  # Diagonal line
                plt.xlabel('Actual Values')
                plt.ylabel('Ensemble Predictions')
                plt.title('Ensemble Model: Actual vs Predicted Values')
                plt.tight_layout()
                
                # Save plot
                plt.savefig('ensemble_predictions.png')
                print("Visualization saved to: ensemble_predictions.png")
        
        # Save predictions
        print(f"Saving predictions to {args.output_path}...")
        # Extract relevant columns for submission
        if 'id' in data.columns:
            output_df = data[['id', 'ensemble_prediction']]
            output_df.to_csv(args.output_path, index=False)
            print(f"Predictions saved to {args.output_path}")
        else:
            data.to_csv(args.output_path, index=True)
            print(f"Full data with predictions saved to {args.output_path}")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()