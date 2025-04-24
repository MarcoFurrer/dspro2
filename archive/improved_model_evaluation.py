#!/usr/bin/env python
"""
improved_model_evaluation.py

This script trains and evaluates the improved model architecture,
comparing it with the baseline model performance.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
import datetime

from src.EfficientCategoricalModel import EfficientCategoricalModel
from src.models.ImprovedModel import model as improved_model
from src.optimizers.ImprovedAdam import optimizer as improved_optimizer
from src.models.Base import model as base_model
from src.optimizers.Adam import optimizer as base_optimizer

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

def train_and_evaluate_model(model, optimizer, name, epochs=10):
    """Train and evaluate a model using the EfficientCategoricalModel framework"""
    print(f"\n{'='*50}")
    print(f"Training {name} model...")
    print(f"{'='*50}")
    
    # Paths to data
    data_path = "data/train.parquet"
    val_path = "data/validation.parquet"
    meta_path = "data/features.json"
    meta_model = "data/meta_model.parquet"
    
    # Create log directory with timestamp for TensorBoard
    log_dir = f"logs/fit/{name}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    os.makedirs(log_dir, exist_ok=True)
    
    # Create TensorBoard callback
    tensorboard_callback = TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True,
        update_freq='epoch'
    )
    
    # Other callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=0.00001,
        verbose=1
    )
    
    # Initialize model
    efficient_model = EfficientCategoricalModel(
        data_path_train=data_path,
        data_path_val=val_path,
        data_path_metadata=meta_path,
        data_path_meta_model=meta_model,
        batch_size=64,
        model=model,
        optimizer=optimizer
    )
    
    # Train with more epochs for better convergence
    model, history = efficient_model.train(
        epochs=epochs,
        callbacks=[tensorboard_callback, early_stopping, reduce_lr]
    )
    
    # Validate model
    validation_results = efficient_model.validate_model()
    
    # Evaluate performance
    metrics = efficient_model.performance_eval()
    
    # Export model
    model_path = efficient_model.export_model(model_name=name)
    
    print(f"\nModel saved to: {model_path}")
    print(f"TensorBoard logs saved to: {log_dir}")
    
    return {
        "model": model,
        "history": history,
        "metrics": metrics,
        "log_dir": log_dir,
        "model_path": model_path
    }

def compare_models(results):
    """Compare results between different models"""
    print("\n" + "="*50)
    print("MODEL COMPARISON")
    print("="*50)
    
    # Create comparison table
    comparison_data = []
    for name, result in results.items():
        metrics = result["metrics"]
        comparison_data.append([
            name,
            metrics.get("mean_correlation", 0),
            metrics.get("sharpe", 0),
            metrics.get("mean_mmc", 0)
        ])
    
    # Create a DataFrame for easier display
    comparison_df = pd.DataFrame(
        comparison_data,
        columns=["Model", "Mean Correlation", "Sharpe", "Mean MMC"]
    )
    
    print(comparison_df)
    
    # Create bar plots for visual comparison
    plt.figure(figsize=(12, 8))
    
    # Plot mean correlation
    plt.subplot(3, 1, 1)
    plt.bar(comparison_df["Model"], comparison_df["Mean Correlation"])
    plt.title("Mean Correlation by Model")
    plt.ylabel("Mean Correlation")
    
    # Plot Sharpe ratio
    plt.subplot(3, 1, 2)
    plt.bar(comparison_df["Model"], comparison_df["Sharpe"])
    plt.title("Sharpe Ratio by Model")
    plt.ylabel("Sharpe")
    
    # Plot MMC
    plt.subplot(3, 1, 3)
    plt.bar(comparison_df["Model"], comparison_df["Mean MMC"])
    plt.title("Mean Meta Model Contribution by Model")
    plt.ylabel("Mean MMC")
    
    plt.tight_layout()
    plt.savefig("model_comparison.png")
    print("\nComparison visualization saved to: model_comparison.png")
    
    return comparison_df

def main():
    try:
        # Dictionary to store results
        results = {}
        
        # Train and evaluate baseline model
        baseline_results = train_and_evaluate_model(
            model=base_model,
            optimizer=base_optimizer,
            name="Baseline",
            epochs=10
        )
        results["Baseline"] = baseline_results
        
        # Train and evaluate improved model
        improved_results = train_and_evaluate_model(
            model=improved_model,
            optimizer=improved_optimizer,
            name="Improved",
            epochs=15  # Give improved model a few more epochs due to complexity
        )
        results["Improved"] = improved_results
        
        # Compare model performance
        comparison = compare_models(results)
        
        print("\nTraining and evaluation complete!")
        print("Run TensorBoard to visualize training metrics:")
        print(f"tensorboard --logdir=logs/fit")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()