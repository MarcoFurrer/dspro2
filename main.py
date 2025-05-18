#!/usr/bin/env python
# filepath: /Users/marcofurrer/Documents/github/dspro2/new_main.py
"""
Consolidated training script for various model architectures.
This script integrates functionality from:
- train_advanced_model.py
- train_best_model.py
- train_correlation_model.py
into a single entry point.
"""

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.losses import MeanAbsoluteError
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
import json
import argparse
from sklearn.model_selection import KFold
from tensorboard.plugins.hparams import api as hp

# Import components from existing project
from src.data_handler import DataHandler
from src.model_evaluator import ModelEvaluator
from src.EfficientCategoricalModel import EfficientCategoricalModel

# Import model architectures
from src.models.Deep import model as deep_model
from src.models.CorrelationModel import model as correlation_model, advanced_model, create_correlation_model
from src.models.ImprovedModel import model as improved_model
from src.models.BestModel import model as best_model
from src.models.Residual import model as residual_model

# Import optimizers
from src.optimizers.Adam import optimizer as adam_optimizer
from src.optimizers.ImprovedAdam import optimizer as improved_optimizer

# Import losses
try:
    from src.losses.DistributionAwareLoss import get_best_loss, DistributionAwareLoss
except ImportError:
    print("Warning: DistributionAwareLoss not found, some functionality may be limited")

# Set seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Try to enable memory growth to prevent TF from allocating all GPU memory at once
try:
    physical_devices = tf.config.list_physical_devices('GPU')
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    print(f"Found {len(physical_devices)} GPU(s), memory growth enabled")
except:
    print("No GPU available or memory growth setting failed")
    pass

#--- Custom Loss Functions ---#

def correlation_loss(y_true, y_pred):
    """
    Custom loss function that maximizes correlation between predictions and targets
    while maintaining accuracy with MAE.
    """
    # Mean centering
    y_true_centered = y_true - tf.reduce_mean(y_true, axis=0)
    y_pred_centered = y_pred - tf.reduce_mean(y_pred, axis=0)
    
    # Calculate correlation
    numerator = tf.reduce_sum(y_true_centered * y_pred_centered, axis=0)
    denominator = tf.sqrt(tf.reduce_sum(tf.square(y_true_centered), axis=0) * 
                         tf.reduce_sum(tf.square(y_pred_centered), axis=0) + tf.keras.backend.epsilon())
    correlation = numerator / denominator
    
    # We want to maximize correlation, so we minimize the negative
    corr_loss = -correlation
    
    # Add MAE loss for general prediction accuracy
    mae_loss = tf.reduce_mean(tf.abs(y_true - y_pred))
    
    # Return weighted combination
    return mae_loss + 0.5 * corr_loss


class CorrelationAwareLoss(tf.keras.losses.Loss):
    """
    Enhanced loss function that encourages learning correlations and 
    penalizes just mimicking the distribution.
    """
    def __init__(self, distribution_penalty=0.2, name="correlation_aware_loss"):
        super().__init__(name=name)
        self.distribution_penalty = distribution_penalty
        self.mae = MeanAbsoluteError()
        
    def call(self, y_true, y_pred):
        # Base MAE loss
        mae_loss = self.mae(y_true, y_pred)
        
        # Calculate batch statistics
        true_mean = tf.reduce_mean(y_true)
        pred_mean = tf.reduce_mean(y_pred)
        
        true_std = tf.math.reduce_std(y_true)
        pred_std = tf.math.reduce_std(y_pred)
        
        # Pearson correlation coefficient
        y_true_centered = y_true - true_mean
        y_pred_centered = y_pred - pred_mean
        
        numerator = tf.reduce_sum(y_true_centered * y_pred_centered)
        denominator = tf.sqrt(tf.reduce_sum(tf.square(y_true_centered)) * 
                             tf.reduce_sum(tf.square(y_pred_centered)) + 1e-8)
        
        correlation = numerator / denominator
        
        # Distribution matching penalty
        distribution_diff = tf.abs(true_mean - pred_mean) + tf.abs(true_std - pred_std)
        
        # Total loss: MAE + distribution penalty - correlation reward
        # We subtract correlation because higher correlation is better
        total_loss = mae_loss + self.distribution_penalty * distribution_diff - (1.0 - correlation)
        
        return total_loss


#--- Helper Functions ---#

def create_callbacks(model_name, config=None, log_dir=None, checkpoint_dir=None):
    """Create training callbacks with appropriate settings"""
    # Create timestamp for unique identification
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Create structured log directory for better organization
    if log_dir is None:
        log_dir = f"logs/experiments/{model_name}/{timestamp}"
    
    # Create directories if they don't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create metrics directory for CSV logs
    metrics_dir = f"logs/metrics"
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Create checkpoint directory if not provided
    if checkpoint_dir is None:
        checkpoint_dir = f"models/checkpoints/{model_name}"
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Define callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, f"{model_name}_{timestamp}_best.keras"),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            write_images=False,  # Set to True to save visualizations of model weights
            update_freq='epoch',
            profile_batch=0  # No profiling by default
        ),
        # Add CSV Logger for easy metric access
        CSVLogger(
            filename=f"{metrics_dir}/{model_name}_{timestamp}_metrics.csv",
            separator=',',
            append=False
        )
    ]
    
    # Record hyperparameters if config is provided
    if config:
        # Create a structured config dictionary if it's not already
        if not isinstance(config, dict):
            config = {}
        
        # Ensure essential config items exist
        config_complete = {
            'model': model_name,
            'timestamp': timestamp,
            'optimizer': config.get('optimizer', 'adam'),
            'learning_rate': config.get('learning_rate', 0.001),
            'batch_size': config.get('batch_size', 512),
            'epochs': config.get('epochs', 100),
            'feature_subset': config.get('feature_subset', 'all'),
            'loss_fn': config.get('loss_fn', 'mae'),
            'dropout_rate': config.get('dropout_rate', 0.3),
            'layers': config.get('layers', []),
            'neurons': config.get('neurons', []),
            'validation_split': config.get('validation_split', 0.15),
            'early_stopping_patience': config.get('early_stopping_patience', 15)
        }
        
        # Update config with all values
        for k, v in config_complete.items():
            if k not in config:
                config[k] = v
        
        # Define the hyperparameters to track with appropriate types
        hparams = {
            'model': hp.HParam('model', hp.Discrete([model_name])),
            'optimizer': hp.HParam('optimizer', hp.Discrete(['adam', 'sgd', 'rmsprop', 'adagrad', 'adadelta', 'nadam', 'adamax'])),
            'learning_rate': hp.HParam('learning_rate', hp.RealInterval(1e-6, 1.0)),
            'batch_size': hp.HParam('batch_size', hp.IntInterval(8, 4096)),
            'feature_subset': hp.HParam('feature_subset', hp.Discrete(['small', 'medium', 'all'])),
            'loss_fn': hp.HParam('loss_fn', hp.Discrete(['mae', 'mse', 'correlation', 'correlation_aware'])),
            'dropout_rate': hp.HParam('dropout_rate', hp.RealInterval(0.0, 0.9)),
        }
        
        # Log the hyperparameters
        with tf.summary.create_file_writer(log_dir).as_default():
            # Define comprehensive metrics to track
            hp.hparams_config(
                hparams=list(hparams.values()),
                metrics=[
                    hp.Metric('epoch_loss', display_name='Training Loss'),
                    hp.Metric('epoch_val_loss', display_name='Validation Loss'),
                    hp.Metric('epoch_mae', display_name='MAE'),
                    hp.Metric('epoch_val_mae', display_name='Validation MAE'),
                    hp.Metric('epoch_mse', display_name='MSE'),
                    hp.Metric('epoch_val_mse', display_name='Validation MSE'),
                    hp.Metric('final_correlation', display_name='Correlation Score'),
                    hp.Metric('final_val_loss', display_name='Final Validation Loss'),
                    hp.Metric('final_era_correlation', display_name='Era Correlation'),
                ],
            )
            
            # Record the hparams used in this trial
            hp_values = {
                'model': model_name,
                'optimizer': config['optimizer'],
                'learning_rate': config['learning_rate'],
                'batch_size': config['batch_size'],
                'feature_subset': config['feature_subset'],
                'loss_fn': config['loss_fn'],
                'dropout_rate': config['dropout_rate']
            }
            hp.hparams(hp_values)
        
        # Save detailed configuration as JSON for easy reference
        with open(f"{log_dir}/config.json", 'w') as f:
            json.dump(config, f, indent=2)
            
        # Also save a summary file with key experiment parameters
        with open(f"{log_dir}/experiment.txt", 'w') as f:
            f.write(f"Experiment: {model_name}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Features: {config['feature_subset']}\n")
            f.write(f"Loss Function: {config['loss_fn']}\n")
            f.write(f"Optimizer: {config['optimizer']} (lr={config['learning_rate']})\n")
            f.write(f"Batch Size: {config['batch_size']}\n")
            f.write(f"Max Epochs: {config['epochs']}\n")
    
    return callbacks, log_dir, checkpoint_dir


def plot_training_history(history, model_name):
    """Plot and save training history"""
    plt.figure(figsize=(15, 6))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Loss During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # MAE plot if available
    plt.subplot(1, 2, 2)
    if 'mae' in history.history and 'val_mae' in history.history:
        plt.plot(history.history['mae'], label='Train')
        plt.plot(history.history['val_mae'], label='Validation')
        title = 'MAE During Training'
    elif 'mse' in history.history and 'val_mse' in history.history:
        plt.plot(history.history['mse'], label='Train')
        plt.plot(history.history['val_mse'], label='Validation')
        title = 'MSE During Training'
    else:
        # Fallback to just showing train/val loss again
        plt.plot(history.history['loss'], label='Train')
        plt.plot(history.history['val_loss'], label='Validation')
        title = 'Loss During Training'
    
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(title.split()[0])  # MAE or MSE or Loss
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{model_name}_training_history.png')
    print(f"Training history plot saved to {model_name}_training_history.png")
    plt.close()


def evaluate_on_validation(model, data_handler, feature_set):
    """Evaluate model on validation data"""
    # Load validation data
    val_data = data_handler.load_validation_data()
    
    # Prepare validation data
    X_val = val_data[feature_set].values
    y_val = val_data["target"].values
    
    # Evaluate
    val_metrics = model.evaluate(X_val, y_val, verbose=1)
    
    # Handle different metrics based on model compile settings
    metric_names = ["loss"]
    if len(model.metrics_names) > 1:
        metric_names.extend(model.metrics_names[1:])
    
    # Print metrics
    for i, metric_name in enumerate(metric_names):
        if i < len(val_metrics):
            print(f"Validation {metric_name}: {val_metrics[i]:.4f}")
    
    # Make predictions
    y_pred = model.predict(X_val)
    
    # Calculate correlation
    correlation = np.corrcoef(y_val, y_pred.flatten())[0, 1]
    print(f"Correlation between predictions and targets: {correlation:.4f}")
    
    # Calculate era-wise correlation if available
    if "era" in val_data.columns:
        era_scores = []
        unique_eras = val_data["era"].unique()
        
        for era in unique_eras:
            era_mask = val_data["era"] == era
            if era_mask.sum() > 5:  # Ensure we have enough samples
                era_y = y_val[era_mask]
                era_pred = y_pred.flatten()[era_mask]
                era_corr = np.corrcoef(era_y, era_pred)[0, 1]
                era_scores.append(era_corr)
        
        mean_era_corr = np.mean(era_scores)
        std_era_corr = np.std(era_scores)
        print(f"Mean era-wise correlation: {mean_era_corr:.4f} (±{std_era_corr:.4f})")
    
    return val_metrics


def log_final_metrics(log_dir, metrics, model_name, config=None):
    """
    Log final evaluation metrics to TensorBoard and save as files
    
    Args:
        log_dir: TensorBoard log directory
        metrics: Dictionary of metrics to log
        model_name: Name of the model
        config: Configuration used for training
    """
    # Ensure metrics is a dictionary
    if not isinstance(metrics, dict):
        if isinstance(metrics, (list, tuple)) and len(metrics) >= 1:
            # Convert from list of metrics to dictionary
            metric_names = ["loss"]
            if len(metrics) > 1:
                metric_names.extend(["mae", "mse"])
            metrics = {name: value for name, value in zip(metric_names, metrics)}
        else:
            metrics = {"loss": metrics}
    
    # Add timestamp to metrics
    metrics['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metrics['model_name'] = model_name
    
    # Log metrics to TensorBoard
    with tf.summary.create_file_writer(log_dir).as_default():
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                tf.summary.scalar(f"final_{metric_name}", value, step=0)
    
    # Save metrics as JSON for easy access
    with open(f"{log_dir}/final_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Update global results file for easy comparison
    results_file = "logs/all_model_results.csv"
    results_dir = os.path.dirname(results_file)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)
    
    # Prepare row for results file
    result_row = {
        'model_name': model_name,
        'timestamp': metrics['timestamp'],
        'loss': metrics.get('loss', None),
        'mae': metrics.get('mae', None),
        'mse': metrics.get('mse', None),
        'correlation': metrics.get('correlation', None),
        'era_correlation': metrics.get('era_correlation', None)
    }
    
    # Add config parameters if provided
    if config:
        result_row.update({
            'optimizer': config.get('optimizer', 'unknown'),
            'learning_rate': config.get('learning_rate', 'unknown'),
            'batch_size': config.get('batch_size', 'unknown'),
            'feature_subset': config.get('feature_subset', 'unknown'),
            'loss_fn': config.get('loss_fn', 'unknown')
        })
    
    # Create or append to CSV
    results_df = pd.DataFrame([result_row])
    if os.path.exists(results_file):
        try:
            existing_df = pd.read_csv(results_file)
            results_df = pd.concat([existing_df, results_df], ignore_index=True)
        except Exception as e:
            print(f"Warning: Could not read existing results file: {e}")
    
    # Save updated results
    results_df.to_csv(results_file, index=False)
    
    print(f"Final metrics saved to {log_dir}/final_metrics.json")
    print(f"Results added to global comparison file: {results_file}")


def analyze_feature_importance(model, feature_set):
    """Analyze the learned feature importance weights"""
    print("\nAnalyzing feature importance...")
    
    # Find the feature importance layer
    for layer in model.layers:
        if 'feature_importance' in layer.name:
            # Get the learned weights
            weights = layer.get_weights()[0]
            
            # Convert to importance scores using softmax
            importance = tf.nn.softmax(weights).numpy()
            
            # Create a dataframe with feature importance
            importance_df = pd.DataFrame({
                'feature': feature_set,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            # Print top 20 most important features
            print("\nTop 20 most important features:")
            print(importance_df.head(20))
            
            # Save feature importance to CSV
            importance_df.to_csv('feature_importance.csv', index=False)
            print("Full feature importance saved to feature_importance.csv")
            
            # Plot feature importance
            plt.figure(figsize=(12, 6))
            plt.bar(range(20), importance_df['importance'].head(20))
            plt.xticks(range(20), importance_df['feature'].head(20), rotation=90)
            plt.title('Top 20 Feature Importance')
            plt.tight_layout()
            plt.savefig('feature_importance.png')
            
            return importance_df
    
    print("No feature importance layer found in the model.")
    return None


#--- Training Functions ---#

def train_standard_model(
    model_fn,
    model_name,
    epochs=100,
    batch_size=512,
    learning_rate=0.0003,
    validation_split=0.15,
    feature_subset="all",
    loss_fn="mae",
    data_path_train="data/train.parquet",
    data_path_val="data/validation.parquet",
    data_path_metadata="data/features.json"
):
    """
    Standard training function for various models.
    
    Args:
        model_fn: Function to create model
        model_name: Name prefix for the model (used for saving)
        epochs: Maximum number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for Adam optimizer
        validation_split: Fraction of training data to use for validation
        feature_subset: Which feature subset to use ('small', 'medium', 'all')
        loss_fn: Loss function to use ('mae', 'mse', 'correlation', 'correlation_aware')
        data_path_*: Paths to datasets
    """
    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)
    
    print(f"\n{'='*80}")
    print(f"Training {model_name} model with {feature_subset.upper()} features")
    print(f"{'='*80}")
    
    # Initialize data handler with specified features
    data_handler = DataHandler(
        data_path_train=data_path_train,
        data_path_val=data_path_val,
        data_path_metadata=data_path_metadata,
        batch_size=batch_size,
        subset_features=None if feature_subset == "all" else feature_subset
    )
    
    # Load and get feature metadata
    feature_metadata = json.load(open(data_path_metadata))
    all_features_count = len(feature_metadata["feature_sets"]["all"])
    
    # Get dataset info
    data_handler.get_dataset_info()
    
    # Load training data
    print("Loading training data...")
    train_data = data_handler.load_train_data()
    
    # Extract features and target
    feature_set = data_handler.feature_set
    X = train_data[feature_set].values
    y = train_data["target"].values
    
    print(f"Feature set size: {len(feature_set)} out of {all_features_count} total features")
    print(f"Training data shape: {X.shape}")
    
    # Create full configuration dictionary
    config = {
        'model': model_name,
        'optimizer': 'adam',
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'epochs': epochs,
        'feature_subset': feature_subset,
        'loss_fn': loss_fn,
        'validation_split': validation_split,
        'dropout_rate': 0.3,  # Default value - extract from model if possible later
        'features_count': len(feature_set),
        'total_features': all_features_count
    }
    
    # Create callbacks with full config
    callbacks, log_dir, checkpoint_dir = create_callbacks(model_name, config)
    
    # Create the model
    print(f"Creating {model_name} model...")
    model = model_fn(input_shape=(len(feature_set),))
    
    # Set the model name if not already set
    if not hasattr(model, 'name') or not model.name:
        model._name = model_name
    
    # Print model summary
    model.summary()
    
    # Select optimizer
    optimizer_name = config.get('optimizer', 'adam').lower()
    try:
        if optimizer_name == 'adam':
            from src.optimizers.Adam import optimizer as custom_optimizer
        elif optimizer_name == 'improvedadam':
            from src.optimizers.ImprovedAdam import optimizer as custom_optimizer
        elif optimizer_name == 'nadam':
            from src.optimizers.Nadam import optimizer as custom_optimizer
        elif optimizer_name == 'rmsprop':
            from src.optimizers.RMSprop import optimizer as custom_optimizer
        elif optimizer_name == 'sgd':
            from src.optimizers.SGD import optimizer as custom_optimizer
        elif optimizer_name == 'adadelta':
            from src.optimizers.Adadelta import optimizer as custom_optimizer
        elif optimizer_name == 'adagrad':
            from src.optimizers.Adagrad import optimizer as custom_optimizer
        elif optimizer_name == 'adamax':
            from src.optimizers.Adamax import optimizer as custom_optimizer
        else:
            print(f"Warning: Optimizer {optimizer_name} not found, using default Adam")
            from src.optimizers.Adam import optimizer as custom_optimizer
        
        print(f"Using optimizer: {custom_optimizer.name}")
    except Exception as e:
        print(f"Error loading optimizer {optimizer_name}: {e}")
        print("Using default Adam optimizer")
        custom_optimizer = "adam"
    
    # Select loss function
    if loss_fn == "mae":
        loss = "mae"
        metrics = ["mse"]
    elif loss_fn == "mse":
        loss = "mse" 
        metrics = ["mae"]
    elif loss_fn == "correlation":
        loss = correlation_loss
        metrics = ["mae", "mse"]
    elif loss_fn == "correlation_aware":
        loss = CorrelationAwareLoss()
        metrics = ["mae", "mse"]
    else:
        # Default to MAE if loss function not recognized
        print(f"Warning: Loss function '{loss_fn}' not recognized, using MAE.")
        loss = "mae"
        metrics = ["mse"]
    
    # Compile the model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    
    # Train the model
    print(f"\nTraining model for up to {epochs} epochs...")
    history = model.fit(
        X, y,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save the model
    model_path = os.path.join('exports', f'{model_name}_best.keras')
    model.save(model_path)
    print(f"Model saved to: {model_path}")
    
    # Plot training history
    plot_training_history(history, model_name)
    
    # Evaluate on validation data
    print("\nEvaluating on validation data...")
    val_metrics = evaluate_on_validation(model, data_handler, feature_set)
    
    # Get prediction correlation
    val_data = data_handler.load_validation_data()
    X_val = val_data[feature_set].values
    y_val = val_data["target"].values
    y_pred = model.predict(X_val)
    
    # Calculate correlation
    correlation = np.corrcoef(y_val, y_pred.flatten())[0, 1]
    
    # Calculate era-wise correlation if available
    era_correlation = None
    if "era" in val_data.columns:
        era_scores = []
        unique_eras = val_data["era"].unique()
        
        for era in unique_eras:
            era_mask = val_data["era"] == era
            if era_mask.sum() > 5:  # Ensure we have enough samples
                era_y = y_val[era_mask]
                era_pred = y_pred.flatten()[era_mask]
                era_corr = np.corrcoef(era_y, era_pred)[0, 1]
                era_scores.append(era_corr)
        
        if era_scores:
            era_correlation = np.mean(era_scores)
    
    # Create metrics dict
    final_metrics = {
        'val_loss': val_metrics[0],
        'correlation': correlation,
    }
    
    # Add other metrics if available
    if len(val_metrics) > 1:
        final_metrics['mae'] = val_metrics[1]
    if len(val_metrics) > 2:
        final_metrics['mse'] = val_metrics[2]
    if era_correlation is not None:
        final_metrics['era_correlation'] = era_correlation
    
    # Log final metrics
    log_final_metrics(log_dir, final_metrics, model_name, config)
    
    # Analyze feature importance if possible
    try:
        importance_df = analyze_feature_importance(model, feature_set)
        if importance_df is not None:
            importance_df.to_csv(f"{log_dir}/feature_importance.csv", index=False)
    except Exception as e:
        print(f"Could not analyze feature importance: {e}")
    
    print(f"\n{model_name} training complete!")
    print(f"TensorBoard logs saved to: {log_dir}")
    print("Run the following command to start TensorBoard:")
    print(f"tensorboard --logdir=logs/experiments")
    
    return model, history


def train_kfold_model(
    model_fn,
    model_name,
    epochs=100,
    batch_size=512,
    learning_rate=0.0003,
    feature_subset="all",
    loss_fn="mae",
    n_folds=3,
    data_path_train="data/train.parquet",
    data_path_val="data/validation.parquet",
    data_path_metadata="data/features.json"
):
    """
    Train models using K-fold cross-validation.
    
    Args:
        model_fn: Function to create model
        model_name: Name prefix for the model (used for saving)
        epochs: Maximum number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for Adam optimizer
        feature_subset: Which feature subset to use ('small', 'medium', 'all')
        loss_fn: Loss function to use ('mae', 'mse', 'correlation', 'correlation_aware')
        n_folds: Number of folds for cross-validation
        data_path_*: Paths to datasets
    """
    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)
    
    print(f"\n{'='*80}")
    print(f"Training {model_name} model with {feature_subset.upper()} features using {n_folds}-fold CV")
    print(f"{'='*80}")
    
    # Initialize data handler with specified features
    data_handler = DataHandler(
        data_path_train=data_path_train,
        data_path_val=data_path_val,
        data_path_metadata=data_path_metadata,
        batch_size=batch_size,
        subset_features=None if feature_subset == "all" else feature_subset
    )
    
    # Get dataset info
    data_handler.get_dataset_info()
    
    # Load training data
    print("Loading training data...")
    train_data = data_handler.load_train_data()
    
    # Extract features and target
    feature_set = data_handler.feature_set
    X = train_data[feature_set].values
    y = train_data["target"].values
    
    # Initialize KFold
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Initialize fold results
    fold_models = []
    fold_histories = []
    fold_val_losses = []
    
    # Loop through folds
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\n{'='*40}")
        print(f"Training fold {fold_idx + 1}/{n_folds}")
        print(f"{'='*40}")
        
        # Split data for this fold
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Create callbacks for this fold
        fold_name = f"{model_name}_fold{fold_idx}"
        callbacks, log_dir, checkpoint_dir = create_callbacks(fold_name)
        
        # Create the model
        print(f"Creating model for fold {fold_idx + 1}...")
        model = model_fn(input_shape=(len(feature_set),))
        
        # Set the model name if not already set
        if not hasattr(model, 'name') or not model.name:
            model._name = fold_name
        
        # Select loss function
        if loss_fn == "mae":
            loss = "mae"
            metrics = ["mse"]
        elif loss_fn == "mse":
            loss = "mse" 
            metrics = ["mae"]
        elif loss_fn == "correlation":
            loss = correlation_loss
            metrics = ["mae", "mse"]
        elif loss_fn == "correlation_aware":
            loss = CorrelationAwareLoss()
            metrics = ["mae", "mse"]
        else:
            # Default to MAE if loss function not recognized
            print(f"Warning: Loss function '{loss_fn}' not recognized, using MAE.")
            loss = "mae"
            metrics = ["mse"]
        
        # Compile the model
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        
        # Train the model
        print(f"\nTraining fold {fold_idx + 1} for up to {epochs} epochs...")
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        # Save the model
        model_path = os.path.join('exports', f'{fold_name}_best.keras')
        model.save(model_path)
        print(f"Model for fold {fold_idx + 1} saved to: {model_path}")
        
        # Plot training history for this fold
        plot_training_history(history, fold_name)
        
        # Evaluate on validation data
        print(f"\nEvaluating fold {fold_idx + 1} on validation data...")
        val_metrics = evaluate_on_validation(model, data_handler, feature_set)
        
        # Store fold results
        fold_models.append(model)
        fold_histories.append(history)
        fold_val_losses.append(val_metrics[0])  # First metric is loss
    
    # Print summary of folds
    print("\n" + "="*80)
    print(f"K-Fold Cross-Validation Summary for {model_name}")
    print("="*80)
    
    for i, val_loss in enumerate(fold_val_losses):
        print(f"Fold {i+1} Validation Loss: {val_loss:.4f}")
    
    mean_val_loss = np.mean(fold_val_losses)
    std_val_loss = np.std(fold_val_losses)
    print(f"\nMean Validation Loss: {mean_val_loss:.4f} ± {std_val_loss:.4f}")
    
    # Identify best fold
    best_fold_idx = np.argmin(fold_val_losses)
    print(f"Best model: Fold {best_fold_idx + 1} with validation loss {fold_val_losses[best_fold_idx]:.4f}")
    
    return fold_models, fold_histories


#--- Main Functions ---#

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Train various models for the dspro2 project",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--model", 
        type=str, 
        default="correlation",
        choices=["base", "deep", "wide", "residual", "correlation", "advanced", "best"],
        help="Model architecture to train"
    )
    
    parser.add_argument(
        "--optimizer",
        type=str,
        default="Adam",
        choices=["Adam", "ImprovedAdam", "Nadam", "RMSprop", "SGD", "Adadelta", "Adagrad", "Adamax"],
        help="Optimizer to use for training"
    )
    
    parser.add_argument(
        "--features",
        type=str,
        default="all",
        choices=["small", "medium", "all"],
        help="Feature set size to use"
    )
    
    parser.add_argument(
        "--loss",
        type=str,
        default="mae",
        choices=["mae", "mse", "correlation", "correlation_aware"],
        help="Loss function to use for training"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        default="standard",
        choices=["standard", "kfold"],
        help="Training mode (standard or k-fold cross-validation)"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Maximum number of training epochs"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Batch size for training"
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.0003,
        help="Learning rate for Adam optimizer"
    )
    
    parser.add_argument(
        "--n_folds",
        type=int,
        default=3,
        help="Number of folds for K-fold cross-validation (used only when mode=kfold)"
    )
    
    parser.add_argument(
        "--train_data",
        type=str,
        default="data/train.parquet",
        help="Path to training data"
    )
    
    parser.add_argument(
        "--val_data",
        type=str,
        default="data/validation.parquet",
        help="Path to validation data"
    )
    
    parser.add_argument(
        "--meta_data",
        type=str,
        default="data/features.json",
        help="Path to feature metadata"
    )
    
    return parser.parse_args()


def main():
    """Entry point of the program"""
    # Parse arguments
    args = parse_args()
    
    # Map model name to model function
    model_funcs = {
        "base": lambda input_shape: tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape, dtype=tf.uint8),
            tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1)
        ], name="BaseModel"),
        "deep": deep_model,
        "wide": lambda input_shape: tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape, dtype=tf.uint8),
            tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32)),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(1)
        ], name="WideModel"),
        "residual": residual_model,
        "correlation": create_correlation_model,
        "advanced": advanced_model,
        "best": best_model
    }
    
    # Select model function based on argument
    if args.model not in model_funcs:
        print(f"Error: Unknown model type '{args.model}'")
        print(f"Available models: {', '.join(model_funcs.keys())}")
        return
    
    model_fn = model_funcs[args.model]
    model_name = f"{args.model.capitalize()}Model"
    
    # Train the model
    if args.mode == "standard":
        model, history = train_standard_model(
            model_fn=model_fn,
            model_name=model_name,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            feature_subset=args.features,
            loss_fn=args.loss,
            data_path_train=args.train_data,
            data_path_val=args.val_data,
            data_path_metadata=args.meta_data
        )
    elif args.mode == "kfold":
        models, histories = train_kfold_model(
            model_fn=model_fn,
            model_name=model_name,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            feature_subset=args.features,
            loss_fn=args.loss,
            n_folds=args.n_folds,
            data_path_train=args.train_data,
            data_path_val=args.val_data,
            data_path_metadata=args.meta_data
        )
    else:
        print(f"Error: Unknown training mode '{args.mode}'")
        return


if __name__ == "__main__":
    main()
