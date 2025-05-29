#!/usr/bin/env python
"""
train_ordinal_model.py

This script trains and evaluates the specialized ordinal model that leverages
domain knowledge about the ordinal relationship in the target values.
Memory-optimized version.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, ModelCheckpoint
import datetime
import gc
from src.model_runner import EfficientCategoricalModel
from src.models.OrdinalModel import (
    model, ordinal_loss, ordinal_mae, 
    OrdinalLayer, FeatureInteractionLayer, SelfAttentionLayer
)
from src.optimizers.OrdinalAdam import optimizer

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Enable memory growth to prevent TF from allocating all GPU memory at once
physical_devices = tf.config.list_physical_devices('GPU')
try:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
except:
    # If no GPU is available or operation isn't supported, continue
    pass

# Limit TensorFlow to only allocate a portion of GPU memory
# This is crucial for reducing memory usage
try:
    for device in physical_devices:
        tf.config.set_logical_device_configuration(
            device,
            [tf.config.LogicalDeviceConfiguration(memory_limit=2048)]  # Limit to 2GB
        )
except:
    # If no GPU is available or operation isn't supported, continue
    pass

# Use mixed precision for more efficient memory usage
tf.keras.mixed_precision.set_global_policy('mixed_float16')

def create_callbacks(model_name="OrdinalModel"):
    """Create training callbacks with appropriate settings"""
    # Create log directory with timestamp
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f"logs/fit/{model_name}_{current_time}"
    os.makedirs(log_dir, exist_ok=True)
    
    # Create export directory
    os.makedirs("exports", exist_ok=True)
    
    # TensorBoard callback - reduce profiling to save memory
    tensorboard = TensorBoard(
        log_dir=log_dir,
        histogram_freq=0,  # Disable histograms to save memory
        write_graph=False,  # Don't write graph to save memory
        update_freq='epoch',
        profile_batch=0  # Disable profiling to save memory
    )
    
    # Early stopping with longer patience
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    # Reduce learning rate on plateau
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    # Model checkpoint - save in h5 format for better memory efficiency
    checkpoint = ModelCheckpoint(
        filepath=f"exports/{model_name}_best.keras",
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    return [tensorboard, early_stopping, reduce_lr, checkpoint], log_dir

def visualize_results(y_true, y_pred, save_path="ordinal_model_results.png"):
    """Create visualization of model performance"""
    plt.figure(figsize=(16, 10))
    
    # Plot actual vs predicted
    plt.subplot(2, 2, 1)
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Ordinal Model: Actual vs Predicted')
    
    # Plot distribution of predictions
    plt.subplot(2, 2, 2)
    plt.hist(y_pred, bins=20, alpha=0.7)
    plt.axvline(0.0, color='r', linestyle='--')
    plt.axvline(0.25, color='r', linestyle='--')
    plt.axvline(0.5, color='r', linestyle='--')
    plt.axvline(0.75, color='r', linestyle='--')
    plt.axvline(1.0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Frequency')
    plt.title('Distribution of Predictions')
    
    # Plot distribution of actual values
    plt.subplot(2, 2, 3)
    plt.hist(y_true, bins=20, alpha=0.7)
    plt.axvline(0.0, color='r', linestyle='--')
    plt.axvline(0.25, color='r', linestyle='--')
    plt.axvline(0.5, color='r', linestyle='--')
    plt.axvline(0.75, color='r', linestyle='--')
    plt.axvline(1.0, color='r', linestyle='--')
    plt.xlabel('Actual Values')
    plt.ylabel('Frequency')
    plt.title('Distribution of Actual Values')
    
    # Plot prediction error distribution
    plt.subplot(2, 2, 4)
    errors = np.abs(y_true - y_pred)
    plt.hist(errors, bins=20, alpha=0.7)
    plt.axvline(np.mean(errors), color='r', linestyle='--', 
                label=f'Mean Error: {np.mean(errors):.4f}')
    plt.xlabel('Absolute Error')
    plt.ylabel('Frequency')
    plt.title('Error Distribution')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    return save_path

def load_data_in_chunks(file_path, chunk_size=10000):
    """Load data in chunks to reduce memory usage"""
    return pd.read_parquet(file_path, engine='pyarrow')

def train_model_with_reduced_memory(model, X_train, y_train, X_val, y_val, callbacks, batch_size, epochs):
    """Train model with techniques to reduce memory usage"""
    # Calculate steps per epoch (processing in smaller batches)
    steps_per_epoch = len(X_train) // batch_size
    validation_steps = len(X_val) // batch_size
    
    # Use a generator to feed data in smaller chunks
    def data_generator(X, y, batch_size):
        indices = np.arange(len(X))
        while True:
            np.random.shuffle(indices)
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:i+batch_size]
                yield X[batch_indices], y[batch_indices]
    
    # Create data generators
    train_gen = data_generator(X_train, y_train, batch_size)
    val_gen = data_generator(X_val, y_val, batch_size)
    
    # Train using generators to reduce memory usage
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_gen,
        validation_steps=validation_steps,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
        use_multiprocessing=False,  # Avoid multiprocessing to save memory
        workers=1  # Minimize worker threads
    )
    
    return history

def train_model_with_ordinal_focus(model, X_train, y_train, X_val, y_val, callbacks, batch_size, epochs):
    """Train model with techniques optimized for ordinal prediction"""
    # Calculate steps per epoch
    steps_per_epoch = len(X_train) // batch_size
    validation_steps = len(X_val) // batch_size
    
    # Create a custom batch generator that emphasizes learning the ordering relationship
    def ordinal_aware_generator(X, y, batch_size):
        indices = np.arange(len(X))
        
        # Calculate class frequencies for balanced sampling
        class_indices = [np.where(np.isclose(y, i/4.0))[0] for i in range(5)]
        class_sizes = [len(indices) for indices in class_indices]
        min_samples = max(50, min(class_sizes))  # Ensure minimum samples per class
        
        while True:
            # Create a balanced batch with equal representation from each class
            batch_indices = []
            for i in range(5):
                # Sample with replacement if needed to ensure enough samples
                if len(class_indices[i]) < min_samples:
                    sampled = np.random.choice(class_indices[i], min_samples, replace=True)
                else:
                    sampled = np.random.choice(class_indices[i], min_samples, replace=False)
                batch_indices.extend(sampled)
            
            # Shuffle the balanced indices
            np.random.shuffle(batch_indices)
            
            # Yield batches from the balanced set
            for i in range(0, len(batch_indices), batch_size):
                current_indices = batch_indices[i:min(i+batch_size, len(batch_indices))]
                yield X[current_indices].astype(np.float32), y[current_indices]
    
    # Create data generators
    train_gen = ordinal_aware_generator(X_train, y_train, batch_size)
    val_gen = ordinal_aware_generator(X_val, y_val, batch_size)
    
    # Calculate steps with balanced sampling
    balanced_steps = (5 * min_samples) // batch_size
    
    # Train using generators with ordinal focus
    history = model.fit(
        train_gen,
        steps_per_epoch=balanced_steps,
        validation_data=val_gen,
        validation_steps=validation_steps,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
        use_multiprocessing=False,
        workers=1
    )
    
    return history

def augment_data(X, y, noise_scale=0.01, augmentation_factor=0.05):
    """
    Improved data augmentation function that is more sensitive to feature types.
    - Adds smaller noise to features
    - Uses a more appropriate augmentation strategy for ordinal regression
    """
    # Create copies for augmentation
    X_aug = X.copy()
    y_aug = y.copy()
    
    # Selectively add noise based on feature importance
    # Identify features that have higher correlation with targets
    # This is a simplified approach - ideally we'd use feature importance from a trained model
    feature_mask = np.random.random(X.shape[1]) < 0.5  # Only modify 50% of features
    
    # Add selective noise to features - smaller values for more stability
    noise = np.random.normal(0, noise_scale, X_aug.shape) * feature_mask
    X_aug = X_aug + noise
    
    # For ordinal regression - small perturbation to target values
    # This smooths the ordinal boundaries a bit and helps generalization
    target_noise = np.random.normal(0, 0.02, y_aug.shape)  # Very small noise for targets
    y_aug = np.clip(y_aug + target_noise, 0, 1)  # Ensure targets stay in [0,1]
    
    return X_aug, y_aug

def main():
    try:
        print("Initializing memory-optimized ordinal model training with enhanced feature learning...")
        
        # Data paths
        data_path = "data/train.parquet"
        val_path = "data/validation.parquet"
        meta_path = "data/features.json"
        meta_model = "data/meta_model.parquet"
        
        # Reduced sample size to decrease memory usage while still learning patterns
        sample_size = 120000  # Slightly increased from 100000 to capture more patterns
        batch_size = 64       # Keep smaller batch size for memory efficiency
        
        # Create callbacks with more patience for the improved model
        callbacks, log_dir = create_callbacks(patience=15)  # Increased patience 
        
        # Load data with optimized settings
        print("Loading data in memory-efficient manner...")
        
        # Initialize EfficientCategoricalModel to get feature set
        print("Initializing EfficientCategoricalModel...")
        ecm = EfficientCategoricalModel(
            data_path_train=data_path,
            data_path_val=val_path,
            data_path_metadata=meta_path,
            data_path_meta_model=meta_model,
            batch_size=batch_size
        )
        
        # Get feature set
        ecm.data_handler.get_dataset_info()
        feature_set = ecm.data_handler.feature_set
        
        # Load and sample data in memory-efficient manner
        print(f"Loading and sampling {sample_size} training examples...")
        train_data = load_data_in_chunks(data_path)
        if sample_size and sample_size < len(train_data):
            # Sample without copying the entire DataFrame
            train_indices = np.random.choice(len(train_data), sample_size, replace=False)
            train_data = train_data.iloc[train_indices]
        
        # Extract features and target
        X_train = train_data[feature_set].values.astype(np.uint8)
        y_train = train_data['target'].values.astype(np.float32)
        
        # Analyze target distribution for class weighting
        print("Analyzing target distribution for balanced training...")
        target_counts = np.zeros(5)
        for val in y_train:
            idx = int(val * 4)
            target_counts[idx] += 1
            
        # Calculate class weights to focus on underrepresented classes
        total = np.sum(target_counts)
        class_weights = {i: total / (5 * target_counts[i]) for i in range(5)}
        print(f"Class weights: {class_weights}")
        
        # Clear train_data from memory to free up space
        del train_data
        gc.collect()
        
        # Load validation data
        print("Loading validation data...")
        val_data = load_data_in_chunks(val_path)
        
        # Use a smaller validation set to reduce memory
        val_sample_size = min(25000, len(val_data))  # Slight increase for better validation
        if val_sample_size < len(val_data):
            val_indices = np.random.choice(len(val_data), val_sample_size, replace=False)
            val_data = val_data.iloc[val_indices]
            
        X_val = val_data[feature_set].values.astype(np.uint8)
        y_val = val_data['target'].values.astype(np.float32)
        
        # Clear val_data from memory
        del val_data
        gc.collect()
        
        # Compile the enhanced model with custom loss and metric
        print("Compiling enhanced ordinal model...")
        model.compile(
            optimizer=optimizer,
            loss=ordinal_loss,
            metrics=[ordinal_mae]
        )
        
        # Display model summary
        model.summary()
        
        # Use a generator with data augmentation for better generalization
        def data_generator_with_augmentation(X, y, batch_size):
            indices = np.arange(len(X))
            
            # Synthetic Minority Over-sampling for rare classes
            class_indices = [np.where(y * 4 == i)[0] for i in range(5)]
            class_sizes = [len(indices) for indices in class_indices]
            max_size = max(class_sizes)
            
            while True:
                # Shuffle the indices
                np.random.shuffle(indices)
                
                for i in range(0, len(indices), batch_size):
                    # Get the batch indices
                    batch_indices = indices[i:min(i+batch_size, len(indices))]
                    batch_X = X[batch_indices]
                    batch_y = y[batch_indices]
                    
                    # Add minimal noise to features (data augmentation)
                    # This can help with generalization without changing the categorical nature
                    noise_mask = np.random.random(batch_X.shape) < 0.05  # 5% chance to modify
                    noise = np.random.randint(0, 2, size=batch_X.shape)  # 0 or 1
                    batch_X = batch_X.astype(np.float32)
                    batch_X[noise_mask] = noise[noise_mask]
                    
                    yield batch_X, batch_y
        
        # Create enhanced data generators
        train_gen = data_generator_with_augmentation(X_train, y_train, batch_size)
        val_gen = data_generator_with_augmentation(X_val, y_val, batch_size)
        
        # Calculate training steps
        steps_per_epoch = len(X_train) // batch_size
        validation_steps = len(X_val) // batch_size
        
        # Train the model with enhanced features
        print("\nTraining enhanced ordinal model with feature interaction learning...")
        history = model.fit(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_gen,
            validation_steps=validation_steps,
            epochs=50,  # Increased epochs for better convergence
            callbacks=callbacks,
            verbose=1,
            use_multiprocessing=False,  # Avoid multiprocessing to save memory
            workers=1,  # Minimize worker threads
            class_weight=class_weights  # Apply class weights for balanced learning
        )
        
        # Free up training data memory
        del X_train, y_train
        gc.collect()
        
        # Get the best model - using the trained model if there's an issue loading the saved one
        best_model_path = "exports/OrdinalModel_best.keras"
        best_model = model  # Default to the current model
        
        if os.path.exists(best_model_path):
            print(f"\nLoading best model from {best_model_path}")
            try:
                # Try loading with correct custom objects
                best_model = tf.keras.models.load_model(
                    best_model_path,
                    custom_objects={
                        'ordinal_loss': ordinal_loss,
                        'ordinal_mae': ordinal_mae,
                        'OrdinalLayer': OrdinalLayer,
                        'FeatureInteractionLayer': FeatureInteractionLayer,
                        'SelfAttentionLayer': SelfAttentionLayer
                    },
                    safe_mode=False  # Allow Lambda layer loading
                )
                print("Successfully loaded best model")
            except Exception as e:
                print(f"Could not load saved model: {str(e)}")
                print("Using current model for predictions")
                best_model = model
        
        # Make predictions with extra analysis
        print("\nMaking predictions on validation data (in batches)...")
        
        # Use batched prediction to save memory
        pred_batch_size = 1000
        num_batches = int(np.ceil(len(X_val) / pred_batch_size))
        
        # Collect predictions and intermediate outputs
        all_raw_predictions = []
        for i in range(num_batches):
            start_idx = i * pred_batch_size
            end_idx = min((i + 1) * pred_batch_size, len(X_val))
            batch_predictions = best_model.predict(X_val[start_idx:end_idx], verbose=0)
            all_raw_predictions.append(batch_predictions)
        
        # Combine batched predictions
        raw_predictions = np.vstack(all_raw_predictions)
        
        # Process the raw predictions (cumulative probabilities)
        # Add a column of ones (last threshold is always crossed)
        pred_probs = np.concatenate([raw_predictions, np.ones((raw_predictions.shape[0], 1))], axis=1)
        
        # Calculate class probabilities from cumulative probabilities
        class_probs = pred_probs[:, :-1] - pred_probs[:, 1:]
        
        # Get predicted class
        pred_classes = np.argmax(class_probs, axis=1)
        
        # Convert back to original scale
        predictions = pred_classes / 4.0  # For classes [0, 0.25, 0.5, 0.75, 1.0]
        
        # Clear memory
        del raw_predictions, pred_probs, class_probs, pred_classes, all_raw_predictions
        gc.collect()
        
        # Calculate metrics
        mae = np.mean(np.abs(y_val - predictions))
        mse = np.mean(np.square(y_val - predictions))
        
        # Calculate error by target class for detailed analysis
        class_errors = {}
        for i in range(5):
            target_value = i / 4.0
            mask = np.isclose(y_val, target_value)
            if np.any(mask):
                class_errors[target_value] = np.mean(np.abs(y_val[mask] - predictions[mask]))
        
        # Display detailed results
        print("\nENHANCED ORDINAL MODEL RESULTS:")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print("\nClass-specific errors:")
        for target_value, error in class_errors.items():
            print(f"  Class {target_value}: MAE = {error:.4f}")
        
        # Create visualization with added confusion matrix
        viz_path = visualize_enhanced_results(y_val, predictions)
        print(f"\nEnhanced results visualization saved to: {viz_path}")
        
        # Export error analysis (in a memory-efficient way)
        print("Creating error analysis file...")
        with open('ordinal_model_errors.csv', 'w') as f:
            f.write('actual,predicted,error\n')
            for i in range(len(y_val)):
                error = abs(y_val[i] - predictions[i])
                f.write(f"{y_val[i]},{predictions[i]},{error}\n")
        
        print("Error analysis saved to: ordinal_model_errors.csv")
        
        print("\nTraining complete!")
        print(f"TensorBoard logs saved to: {log_dir}")
        print("Run: tensorboard --logdir=logs/fit")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()