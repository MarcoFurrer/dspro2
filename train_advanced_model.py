import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
import json

from src.data_handler import DataHandler
from src.models.CorrelationModel import advanced_model
from src.model_evaluator import ModelEvaluator

# Custom correlation loss function
def correlation_loss(y_true, y_pred):
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

def train_advanced_model(
    seed=42,
    epochs=100,
    batch_size=512,
    learning_rate=0.0003,
    model_name="AdvancedFullModel",
    validation_split=0.15,
    use_all_features=True
):
    # Set random seeds for reproducibility
    tf.random.set_seed(seed)
    np.random.seed(seed)
    
    print(f"\n{'='*80}")
    print(f"Training Advanced Full Model with {'ALL' if use_all_features else 'MEDIUM'} features")
    print(f"{'='*80}")
    
    # Initialize data handler with all features
    data_handler = DataHandler(
        data_path_train="data/train.parquet",
        data_path_val="data/validation.parquet",
        data_path_metadata="data/features.json",
        batch_size=batch_size,
        subset_features=None if use_all_features else "medium"  # None means use all features
    )
    
    # Load and get feature metadata
    feature_metadata = json.load(open("data/features.json"))
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
    
    # Create log directory with timestamp for TensorBoard
    log_dir = "logs/fit/advanced_" + datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(log_dir, exist_ok=True)
    
    # Create checkpoint directory
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
            filepath=os.path.join(checkpoint_dir, f"{model_name}_best.keras"),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            update_freq='epoch'
        )
    ]
    
    # Create the advanced model
    print("Creating advanced model...")
    model = advanced_model(input_shape=(len(feature_set),))
    
    # Print model summary
    model.summary()
    
    # Compile the model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=correlation_loss,  # Use correlation-aware loss
        metrics=['mae', 'mse']
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
    evaluate_on_validation(model, data_handler, feature_set)
    
    # Analyze feature importance if the model has the feature importance layer
    try:
        analyze_feature_importance(model, feature_set)
    except:
        print("Could not analyze feature importance - layer not found.")
    
    print(f"\n{model_name} training complete!")
    print(f"TensorBoard logs saved to: {log_dir}")
    print("Run the following command to start TensorBoard:")
    print(f"tensorboard --logdir={log_dir}")
    
    return model, history

def evaluate_on_validation(model, data_handler, feature_set):
    """Evaluate model on validation data"""
    # Load validation data
    val_data = data_handler.load_validation_data()
    
    # Prepare validation data
    X_val = val_data[feature_set].values
    y_val = val_data["target"].values
    
    # Evaluate
    val_metrics = model.evaluate(X_val, y_val, verbose=1)
    print(f"Validation MAE: {val_metrics[1]:.4f}")
    print(f"Validation MSE: {val_metrics[2]:.4f}")
    
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
    
    # MAE plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train')
    plt.plot(history.history['val_mae'], label='Validation')
    plt.title('MAE During Training')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{model_name}_training_history.png')
    plt.close()

if __name__ == "__main__":
    # Train the advanced model with all features
    model, history = train_advanced_model(
        epochs=150,
        batch_size=512,  # Smaller batch size to handle the larger model
        learning_rate=0.0003,
        model_name="AdvancedFullModel",
        use_all_features=True  # Set to True to use all ~2400 features
    )