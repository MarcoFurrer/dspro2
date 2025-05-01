import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
import json
from sklearn.model_selection import KFold

from src.data_handler import DataHandler
from src.models.BestModel import model as best_model
from src.losses.DistributionAwareLoss import get_best_loss, DistributionAwareLoss
from src.model_evaluator import ModelEvaluator

# Set seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Enable memory growth to prevent TF from allocating all GPU memory at once
physical_devices = tf.config.list_physical_devices('GPU')
try:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
except:
    # If no GPU is available or operation isn't supported, continue
    pass

def train_best_model(
    seed=42,
    epochs=150,
    batch_size=512,
    learning_rate=0.0002,
    model_name="BestModel",
    feature_subset="medium",
    n_folds=3
):
    """
    Train our best model that is specifically designed to avoid target distribution memorization.
    
    Args:
        seed: Random seed for reproducibility
        epochs: Maximum number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for Adam optimizer
        model_name: Name prefix for the saved model
        feature_subset: Which feature subset to use ('small', 'medium', 'all')
        n_folds: Number of cross-validation folds
    """
    # Set random seeds for reproducibility
    tf.random.set_seed(seed)
    np.random.seed(seed)

    print(f"\n{'='*80}")
    print(f"Training Best Model to avoid target distribution memorization")
    print(f"Using {feature_subset} feature set with {n_folds}-fold validation")
    print(f"{'='*80}\n")
    
    # Initialize data handler with appropriate settings
    data_handler = DataHandler(
        data_path_train="data/train.parquet",
        data_path_val="data/validation.parquet",
        data_path_metadata="data/features.json",
        batch_size=batch_size,
        subset_features=feature_subset
    )
    
    # Get dataset info and feature set
    data_handler.get_dataset_info()
    feature_set = data_handler.feature_set
    print(f"Feature set size: {len(feature_set)}")
    
    # Load all training data - using pandas to load directly
    print("Loading training data...")
    train_data = pd.read_parquet(
        data_handler.data_path_train,
        columns=["era", "target"] + feature_set
    )
    
    # Extract features and target
    X = train_data[feature_set].values
    y = train_data["target"].values
    
    print(f"Training data shape: {X.shape}")
    
    # Prepare cross-validation
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    # For storing fold results
    fold_histories = []
    fold_val_scores = []
    fold_models = []
    
    # Create necessary directories
    os.makedirs('logs/fit', exist_ok=True)
    os.makedirs('models/checkpoints', exist_ok=True)
    os.makedirs('exports', exist_ok=True)
    os.makedirs('analysis', exist_ok=True)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\nTraining Fold {fold+1}/{n_folds}")
        
        # Split data for this fold
        X_train_fold = X[train_idx]
        y_train_fold = y[train_idx]
        X_val_fold = X[val_idx]
        y_val_fold = y[val_idx]
        
        # Create log directory with timestamp for TensorBoard
        log_dir = f"logs/fit/{model_name}_fold{fold}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        os.makedirs(log_dir, exist_ok=True)
        
        # Create checkpoint directory
        checkpoint_dir = os.path.join('models', 'checkpoints', model_name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Define callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=20,  # More patience for this complex model
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=0.00001,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=os.path.join(checkpoint_dir, f"{model_name}_best_fold{fold}.keras"),
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

        # Create model
        print(f"Building model with input shape: {X_train_fold.shape[1]}")
        model = best_model(input_shape=(X_train_fold.shape[1],))
        model.summary()
        
        # Custom loss function that addresses target distribution memorization
        loss_fn = get_best_loss()
        
        # Compile model with Adam optimizer
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=['mae', 'mse']  # Track standard MAE and MSE for comparison
        )
        
        # Train model
        print(f"\nTraining fold {fold+1}...")
        history = model.fit(
            X_train_fold, y_train_fold,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val_fold, y_val_fold),
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate model
        val_score = model.evaluate(X_val_fold, y_val_fold, verbose=1)
        print(f"Fold {fold+1} Validation MAE: {val_score[1]:.4f}")
        
        # Save history and scores
        fold_histories.append(history.history)
        fold_val_scores.append(val_score[1])  # MAE score
        fold_models.append(model)
        
        # Feature importance analysis
        analyze_feature_importance(model, feature_set, fold=fold)
    
    # Calculate average validation score
    avg_val_score = np.mean(fold_val_scores)
    print(f"\nAverage Validation MAE across {n_folds} folds: {avg_val_score:.4f}")
    
    # Determine best model
    best_fold_idx = np.argmin(fold_val_scores)
    best_model_fold = fold_models[best_fold_idx]
    print(f"Best model from fold {best_fold_idx+1} with MAE: {fold_val_scores[best_fold_idx]:.4f}")
    
    # Save best model
    best_model_path = os.path.join('exports', f'{model_name}_best.keras')
    best_model_fold.save(best_model_path)
    print(f"\nBest model saved to: {best_model_path}")
    
    # Plot training history
    plot_training_history(fold_histories, model_name)
    
    # Load and evaluate on validation data
    validate_on_unseen_data(best_model_fold, data_handler, feature_set)
    
    # Check for distribution memorization
    check_distribution_memorization(best_model_fold, data_handler, feature_set)
    
    return best_model_fold, fold_histories


def analyze_feature_importance(model, feature_set, fold=0):
    """Analyze feature importance from the model if available."""
    try:
        # Try to extract the weights from the ConditionalDistributionLayer
        distribution_layer = None
        for layer in model.layers:
            if "conditional_distribution" in layer.name.lower():
                distribution_layer = layer
                break
        
        if distribution_layer is not None:
            # Get the importance weights
            importance_weights = distribution_layer.get_weights()[1]  # mixing_weights
            
            # Create DataFrame with feature names and importance
            importance_df = pd.DataFrame({
                'Feature': feature_set,
                'Importance': importance_weights
            })
            
            # Sort by importance
            importance_df = importance_df.sort_values('Importance', ascending=False)
            
            # Print top 10 most important features
            print("\nTop 10 most important features:")
            print(importance_df.head(10))
            
            # Save to CSV
            os.makedirs('analysis', exist_ok=True)
            importance_df.to_csv(f'analysis/feature_importance_fold{fold}.csv', index=False)
            
            # Plot top 20 features
            plt.figure(figsize=(10, 8))
            plt.barh(importance_df.head(20)['Feature'], importance_df.head(20)['Importance'])
            plt.xlabel('Importance')
            plt.title(f'Top 20 Feature Importance (Fold {fold})')
            plt.gca().invert_yaxis()  # Invert y-axis to have highest at the top
            plt.tight_layout()
            plt.savefig(f'analysis/feature_importance_fold{fold}.png')
            plt.close()
            
    except Exception as e:
        print(f"Could not analyze feature importance: {e}")


def validate_on_unseen_data(model, data_handler, feature_set):
    """Validate model performance on unseen validation data with detailed analysis."""
    print("\nValidating on unseen data...")
    
    # Load validation data
    val_data = data_handler.load_validation_data()
    
    # Prepare validation data
    X_val = val_data[feature_set].values
    y_val = val_data["target"].values
    
    # Make predictions
    y_pred = model.predict(X_val)
    
    # Calculate various error metrics
    mae = np.mean(np.abs(y_val - y_pred))
    mse = np.mean(np.square(y_val - y_pred))
    rmse = np.sqrt(mse)
    
    print(f"Validation MAE: {mae:.4f}")
    print(f"Validation MSE: {mse:.4f}")
    print(f"Validation RMSE: {rmse:.4f}")
    
    # Calculate correlation
    correlation = np.corrcoef(y_val, y_pred.flatten())[0, 1]
    print(f"Correlation between predictions and targets: {correlation:.4f}")
    
    # Calculate error by target value
    target_values = np.unique(y_val)
    print("\nError by target value:")
    for value in target_values:
        mask = np.isclose(y_val, value)
        if np.any(mask):
            value_mae = np.mean(np.abs(y_val[mask] - y_pred[mask]))
            print(f"  Target {value:.2f}: MAE = {value_mae:.4f}")
    
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
    
    # Create a scatter plot of predictions vs targets
    plt.figure(figsize=(10, 8))
    plt.scatter(y_val, y_pred, alpha=0.3)
    plt.plot([0, 1], [0, 1], 'r--')  # Diagonal line for reference
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('Model Predictions vs True Values')
    plt.savefig('analysis/prediction_scatter.png')
    plt.close()


def check_distribution_memorization(model, data_handler, feature_set):
    """Check if model is just memorizing the target distribution."""
    print("\nChecking for target distribution memorization...")
    
    # Load validation data
    val_data = data_handler.load_validation_data()
    
    # Prepare validation data
    X_val = val_data[feature_set].values
    y_val = val_data["target"].values
    
    # Make predictions
    y_pred = model.predict(X_val).flatten()
    
    # Plot histograms of targets and predictions
    plt.figure(figsize=(12, 6))
    
    # Create a combined histogram
    plt.hist(y_val, bins=20, alpha=0.5, label='True Values', density=True)
    plt.hist(y_pred, bins=20, alpha=0.5, label='Predictions', density=True)
    
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title('Distribution of Target Values vs Predictions')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('analysis/distribution_comparison.png')
    plt.close()
    
    # Calculate KL divergence between distributions
    from scipy.stats import entropy
    
    # Create histograms for KL divergence calculation
    hist_true, bin_edges = np.histogram(y_val, bins=20, density=True)
    hist_pred, _ = np.histogram(y_pred, bins=bin_edges, density=True)
    
    # Add small constant to avoid division by zero
    hist_true = hist_true + 1e-10
    hist_pred = hist_pred + 1e-10
    
    # Normalize
    hist_true = hist_true / np.sum(hist_true)
    hist_pred = hist_pred / np.sum(hist_pred)
    
    # Calculate KL divergence in both directions
    kl_true_to_pred = entropy(hist_true, hist_pred)
    kl_pred_to_true = entropy(hist_pred, hist_true)
    
    print(f"KL divergence (true to pred): {kl_true_to_pred:.4f}")
    print(f"KL divergence (pred to true): {kl_pred_to_true:.4f}")
    
    # Check conditional prediction variation
    target_values = np.sort(np.unique(y_val))
    print("\nConditional prediction variation:")
    
    for target in target_values:
        mask = np.isclose(y_val, target)
        if np.sum(mask) > 10:  # Only if we have enough samples
            target_preds = y_pred[mask]
            pred_std = np.std(target_preds)
            print(f"  Target {target:.2f}: Prediction std dev = {pred_std:.4f}")
    
    # Plot prediction distribution conditional on target value
    plt.figure(figsize=(15, 10))
    
    for i, target in enumerate(target_values):
        mask = np.isclose(y_val, target)
        if np.sum(mask) > 10:
            plt.subplot(len(target_values), 1, i+1)
            plt.hist(y_pred[mask], bins=20, alpha=0.7)
            plt.axvline(target, color='r', linestyle='--')
            plt.title(f'Predictions when target = {target:.2f}')
            plt.ylabel('Count')
            if i == len(target_values) - 1:
                plt.xlabel('Predicted Value')
    
    plt.tight_layout()
    plt.savefig('analysis/conditional_predictions.png')
    plt.close()


def plot_training_history(fold_histories, model_name):
    """Plot training history across folds."""
    plt.figure(figsize=(15, 10))
    
    # Plot loss
    plt.subplot(2, 1, 1)
    for i, history in enumerate(fold_histories):
        plt.plot(history['loss'], label=f'Train Loss (Fold {i+1})')
        plt.plot(history['val_loss'], label=f'Val Loss (Fold {i+1})')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Plot MAE
    plt.subplot(2, 1, 2)
    for i, history in enumerate(fold_histories):
        plt.plot(history['mae'], label=f'Train MAE (Fold {i+1})')
        plt.plot(history['val_mae'], label=f'Val MAE (Fold {i+1})')
    plt.title('Model MAE')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'analysis/{model_name}_training_history.png')
    plt.close()


if __name__ == "__main__":
    # Create analysis directory
    os.makedirs('analysis', exist_ok=True)
    
    print("Training Best Model to avoid target distribution memorization problem...")
    
    # Uncomment to train with all features if preferred
    trained_model, histories = train_best_model(
        epochs=150,
        batch_size=512,
        learning_rate=0.0002,
        model_name="BestModel",
        feature_subset="medium",  # "all" for all features
        n_folds=3
    )
    
    print("\nBest Model training and evaluation complete!")
    print("Check the analysis directory for detailed evaluation and visualizations.")