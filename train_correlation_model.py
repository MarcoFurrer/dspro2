import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.losses import MeanAbsoluteError
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os

from src.data_handler import DataHandler
from src.models.CorrelationModel import create_correlation_model
from src.model_evaluator import ModelEvaluator

# Custom loss function to encourage learning correlations and penalize just mimicking the distribution
class CorrelationAwareLoss(tf.keras.losses.Loss):
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

class FocalMAELoss(tf.keras.losses.Loss):
    def __init__(self, alpha=0.25, gamma=2.0, name="focal_mae_loss"):
        super().__init__(name=name)
        self.alpha = alpha
        self.gamma = gamma
        
    def call(self, y_true, y_pred):
        # Standard MAE calculation
        mae = tf.abs(y_true - y_pred)
        
        # Apply focal weighting: focus more on hard examples
        # Hard examples have higher error
        focal_weight = tf.pow(1.0 - tf.exp(-mae), self.gamma)
        
        # Apply weighting factor
        weighted_mae = self.alpha * focal_weight * mae
        
        return tf.reduce_mean(weighted_mae)

def train_correlation_model(
    seed=42,
    epochs=100,
    batch_size=1024,
    learning_rate=0.0005,
    model_name="CorrelationModel",
    use_correlation_loss=True,
    n_folds=5
):
    # Set random seeds for reproducibility
    tf.random.set_seed(seed)
    np.random.seed(seed)
    
    # Initialize data handler and get data
    data_handler = DataHandler()
    X_train, y_train = data_handler.get_train_data()
    X_val, y_val = data_handler.get_validation_data()
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    
    # Initialize feature and target sets for cross-validation
    features = pd.concat([X_train, X_val])
    targets = pd.concat([y_train, y_val])
    
    # Create output directories if they don't exist
    checkpoint_dir = os.path.join('models', 'checkpoints', model_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # For storing fold results
    fold_histories = []
    fold_val_scores = []
    fold_models = []
    
    # Cross-validation
    kfold = data_handler.get_cv_splits(n_splits=n_folds)
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(features)):
        print(f"\nTraining Fold {fold+1}/{n_folds}")
        
        # Split data for this fold
        X_train_fold = features.iloc[train_idx]
        y_train_fold = targets.iloc[train_idx]
        X_val_fold = features.iloc[val_idx]
        y_val_fold = targets.iloc[val_idx]
        
        # Create model
        model = create_correlation_model(input_shape=(X_train_fold.shape[1],))
        
        # Select loss function
        if use_correlation_loss:
            loss = CorrelationAwareLoss(distribution_penalty=0.2)
        else:
            loss = FocalMAELoss(alpha=0.25, gamma=2.0)
        
        # Compile model with Adam optimizer
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=['mae']  # Track standard MAE for comparison
        )
        
        # Setup callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
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
                log_dir=f'logs/fit/{model_name}_fold{fold}_{datetime.now().strftime("%Y%m%d-%H%M%S")}',
                histogram_freq=1,
                update_freq='epoch'
            )
        ]
        
        # Train model
        history = model.fit(
            X_train_fold, y_train_fold,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val_fold, y_val_fold),
            callbacks=callbacks,
            verbose=2
        )
        
        # Evaluate model
        val_score = model.evaluate(X_val_fold, y_val_fold, verbose=0)
        print(f"Fold {fold+1} Validation MAE: {val_score[1]:.4f}")
        
        # Save history and scores
        fold_histories.append(history.history)
        fold_val_scores.append(val_score[1])  # MAE score
        fold_models.append(model)
    
    # Calculate average validation score
    avg_val_score = np.mean(fold_val_scores)
    print(f"\nAverage Validation MAE across {n_folds} folds: {avg_val_score:.4f}")
    
    # Determine best model
    best_fold_idx = np.argmin(fold_val_scores)
    best_model = fold_models[best_fold_idx]
    print(f"Best model from fold {best_fold_idx+1} with MAE: {fold_val_scores[best_fold_idx]:.4f}")
    
    # Save best model
    best_model.save(os.path.join('exports', f'{model_name}_best.keras'))
    
    # Plot training history
    plot_training_history(fold_histories, model_name)
    
    # Return best model for further use
    return best_model

def plot_training_history(fold_histories, model_name):
    plt.figure(figsize=(15, 6))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    for i, history in enumerate(fold_histories):
        plt.plot(history['loss'], label=f'Fold {i+1} Train')
        plt.plot(history['val_loss'], label=f'Fold {i+1} Val', linestyle='--')
    
    plt.title('Loss During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # MAE plot
    plt.subplot(1, 2, 2)
    for i, history in enumerate(fold_histories):
        plt.plot(history['mae'], label=f'Fold {i+1} Train')
        plt.plot(history['val_mae'], label=f'Fold {i+1} Val', linestyle='--')
    
    plt.title('MAE During Training')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{model_name}_training_history.png')
    plt.close()

if __name__ == "__main__":
    # Train with correlation-aware loss
    best_model = train_correlation_model(
        epochs=150,
        batch_size=1024,
        learning_rate=0.0005,
        model_name="CorrelationModel",
        use_correlation_loss=True,
        n_folds=3  # Reduced number of folds for faster training
    )
    
    # Optional: Train with focal MAE loss for comparison
    # focal_model = train_correlation_model(
    #     epochs=150,
    #     batch_size=1024,
    #     learning_rate=0.0005,
    #     model_name="FocalModel",
    #     use_correlation_loss=False,
    #     n_folds=3
    # )