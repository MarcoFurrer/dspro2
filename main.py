from src.EfficientCategoricalModel import EfficientCategoricalModel
from src.models.Deep import model as deep_model
from src.models.ImprovedModel import model as improved_model
from src.models.CorrelationModel import model as correlation_model
from src.optimizers.Adam import optimizer as adam_optimizer
from src.optimizers.ImprovedAdam import optimizer as improved_optimizer
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import datetime
import os
import argparse
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold

# Set seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def create_callbacks(model_name="model", use_advanced=False):
    """Create training callbacks with appropriate settings"""
    # Create log directory with timestamp for TensorBoard
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(log_dir, exist_ok=True)
    
    # Create TensorBoard callback
    tensorboard = TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True,
        update_freq='epoch'
    )
    
    callbacks = [tensorboard]
    
    if use_advanced:
        # Create checkpoint directory
        checkpoint_dir = f"models/checkpoints/{model_name}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Add advanced callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=8,
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
        
        checkpoint = ModelCheckpoint(
            filepath=f"{checkpoint_dir}/best_model.keras",
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
        
        callbacks.extend([early_stopping, reduce_lr, checkpoint])
    
    return callbacks, log_dir

def train_standard(args):
    """Train using standard approach"""
    print("Initializing standard model training...")
    
    # Select model and optimizer based on args
    model = improved_model if args.use_improved_model else deep_model
    optimizer = improved_optimizer if args.use_improved_optimizer else adam_optimizer
    
    # Create callbacks
    callbacks, log_dir = create_callbacks(use_advanced=False)
    
    # Initialize model
    efficient_model = EfficientCategoricalModel(
        data_path_train=args.train_data,
        data_path_val=args.val_data,
        data_path_metadata=args.meta_data,
        data_path_meta_model=args.meta_model,
        batch_size=args.batch_size,
        model=model,
        optimizer=optimizer
    )
    
    # Train the model
    model, history = efficient_model.train(
        epochs=args.epochs,
        callbacks=callbacks
    )
    
    # Validate model if requested
    if args.validate:
        print("\nValidating model...")
        efficient_model.validate_model()
        performance_metrics = efficient_model.performance_eval()
        print("\nPerformance Metrics:")
        for metric, value in performance_metrics.items():
            print(f"{metric}: {value}")
    
    print("Training complete!")
    print(f"TensorBoard logs saved to: {log_dir}")
    print("Run the following command to start TensorBoard:")
    print(f"tensorboard --logdir={log_dir}")
    
    return model, history, efficient_model

def train_advanced(args):
    """Train using K-fold cross-validation and ensemble methods"""
    print("\n" + "="*50)
    print("Starting advanced training with K-fold cross-validation")
    print("="*50)
    
    # Load data 
    print("Loading data...")
    data_path = args.train_data
    val_path = args.val_data
    meta_path = args.meta_data
    meta_model = args.meta_model
    
    # Initialize fold results
    fold_models = []
    fold_results = {}
    best_val_loss = float('inf')
    best_model_path = None
    
    # Initialize a base model to get feature information
    print("Initializing base model to get feature information...")
    base_model = EfficientCategoricalModel(
        data_path_train=data_path,
        data_path_val=val_path,
        data_path_metadata=meta_path,
        data_path_meta_model=meta_model,
        batch_size=args.batch_size
    )
    base_model.data_handler.get_dataset_info()
    feature_set = base_model.data_handler.feature_set
    
    # Initialize K-fold cross-validation
    print(f"\nPerforming {args.n_folds}-fold cross-validation...")
    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    
    # Load data for K-fold split
    from src.data_handler import DataHandler
    data_handler = DataHandler(
        data_path_train=data_path,
        data_path_val=val_path,
        data_path_metadata=meta_path,
        batch_size=args.batch_size
    )
    
    # Load the train data
    train_data = data_handler.load_train_data()
    
    # Perform K-fold cross-validation
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_data)):
        print(f"\n{'='*50}")
        print(f"Training fold {fold+1}/{args.n_folds}")
        print(f"{'='*50}")
        
        # Split data for this fold
        fold_train = train_data.iloc[train_idx]
        fold_val = train_data.iloc[val_idx]
        
        # Create model name for this fold
        model_name = f"AdvancedModel_fold{fold}"
        
        # Create callbacks for this fold
        callbacks, log_dir = create_callbacks(model_name=model_name, use_advanced=True)
        
        # Initialize new model for this fold with improved model and optimizer
        efficient_model = EfficientCategoricalModel(
            data_path_train=data_path,
            data_path_val=val_path,
            data_path_metadata=meta_path,
            data_path_meta_model=meta_model,
            batch_size=args.batch_size,
            model=improved_model,
            optimizer=improved_optimizer,
            subset_features="medium"
        )
        
        # Set the feature_set directly from the base model
        efficient_model.data_handler._feature_set = feature_set
        efficient_model.data_handler.feature_count = len(feature_set)
        
        # Prepare data for training
        X_train = fold_train[feature_set].values
        y_train = fold_train['target'].values
        
        X_val = fold_val[feature_set].values
        y_val = fold_val['target'].values
        
        # Get the model from model manager
        model = efficient_model.model_manager.model
        
        # Compile model
        model.compile(
            optimizer=efficient_model.model_manager.optimizer,
            loss='mae',
            metrics=['mae', 'mse']
        )
        
        # Train the model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Store fold results
        fold_results[fold] = {
            'history': history.history,
            'model': model,
            'val_loss': min(history.history['val_loss'])
        }
        
        # Add model to ensemble
        fold_models.append(model)
        
        # Save model for this fold
        model_path = f"exports/{model_name}.keras"
        model.save(model_path)
        print(f"Model for fold {fold+1} saved to: {model_path}")
        
        # Check if this is the best model so far
        if fold_results[fold]['val_loss'] < best_val_loss:
            best_val_loss = fold_results[fold]['val_loss']
            best_model_path = model_path
            print(f"New best model: {model_path} with validation loss: {best_val_loss:.4f}")
    
    # Calculate average performance across folds
    avg_val_loss = np.mean([fold_results[fold]['val_loss'] for fold in range(args.n_folds)])
    print(f"\nAverage validation loss across all folds: {avg_val_loss:.4f}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best model saved to: {best_model_path}")
    
    print("\nAdvanced training complete!")
    print("Run TensorBoard to visualize training results for all folds:")
    print("tensorboard --logdir=logs/fit")
    
    return fold_models, fold_results, best_model_path

def train_correlation(args):
    """Train using a model specifically designed to learn feature correlations rather than distribution"""
    print("\n" + "="*50)
    print("Starting correlation-focused model training")
    print("="*50)
    
    # Create callbacks with advanced settings
    model_name = "CorrelationModel"
    callbacks, log_dir = create_callbacks(model_name=model_name, use_advanced=True)
    
    # Initialize model with the correlation model
    print("Initializing correlation model...")
    efficient_model = EfficientCategoricalModel(
        data_path_train=args.train_data,
        data_path_val=args.val_data,
        data_path_metadata=args.meta_data,
        data_path_meta_model=args.meta_model,
        batch_size=args.batch_size,
        model=correlation_model,
        optimizer=improved_optimizer,  # Using improved optimizer for better convergence
        subset_features="medium"  # Changed from "large" to "medium" to match available feature sets
    )
    
    # Train the model
    model, history = efficient_model.train(
        epochs=args.epochs,
        callbacks=callbacks
    )
    
    # Save the trained model
    model_path = f"exports/{model_name}_best.keras"
    model.save(model_path)
    print(f"Correlation model saved to: {model_path}")
    
    # Validate model if requested
    if args.validate:
        print("\nValidating correlation model...")
        efficient_model.validate_model()
        performance_metrics = efficient_model.performance_eval()
        print("\nPerformance Metrics:")
        for metric, value in performance_metrics.items():
            print(f"{metric}: {value}")
    
    # Analyze feature importance and correlations
    print("\nAnalyzing feature correlations...")
    
    # Get feature importances from the model
    try:
        # Prepare validation data
        val_data = efficient_model.data_handler.load_val_data()
        X_val = val_data[efficient_model.data_handler.feature_set].values
        y_val = val_data['target'].values
        
        # Make predictions
        y_pred = model.predict(X_val)
        
        # Compute feature correlations with target
        import pandas as pd
        feature_df = pd.DataFrame(X_val, columns=efficient_model.data_handler.feature_set)
        feature_df['target'] = y_val
        feature_df['prediction'] = y_pred.flatten()
        
        # Get top correlations with target
        target_corrs = feature_df.corr()['target'].sort_values(ascending=False)
        print("\nTop 10 features correlated with target:")
        print(target_corrs.head(11))  # +1 because target itself will be first
        
        # Get top correlations with prediction
        pred_corrs = feature_df.corr()['prediction'].sort_values(ascending=False)
        print("\nTop 10 features correlated with prediction:")
        print(pred_corrs.head(11))  # +1 because prediction itself will be first
    except Exception as e:
        print(f"Error analyzing correlations: {str(e)}")
    
    print("\nCorrelation model training complete!")
    print(f"TensorBoard logs saved to: {log_dir}")
    print("Run the following command to start TensorBoard:")
    print(f"tensorboard --logdir={log_dir}")
    
    return model, history, efficient_model

def main():
    """Main function to handle different training modes"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Efficient Categorical Model')
    parser.add_argument('--mode', type=str, default='standard', 
                        choices=['standard', 'advanced', 'correlation'],
                        help='Training mode: standard, advanced with k-fold, or correlation-focused')
    parser.add_argument('--train_data', type=str, default='data/train.parquet',
                        help='Path to training data')
    parser.add_argument('--val_data', type=str, default='data/validation.parquet',
                        help='Path to validation data')
    parser.add_argument('--meta_data', type=str, default='data/features.json',
                        help='Path to metadata')
    parser.add_argument('--meta_model', type=str, default='data/meta_model.parquet',
                        help='Path to meta model')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs to train')
    parser.add_argument('--n_folds', type=int, default=5,
                        help='Number of folds for cross-validation (advanced mode only)')
    parser.add_argument('--use_improved_model', action='store_true',
                        help='Use improved model architecture (standard mode only)')
    parser.add_argument('--use_improved_optimizer', action='store_true',
                        help='Use improved optimizer (standard mode only)')
    parser.add_argument('--validate', action='store_true',
                        help='Validate model after training')
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'standard':
            train_standard(args)
        elif args.mode == 'advanced':
            train_advanced(args)
        elif args.mode == 'correlation':
            train_correlation(args)
        else:
            print(f"Unknown mode: {args.mode}")
            print("Choose from 'standard', 'advanced', or 'correlation'")
            
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

