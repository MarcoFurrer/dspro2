from src.EfficientCategoricalModel import EfficientCategoricalModel
from src.models.Deep import model as deep_model
from src.models.ImprovedModel import model as improved_model
from src.models.CorrelationModel import model as correlation_model
from src.optimizers.Adam import optimizer as adam_optimizer
from src.optimizers.ImprovedAdam import optimizer as improved_optimizer
from keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import datetime
import os
import argparse
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
import pandas as pd

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

def train_model(args):
    """Unified training function that handles all model types and training strategies"""
    print("\n" + "="*50)
    print(f"Starting model training: {args.model_type} mode with {args.training_mode}")
    print("="*50)
    
    # Model selection based on model_type
    models = {
        "deep": deep_model,
        "improved": improved_model,
        "correlation": correlation_model
    }
    
    # Optimizer selection
    optimizers = {
        "adam": adam_optimizer,
        "improved": improved_optimizer
    }
    
    # Feature subset selection
    feature_subsets = {
        "small": "small",
        "medium": "medium",
        "all": "all"
    }
    
    # Select model and optimizer based on args
    model = models.get(args.model_type)
    if not model:
        raise ValueError(f"Unknown model type: {args.model_type}. Choose from: {', '.join(models.keys())}")
    
    optimizer = optimizers.get(args.optimizer)
    if not optimizer:
        raise ValueError(f"Unknown optimizer: {args.optimizer}. Choose from: {', '.join(optimizers.keys())}")
    
    subset = feature_subsets.get(args.feature_subset)
    if not subset:
        raise ValueError(f"Unknown feature subset: {args.feature_subset}. Choose from: {', '.join(feature_subsets.keys())}")
    
    # Create model name based on configuration
    model_name = f"{args.model_type.capitalize()}Model"
    
    # Training mode selection
    if args.training_mode == "standard":
        return train_single_model(args, model, optimizer, subset, model_name)
    elif args.training_mode == "kfold":
        return train_with_kfold(args, model, optimizer, subset, model_name)
    else:
        raise ValueError(f"Unknown training mode: {args.training_mode}. Choose from: standard, kfold")

def train_single_model(args, model, optimizer, feature_subset, model_name):
    """Train a single model (no cross-validation)"""
    # Create callbacks based on model name and settings
    callbacks, log_dir = create_callbacks(model_name=model_name, use_advanced=args.use_advanced_callbacks)
    
    # Initialize model
    print(f"Initializing {model_name} model...")
    efficient_model = EfficientCategoricalModel(
        data_path_train=args.train_data,
        data_path_val=args.val_data,
        data_path_metadata=args.meta_data,
        data_path_meta_model=args.meta_model,
        batch_size=args.batch_size,
        model=model,
        optimizer=optimizer,
        subset_features=feature_subset
    )
    
    # Train the model
    model, history = efficient_model.train(
        epochs=args.epochs,
        callbacks=callbacks
    )
    
    # Save the trained model
    model_path = f"exports/{model_name}_best.keras"
    model.save(model_path)
    print(f"Model saved to: {model_path}")
    
    # Validate model if requested
    if args.validate:
        print("\nValidating model...")
        efficient_model.validate_model()
        performance_metrics = efficient_model.performance_eval()
        print("\nPerformance Metrics:")
        for metric, value in performance_metrics.items():
            print(f"{metric}: {value}")
        
        # Display prediction examples with continuous values
        display_prediction_examples(efficient_model)
    
    # Analyze feature importance if requested
    if args.analyze_features:
        analyze_feature_correlations(efficient_model)
    
    print(f"\n{model_name} training complete!")
    print(f"TensorBoard logs saved to: {log_dir}")
    print("Run the following command to start TensorBoard:")
    print(f"tensorboard --logdir={log_dir}")
    
    return model, history, efficient_model

def train_with_kfold(args, model, optimizer, feature_subset, model_name):
    """Train using K-fold cross-validation and ensemble methods"""
    print(f"\nPerforming {args.n_folds}-fold cross-validation...")
    
    # Initialize fold results
    fold_models = []
    fold_results = {}
    best_val_loss = float('inf')
    best_model_path = None
    
    # Initialize a base model to get feature information
    print("Initializing base model to get feature information...")
    base_model = EfficientCategoricalModel(
        data_path_train=args.train_data,
        data_path_val=args.val_data,
        data_path_metadata=args.meta_data,
        data_path_meta_model=args.meta_model,
        batch_size=args.batch_size
    )
    base_model.data_handler.get_dataset_info()
    feature_set = base_model.data_handler.feature_set
    
    # Initialize K-fold cross-validation
    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    
    # Load data for K-fold split
    from src.data_handler import DataHandler
    data_handler = DataHandler(
        data_path_train=args.train_data,
        data_path_val=args.val_data,
        data_path_metadata=args.meta_data,
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
        fold_model_name = f"{model_name}_fold{fold}"
        
        # Create callbacks for this fold
        callbacks, log_dir = create_callbacks(model_name=fold_model_name, use_advanced=True)
        
        # Initialize new model for this fold
        efficient_model = EfficientCategoricalModel(
            data_path_train=args.train_data,
            data_path_val=args.val_data,
            data_path_metadata=args.meta_data,
            data_path_meta_model=args.meta_model,
            batch_size=args.batch_size,
            model=model,
            optimizer=optimizer,
            subset_features=feature_subset
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
        fold_model = efficient_model.model_manager.model
        
        # Compile model
        fold_model.compile(
            optimizer=efficient_model.model_manager.optimizer,
            loss='mae',
            metrics=['mae', 'mse']
        )
        
        # Train the model
        history = fold_model.fit(
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
            'model': fold_model,
            'val_loss': min(history.history['val_loss'])
        }
        
        # Add model to ensemble
        fold_models.append(fold_model)
        
        # Save model for this fold
        model_path = f"exports/{fold_model_name}.keras"
        fold_model.save(model_path)
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
    
    print("\nCross-validation training complete!")
    print("Run TensorBoard to visualize training results for all folds:")
    print("tensorboard --logdir=logs/fit")
    
    return fold_models, fold_results, best_model_path

def analyze_feature_correlations(efficient_model):
    """Analyze feature importance and correlations"""
    print("\nAnalyzing feature correlations...")
    
    try:
        # Prepare validation data
        val_data = efficient_model.data_handler.load_val_data()
        X_val = val_data[efficient_model.data_handler.feature_set].values
        y_val = val_data['target'].values
        
        # Make predictions
        y_pred = efficient_model.model_manager.model.predict(X_val)
        
        # Compute feature correlations with target
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

def display_prediction_examples(efficient_model, num_examples=10):
    """Display examples of predictions and targets with continuous values"""
    print("\n" + "="*50)
    print("Prediction Examples (Continuous Values)")
    print("="*50)
    
    try:
        # Prepare validation data
        val_data = efficient_model.data_handler.load_val_data()
        X_val = val_data[efficient_model.data_handler.feature_set].values
        y_val = val_data['target'].values
        
        # Get a sample of indices to display
        np.random.seed(42)  # For reproducibility
        sample_indices = np.random.choice(len(y_val), size=min(num_examples, len(y_val)), replace=False)
        
        # Make predictions for these samples
        X_sample = X_val[sample_indices]
        y_sample = y_val[sample_indices]
        y_pred_sample = efficient_model.model_manager.model.predict(X_sample).flatten()
        
        # Create a DataFrame to display the results nicely
        results_df = pd.DataFrame({
            'Example': range(1, len(sample_indices) + 1),
            'True Value': y_sample,
            'Prediction (Continuous)': y_pred_sample,
            'Absolute Error': np.abs(y_sample - y_pred_sample)
        })
        
        # Add some analysis
        results_df['Prediction (Rounded)'] = np.round(results_df['Prediction (Continuous)'] * 4) / 4
        results_df['Correct?'] = results_df['Prediction (Rounded)'] == results_df['True Value']
        
        # Display the results
        pd.set_option('display.precision', 4)
        print(results_df)
        
        # Calculate and display summary statistics
        mae = np.mean(np.abs(y_sample - y_pred_sample))
        accuracy = np.mean(results_df['Correct?'])
        
        print("\nSummary Statistics:")
        print(f"Mean Absolute Error: {mae:.4f}")
        print(f"Accuracy (after rounding): {accuracy:.2%}")
        
        # Reset display options
        pd.reset_option('display.precision')
        
    except Exception as e:
        print(f"Error displaying prediction examples: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    """Main function to handle different training modes"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Efficient Categorical Model')
    
    # Model configuration
    parser.add_argument('--model_type', type=str, default='deep', 
                        choices=['deep', 'improved', 'correlation'],
                        help='Model architecture to use')
    parser.add_argument('--optimizer', type=str, default='adam', 
                        choices=['adam', 'improved'],
                        help='Optimizer to use for training')
    parser.add_argument('--feature_subset', type=str, default='medium', 
                        choices=['small', 'medium', 'all'],
                        help='Size of feature subset to use')
    
    # Training configuration
    parser.add_argument('--training_mode', type=str, default='standard',
                        choices=['standard', 'kfold'],
                        help='Training mode: standard (single model) or kfold (cross-validation)')
    
    # Data paths
    parser.add_argument('--train_data', type=str, default='data/train.parquet',
                        help='Path to training data')
    parser.add_argument('--val_data', type=str, default='data/validation.parquet',
                        help='Path to validation data')
    parser.add_argument('--meta_data', type=str, default='data/features.json',
                        help='Path to metadata')
    parser.add_argument('--meta_model', type=str, default='data/meta_model.parquet',
                        help='Path to meta model')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs to train')
    parser.add_argument('--n_folds', type=int, default=5,
                        help='Number of folds for cross-validation (kfold mode only)')
    parser.add_argument('--use_advanced_callbacks', action='store_true',
                        help='Use advanced callbacks (early stopping, LR reduction, etc.)')
    
    # Evaluation and analysis
    parser.add_argument('--validate', action='store_true',
                        help='Validate model after training')
    parser.add_argument('--analyze_features', action='store_true',
                        help='Analyze feature correlations after training')
    
    # For backward compatibility with old command line format
    parser.add_argument('--mode', type=str, 
                        choices=['standard', 'advanced', 'correlation'],
                        help='Legacy mode parameter (use --model_type and --training_mode instead)')
    parser.add_argument('--use_improved_model', action='store_true',
                        help='Legacy parameter (use --model_type improved instead)')
    parser.add_argument('--use_improved_optimizer', action='store_true',
                        help='Legacy parameter (use --optimizer improved instead)')
    
    args = parser.parse_args()
    
    # Handle legacy parameters for backward compatibility
    if args.mode:
        if args.mode == 'standard':
            args.model_type = 'deep' if not args.use_improved_model else 'improved'
            args.optimizer = 'adam' if not args.use_improved_optimizer else 'improved'
            args.training_mode = 'standard'
        elif args.mode == 'advanced':
            args.model_type = 'improved'
            args.optimizer = 'improved'
            args.training_mode = 'kfold'
            args.use_advanced_callbacks = True
        elif args.mode == 'correlation':
            args.model_type = 'correlation'
            args.optimizer = 'improved'
            args.training_mode = 'standard'
            args.use_advanced_callbacks = True
    
    try:
        # Train the model with unified approach
        train_model(args)
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

