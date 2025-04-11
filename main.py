import argparse
import os
import sys
import tensorflow as tf
from keras.models import load_model
from keras.optimizers import Adam, RMSprop, SGD
from src.EfficientCategoricalModel import EfficientCategoricalModel

def load_model_from_path(model_path):
    """Load a Keras model from the specified path"""
    if not model_path:
        return None
    
    if os.path.exists(model_path):
        try:
            print(f"Loading model from {model_path}")
            return load_model(model_path)
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return None
    else:
        print(f"Model path {model_path} does not exist.")
        return None

def get_optimizer(optimizer_name, learning_rate=0.001):
    """Return the requested optimizer instance"""
    optimizer_name = optimizer_name.lower()
    
    if optimizer_name == 'adam':
        return Adam(learning_rate=learning_rate)
    elif optimizer_name == 'rmsprop':
        return RMSprop(learning_rate=learning_rate)
    elif optimizer_name == 'sgd':
        return SGD(learning_rate=learning_rate)
    else:
        print(f"Unknown optimizer: {optimizer_name}, using Adam as default")
        return Adam(learning_rate=learning_rate)

def main():
    # Define command line arguments
    parser = argparse.ArgumentParser(description="Train a categorical model with specified parameters")
    
    # Data paths
    parser.add_argument('--data_path', type=str, default='data/train.parquet',
                        help='Path to the training data Parquet file')
    parser.add_argument('--output_path', type=str, default='models/exports',
                        help='Path to save the trained models')
    
    # Model selection
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to pre-trained model (if using existing model)')
    
    # Training parameters
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'rmsprop', 'sgd'],
                        help='Optimizer to use for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate for the optimizer')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of epochs to train for')
    parser.add_argument('--validation_split', type=float, default=0.1,
                        help='Fraction of data to use for validation')
    
    # Parse arguments
    args = parser.parse_args()
    
    try:
        # Load model if specified
        model = load_model_from_path(args.model_path)
        
        # Create optimizer
        optimizer = get_optimizer(args.optimizer, args.learning_rate)
        
        # If model exists, recompile with the selected optimizer
        if model is not None:
            print(f"Recompiling model with {args.optimizer} optimizer")
            model.compile(
                optimizer=optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
        
        # Initialize the model
        print(f"Initializing EfficientCategoricalModel with batch size {args.batch_size}")
        efficient_model = EfficientCategoricalModel(
            data_path=args.data_path,
            output_path=args.output_path,
            batch_size=args.batch_size,
            model=model
        )
        
        # Train the model
        print(f"Training for {args.epochs} epochs with validation split {args.validation_split}")
        trained_model, history = efficient_model.train(
            validation_split=args.validation_split,
            epochs=args.epochs
        )
        
        print("Training complete!")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())


