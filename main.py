#!/usr/bin/env python3
"""
Simple Main Training Script

Usage:
    python main.py --model base --optimizer adam
    python main.py --model advanced --optimizer sgd
"""

import os
import sys
import argparse
import importlib

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from model_runner import ModelRunner

def load_model(model_name):
    """Load a model by name"""
    module = importlib.import_module(f"models.{model_name}")
    return module.model

def load_optimizer(optimizer_name):
    """Load an optimizer by name"""
    module = importlib.import_module(f"optimizers.{optimizer_name}")
    return module.optimizer

def main():
    parser = argparse.ArgumentParser(description="Train model with selected architecture and optimizer")
    parser.add_argument('--model', type=str, default='Base', help='Model to use (e.g., Base, Advanced, Deep)')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer to use (e.g., Adam, SGD, RMSprop)')
    
    args = parser.parse_args()
    
    print(f"Loading model: {args.model}")
    print(f"Loading optimizer: {args.optimizer}")
    
    # Load model and optimizer
    model = load_model(args.model)
    optimizer = load_optimizer(args.optimizer)
    
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Create model runner
    runner = ModelRunner(model=model)
    
    # Train
    print("Starting training...")
    model, history = runner.train(epochs=10)
    
    # Validate
    print("Running validation...")
    validation_results = runner.validate_model()
    
    # Evaluate performance
    print("Evaluating performance...")
    runner.performance_eval()
    
    print("Training complete!")

if __name__ == "__main__":
    main()