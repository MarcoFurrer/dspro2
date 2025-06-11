#!/usr/bin/env python3
"""
ML Model API Server
Provides REST endpoints for model creation, training, and management
"""

import os
import sys
import json
import uuid
import threading
from datetime import datetime
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import importlib

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from model_runner import ModelRunner

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Configuration
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'exports')
METADATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'metadata')
LOGS_DIR = os.path.join(os.path.dirname(__file__), '..', 'logs')

# Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(METADATA_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Track training jobs
training_jobs = {}

def load_model_class(model_name):
    """Load a model class by name"""
    try:
        module = importlib.import_module(f"models.{model_name}")
        return module.model
    except ImportError as e:
        raise ValueError(f"Model '{model_name}' not found: {e}")

def load_optimizer_class(optimizer_name):
    """Load an optimizer class by name"""
    try:
        module = importlib.import_module(f"optimizers.{optimizer_name}")
        return module.optimizer
    except ImportError as e:
        raise ValueError(f"Optimizer '{optimizer_name}' not found: {e}")

def generate_model_id():
    """Generate unique model ID"""
    return str(uuid.uuid4())[:8]

def save_model_metadata(model_id, metadata):
    """Save model metadata to JSON file"""
    metadata_path = os.path.join(METADATA_DIR, f"{model_id}.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

def load_model_metadata(model_id):
    """Load model metadata from JSON file"""
    metadata_path = os.path.join(METADATA_DIR, f"{model_id}.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            return json.load(f)
    return None

def get_all_models_metadata():
    """Get metadata for all models"""
    models = []
    for filename in os.listdir(METADATA_DIR):
        if filename.endswith('.json'):
            model_id = filename[:-5]  # Remove .json extension
            metadata = load_model_metadata(model_id)
            if metadata:
                models.append(metadata)
    return sorted(models, key=lambda x: x.get('created_at', ''), reverse=True)

def train_model_async(model_id, model_config):
    """Train model in background thread"""
    try:
        # Update status to training
        metadata = load_model_metadata(model_id)
        metadata['status'] = 'training'
        metadata['started_at'] = datetime.now().isoformat()
        save_model_metadata(model_id, metadata)
        
        # Load model and optimizer
        model_class = load_model_class(model_config['model_type'])
        optimizer_class = load_optimizer_class(model_config['optimizer'])
        
        # Compile model
        model_class.compile(
            optimizer=optimizer_class,
            loss=model_config.get('loss', 'mae'),
            metrics=['mae']
        )
        
        # Create model runner with custom output path
        output_path = os.path.join(MODELS_DIR, model_id)
        runner = ModelRunner(
            model=model_class,
            output_path=output_path,
            batch_size=model_config.get('batch_size', 32),
            subset_features=model_config.get('feature_set', 'small')
        )
        
        # Train model
        trained_model, history = runner.train(
            epochs=model_config.get('epochs', 10),
            validation_split=model_config.get('validation_split', 0.1)
        )
        
        # Run validation
        validation_results = runner.validate_model()
        performance_metrics = runner.performance_eval()
        
        # Update metadata with results
        metadata['status'] = 'trained'
        metadata['completed_at'] = datetime.now().isoformat()
        metadata['metrics'] = {
            'rmse': float(performance_metrics['mean']['CORR']) if performance_metrics is not None else 0.0,
            'mae': float(performance_metrics['std']['CORR']) if performance_metrics is not None else 0.0,
            'validation_samples': len(validation_results) if validation_results is not None else 0
        }
        metadata['model_path'] = os.path.join(output_path, 'model.keras')
        
        save_model_metadata(model_id, metadata)
        
        # Remove from active jobs
        if model_id in training_jobs:
            del training_jobs[model_id]
            
    except Exception as e:
        # Update status to error
        metadata = load_model_metadata(model_id)
        metadata['status'] = 'error'
        metadata['error'] = str(e)
        metadata['completed_at'] = datetime.now().isoformat()
        save_model_metadata(model_id, metadata)
        
        # Remove from active jobs
        if model_id in training_jobs:
            del training_jobs[model_id]

@app.route('/api/models', methods=['GET'])
def get_models():
    """Get all models with metadata"""
    try:
        models = get_all_models_metadata()
        return jsonify({
            'status': 'success',
            'models': models
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/models', methods=['POST'])
def create_model():
    """Create and train a new model"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['name', 'model_type', 'optimizer']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'status': 'error',
                    'message': f'Missing required field: {field}'
                }), 400
        
        # Generate unique model ID
        model_id = generate_model_id()
        
        # Create metadata
        metadata = {
            'id': model_id,
            'name': data['name'],
            'model_type': data['model_type'],
            'optimizer': data['optimizer'],
            'parameters': {
                'learning_rate': data.get('learning_rate', 0.001),
                'batch_size': data.get('batch_size', 32),
                'epochs': data.get('epochs', 10),
                'validation_split': data.get('validation_split', 0.1),
                'feature_set': data.get('feature_set', 'small')
            },
            'status': 'created',
            'created_at': datetime.now().isoformat(),
            'started_at': None,
            'completed_at': None,
            'metrics': {},
            'model_path': None
        }
        
        # Save metadata
        save_model_metadata(model_id, metadata)
        
        # Start training in background
        training_thread = threading.Thread(
            target=train_model_async,
            args=(model_id, data)
        )
        training_thread.daemon = True
        training_thread.start()
        
        # Track training job
        training_jobs[model_id] = {
            'thread': training_thread,
            'started_at': datetime.now().isoformat()
        }
        
        return jsonify({
            'status': 'success',
            'model_id': model_id,
            'message': 'Model training started'
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/models/<model_id>', methods=['GET'])
def get_model(model_id):
    """Get specific model metadata"""
    try:
        metadata = load_model_metadata(model_id)
        if not metadata:
            return jsonify({
                'status': 'error',
                'message': 'Model not found'
            }), 404
            
        return jsonify({
            'status': 'success',
            'model': metadata
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/models/<model_id>/status', methods=['GET'])
def get_model_status(model_id):
    """Get model training status"""
    try:
        metadata = load_model_metadata(model_id)
        if not metadata:
            return jsonify({
                'status': 'error',
                'message': 'Model not found'
            }), 404
            
        # Check if still training
        is_training = model_id in training_jobs
        
        return jsonify({
            'status': 'success',
            'model_id': model_id,
            'training_status': metadata['status'],
            'is_training': is_training,
            'created_at': metadata.get('created_at'),
            'started_at': metadata.get('started_at'),
            'completed_at': metadata.get('completed_at'),
            'metrics': metadata.get('metrics', {})
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/models/<model_id>', methods=['DELETE'])
def delete_model(model_id):
    """Delete a model and its metadata"""
    try:
        # Check if model exists
        metadata = load_model_metadata(model_id)
        if not metadata:
            return jsonify({
                'status': 'error',
                'message': 'Model not found'
            }), 404
            
        # Check if model is currently training
        if model_id in training_jobs:
            return jsonify({
                'status': 'error',
                'message': 'Cannot delete model while training is in progress'
            }), 400
            
        # Delete model file if it exists
        model_dir = os.path.join(MODELS_DIR, model_id)
        if os.path.exists(model_dir):
            import shutil
            shutil.rmtree(model_dir)
            
        # Delete metadata file
        metadata_path = os.path.join(METADATA_DIR, f"{model_id}.json")
        if os.path.exists(metadata_path):
            os.remove(metadata_path)
            
        return jsonify({
            'status': 'success',
            'message': 'Model deleted successfully'
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/available-models', methods=['GET'])
def get_available_models():
    """Get available model types and optimizers"""
    try:
        # Scan models directory
        models_dir = os.path.join(os.path.dirname(__file__), '..', 'src', 'models')
        model_types = []
        if os.path.exists(models_dir):
            for file in os.listdir(models_dir):
                if file.endswith('.py') and not file.startswith('__'):
                    model_types.append(file[:-3])  # Remove .py extension
        
        # Scan optimizers directory
        optimizers_dir = os.path.join(os.path.dirname(__file__), '..', 'src', 'optimizers')
        optimizer_types = []
        if os.path.exists(optimizers_dir):
            for file in os.listdir(optimizers_dir):
                if file.endswith('.py') and not file.startswith('__'):
                    optimizer_types.append(file[:-3])  # Remove .py extension
        
        return jsonify({
            'status': 'success',
            'model_types': model_types,
            'optimizer_types': optimizer_types,
            'feature_sets': ['small', 'medium', 'all']
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'success',
        'message': 'ML Model API Server is running',
        'timestamp': datetime.now().isoformat(),
        'active_training_jobs': len(training_jobs)
    })

if __name__ == '__main__':
    print("Starting ML Model API Server...")
    print(f"Models directory: {MODELS_DIR}")
    print(f"Metadata directory: {METADATA_DIR}")
    print(f"Server running on http://localhost:5002")
    
    app.run(
        host='0.0.0.0',
        port=5002,
        debug=True,
        threaded=True
    )