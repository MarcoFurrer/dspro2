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
        print(f"Serving {len(models)} models to client")
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

@app.route('/api/models/trained', methods=['GET'])
def get_trained_models():
    """Get only successfully trained models"""
    try:
        all_models = get_all_models_metadata()
        trained_models = [model for model in all_models if model['status'] == 'trained']
        return jsonify({
            'status': 'success',
            'models': trained_models
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/predict', methods=['POST'])
def make_prediction():
    """Make predictions on uploaded dataset and return as parquet download"""
    try:
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({
                'status': 'error',
                'message': 'No file uploaded'
            }), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                'status': 'error',
                'message': 'No file selected'
            }), 400
        
        # Get form data
        model_id = request.form.get('model_id')
        prediction_name = request.form.get('prediction_name', 'predictions')
        
        if not model_id:
            return jsonify({
                'status': 'error',
                'message': 'Missing required field: model_id'
            }), 400
        
        # Load model metadata
        metadata_path = os.path.join(METADATA_DIR, f"{model_id}.json")
        if not os.path.exists(metadata_path):
            return jsonify({
                'status': 'error',
                'message': f'Model {model_id} not found'
            }), 404
        
        with open(metadata_path, 'r') as f:
            model_metadata = json.load(f)
        
        if model_metadata['status'] != 'trained':
            return jsonify({
                'status': 'error',
                'message': f'Model {model_id} is not trained yet. Status: {model_metadata["status"]}'
            }), 400
        
        # Check if there was a training error
        if 'error' in model_metadata and model_metadata['error']:
            return jsonify({
                'status': 'error',
                'message': f'Model {model_id} had training errors: {model_metadata["error"]}'
            }), 400
        
        # Find the model file - check both metadata path and exports directory
        model_path = model_metadata.get('model_path')
        
        if not model_path or not os.path.exists(model_path):
            # Try to find the model file in the exports directory
            model_dir = os.path.join(MODELS_DIR, model_id)
            if os.path.exists(model_dir):
                # Look for .keras files in the model directory
                keras_files = [f for f in os.listdir(model_dir) if f.endswith('.keras')]
                if keras_files:
                    model_path = os.path.join(model_dir, keras_files[0])
                    print(f"Found model file: {model_path}")
                else:
                    return jsonify({
                        'status': 'error',
                        'message': f'No .keras model file found in {model_dir}'
                    }), 404
            else:
                return jsonify({
                    'status': 'error',
                    'message': f'Model directory not found: {model_dir}'
                }), 404
        
        if not os.path.exists(model_path):
            return jsonify({
                'status': 'error',
                'message': f'Model file not found: {model_path}'
            }), 404
        
        # Load model using TensorFlow
        import tensorflow as tf
        import pandas as pd
        import numpy as np
        
        try:
            print(f"Loading model from: {model_path}")
            model = tf.keras.models.load_model(model_path, safe_mode=False)
            print(f"Model loaded successfully. Input shape: {model.input_shape}")
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': f'Failed to load model: {str(e)}'
            }), 500
        
        # Read uploaded file
        try:
            if file.filename.endswith('.parquet'):
                df = pd.read_parquet(file)
            elif file.filename.endswith('.csv'):
                df = pd.read_csv(file)
            else:
                return jsonify({
                    'status': 'error',
                    'message': 'Unsupported file format. Use .parquet or .csv'
                }), 400
            
            print(f"Dataset loaded: {len(df)} rows, {len(df.columns)} columns")
            
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': f'Failed to read uploaded file: {str(e)}'
            }), 400
        
        # Load feature metadata
        try:
            features_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'features.json')
            with open(features_path, 'r') as f:
                feature_set = json.load(f)
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': f'Failed to load feature metadata: {str(e)}'
            }), 500
        
        # Use the same feature set as the model
        subset_features = model_metadata['parameters'].get('feature_set', 'small')
        
        if subset_features not in feature_set['feature_sets']:
            return jsonify({
                'status': 'error',
                'message': f'Feature set "{subset_features}" not found in feature metadata'
            }), 500
            
        feature_names = feature_set['feature_sets'][subset_features]
        print(f"Using feature set '{subset_features}' with {len(feature_names)} features")
        
        # Check if required features exist
        missing_features = [f for f in feature_names if f not in df.columns]
        if missing_features:
            return jsonify({
                'status': 'error',
                'message': f'Missing required features: {missing_features[:10]}... (showing first 10 of {len(missing_features)} missing)'
            }), 400
        
        # Extract features and make predictions
        try:
            X = df[feature_names].values.astype(np.float32)
            print(f"Feature matrix shape: {X.shape}")
            
            # Check for NaN values
            if np.isnan(X).any():
                nan_count = np.isnan(X).sum()
                print(f"Warning: Found {nan_count} NaN values in features, filling with 0")
                X = np.nan_to_num(X, nan=0.0)
            
            predictions = model.predict(X, batch_size=32, verbose=0)
            predictions = predictions.squeeze()
            print(f"Predictions shape: {predictions.shape}")
            
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': f'Failed to make predictions: {str(e)}'
            }), 500
        
        # Create predictions DataFrame
        try:
            predictions_df = pd.DataFrame({
                'prediction': predictions
            })
            
            # Add original index if it exists
            if hasattr(df, 'index'):
                predictions_df.index = df.index
            
            print(f"Created predictions DataFrame with {len(predictions_df)} rows")
            
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': f'Failed to create predictions DataFrame: {str(e)}'
            }), 500
        
        # Save to temporary parquet file
        try:
            import tempfile
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.parquet')
            predictions_df.to_parquet(temp_file.name)
            
            # Generate filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{prediction_name}_{model_id}_{timestamp}.parquet"
            
            print(f"Predictions saved to temporary file: {temp_file.name}")
            print(f"Download filename: {filename}")
            
            return send_file(
                temp_file.name,
                as_attachment=True,
                download_name=filename,
                mimetype='application/octet-stream'
            )
            
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': f'Failed to create download file: {str(e)}'
            }), 500
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Prediction error: {str(e)}")
        print(f"Error details: {error_details}")
        
        return jsonify({
            'status': 'error',
            'message': f'Prediction failed: {str(e)}',
            'details': error_details if app.debug else None
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