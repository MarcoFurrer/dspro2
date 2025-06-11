import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';

const ModelCreation = () => {
  const navigate = useNavigate();
  const [formData, setFormData] = useState({
    name: '',
    model_type: 'Base',
    optimizer: 'Adam',
    learning_rate: 0.001,
    batch_size: 32,
    epochs: 10,
    validation_split: 0.1,
    feature_set: 'small'
  });
  const [isCreating, setIsCreating] = useState(false);
  const [availableOptions, setAvailableOptions] = useState({
    model_types: [],
    optimizer_types: [],
    feature_sets: []
  });
  const [error, setError] = useState('');

  // Backend API URL
  const API_BASE_URL = 'http://localhost:5002/api';

  useEffect(() => {
    // Load available model types and optimizers from backend
    const loadAvailableOptions = async () => {
      try {
        const response = await axios.get(`${API_BASE_URL}/available-models`);
        if (response.data.status === 'success') {
          setAvailableOptions(response.data);
          // Set default model type if available
          if (response.data.model_types.length > 0) {
            setFormData(prev => ({
              ...prev,
              model_type: response.data.model_types[0]
            }));
          }
        }
      } catch (error) {
        console.error('Failed to load available options:', error);
        setError('Failed to connect to backend. Make sure the API server is running.');
      }
    };

    loadAvailableOptions();
  }, []);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: name === 'learning_rate' || name === 'validation_split' ? parseFloat(value) :
              name === 'batch_size' || name === 'epochs' ? parseInt(value) : value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsCreating(true);
    setError('');

    try {
      console.log('Creating model with config:', formData);
      
      const response = await axios.post(`${API_BASE_URL}/models`, formData);
      
      if (response.data.status === 'success') {
        console.log('Model creation started:', response.data);
        // Navigate to home with success message
        navigate('/', { 
          state: { 
            message: `Model "${formData.name}" is being trained! (ID: ${response.data.model_id})`,
            type: 'success'
          }
        });
      } else {
        setError(response.data.message || 'Failed to create model');
      }
    } catch (error) {
      console.error('Error creating model:', error);
      setError(error.response?.data?.message || 'Failed to create model. Check backend connection.');
    } finally {
      setIsCreating(false);
    }
  };

  return (
    <div className="page-container">
      <h1 className="page-title">Create New Model</h1>
      <p className="page-subtitle">
        Configure and train a new machine learning model with advanced settings
      </p>

      {error && (
        <div className="error-message">
          {error}
        </div>
      )}

      <form onSubmit={handleSubmit} className="mt-3">
        <div className="form-group">
          <label htmlFor="name" className="form-label">
            Model Name *
          </label>
          <input
            type="text"
            id="name"
            name="name"
            value={formData.name}
            onChange={handleInputChange}
            className="form-input"
            placeholder="Enter a descriptive name for your model"
            required
          />
        </div>

        <div className="form-group">
          <label htmlFor="model_type" className="form-label">
            Model Type
          </label>
          <select
            id="model_type"
            name="model_type"
            value={formData.model_type}
            onChange={handleInputChange}
            className="form-select"
          >
            {availableOptions.model_types.map(type => (
              <option key={type} value={type}>{type}</option>
            ))}
          </select>
        </div>

        <div className="form-group">
          <label htmlFor="optimizer" className="form-label">
            Optimizer
          </label>
          <select
            id="optimizer"
            name="optimizer"
            value={formData.optimizer}
            onChange={handleInputChange}
            className="form-select"
          >
            {availableOptions.optimizer_types.map(opt => (
              <option key={opt} value={opt}>{opt}</option>
            ))}
          </select>
        </div>

        <div className="form-group">
          <label htmlFor="feature_set" className="form-label">
            Feature Set
          </label>
          <select
            id="feature_set"
            name="feature_set"
            value={formData.feature_set}
            onChange={handleInputChange}
            className="form-select"
          >
            {availableOptions.feature_sets.map(set => (
              <option key={set} value={set}>{set.charAt(0).toUpperCase() + set.slice(1)}</option>
            ))}
          </select>
        </div>

        <div className="form-group">
          <label htmlFor="learning_rate" className="form-label">
            Learning Rate
          </label>
          <input
            type="number"
            id="learning_rate"
            name="learning_rate"
            value={formData.learning_rate}
            onChange={handleInputChange}
            className="form-input"
            step="0.0001"
            min="0.0001"
            max="1"
          />
        </div>

        <div className="form-group">
          <label htmlFor="batch_size" className="form-label">
            Batch Size
          </label>
          <input
            type="number"
            id="batch_size"
            name="batch_size"
            value={formData.batch_size}
            onChange={handleInputChange}
            className="form-input"
            min="1"
            max="512"
          />
        </div>

        <div className="form-group">
          <label htmlFor="epochs" className="form-label">
            Epochs
          </label>
          <input
            type="number"
            id="epochs"
            name="epochs"
            value={formData.epochs}
            onChange={handleInputChange}
            className="form-input"
            min="1"
            max="1000"
          />
        </div>

        <div className="form-group">
          <label htmlFor="validation_split" className="form-label">
            Validation Split
          </label>
          <input
            type="number"
            id="validation_split"
            name="validation_split"
            value={formData.validation_split}
            onChange={handleInputChange}
            className="form-input"
            step="0.01"
            min="0.01"
            max="0.5"
          />
        </div>

        <div className="form-actions">
          <button
            type="button"
            onClick={() => navigate('/')}
            className="btn btn-secondary"
            disabled={isCreating}
          >
            Cancel
          </button>
          <button
            type="submit"
            className="btn btn-primary"
            disabled={isCreating || !formData.name}
          >
            {isCreating ? (
              <>
                <span className="spinner"></span>
                Creating Model...
              </>
            ) : (
              'Create & Train Model'
            )}
          </button>
        </div>
      </form>

      {isCreating && (
        <div className="mt-3 text-center">
          <div className="loading">Training in progress... This may take a few minutes.</div>
        </div>
      )}
    </div>
  );
};

export default ModelCreation;