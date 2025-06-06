import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';

const ModelCreation = () => {
  const navigate = useNavigate();
  const [formData, setFormData] = useState({
    modelName: '',
    optimizer: 'Adam',
    learningRate: 0.001,
    batchSize: 32,
    epochs: 100,
    validationSplit: 0.2
  });
  const [isCreating, setIsCreating] = useState(false);

  const optimizers = [
    'Adam',
    'RMSprop', 
    'SGD',
    'Adagrad',
    'Adadelta',
    'Adamax',
    'Nadam'
  ];

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsCreating(true);

    // Simulate model creation process
    setTimeout(() => {
      console.log('Creating model with config:', formData);
      setIsCreating(false);
      // Navigate back to home or show success message
      navigate('/');
    }, 3000);
  };

  return (
    <div className="page-container">
      <h1 className="page-title">Create New Model</h1>
      <p className="page-subtitle">
        Configure and train a new machine learning model with advanced settings
      </p>

      <form onSubmit={handleSubmit} className="mt-3">
        <div className="form-group">
          <label htmlFor="modelName" className="form-label">
            Model Name *
          </label>
          <input
            type="text"
            id="modelName"
            name="modelName"
            value={formData.modelName}
            onChange={handleInputChange}
            className="form-input"
            placeholder="Enter a descriptive name for your model"
            required
          />
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
            {optimizers.map(opt => (
              <option key={opt} value={opt}>{opt}</option>
            ))}
          </select>
        </div>

        <div className="form-group">
          <label htmlFor="learningRate" className="form-label">
            Learning Rate
          </label>
          <input
            type="number"
            id="learningRate"
            name="learningRate"
            value={formData.learningRate}
            onChange={handleInputChange}
            className="form-input"
            step="0.0001"
            min="0.0001"
            max="1"
          />
        </div>

        <div className="form-group">
          <label htmlFor="batchSize" className="form-label">
            Batch Size
          </label>
          <input
            type="number"
            id="batchSize"
            name="batchSize"
            value={formData.batchSize}
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
          <label htmlFor="validationSplit" className="form-label">
            Validation Split
          </label>
          <input
            type="number"
            id="validationSplit"
            name="validationSplit"
            value={formData.validationSplit}
            onChange={handleInputChange}
            className="form-input"
            step="0.01"
            min="0.1"
            max="0.5"
          />
        </div>

        <div className="text-center">
          <button
            type="submit"
            className="btn btn-primary"
            disabled={isCreating || !formData.modelName.trim()}
          >
            {isCreating ? 'Creating Model...' : 'Create Model'}
          </button>
          <button
            type="button"
            className="btn btn-secondary"
            style={{ marginLeft: '1rem' }}
            onClick={() => navigate('/')}
          >
            Cancel
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