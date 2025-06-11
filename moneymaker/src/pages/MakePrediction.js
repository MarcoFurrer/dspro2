import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';

const MakePrediction = () => {
  const navigate = useNavigate();
  const [formData, setFormData] = useState({
    modelId: '',
    datasetFile: null,
    predictionName: '',
    confidenceInterval: 0.95
  });
  const [isProcessing, setIsProcessing] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const [availableModels, setAvailableModels] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Fetch trained models on component mount
  useEffect(() => {
    fetchTrainedModels();
  }, []);

  const fetchTrainedModels = async () => {
    try {
      const response = await fetch('http://localhost:5002/api/models/trained');
      const data = await response.json();
      
      if (data.status === 'success') {
        setAvailableModels(data.models);
      } else {
        setError('Failed to load trained models');
      }
    } catch (err) {
      setError('Failed to connect to server');
      console.error('Error fetching models:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    if (file && (file.name.endsWith('.parquet') || file.name.endsWith('.csv'))) {
      setFormData(prev => ({
        ...prev,
        datasetFile: file
      }));
    } else {
      alert('Please select a valid .parquet or .csv file');
    }
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    const file = e.dataTransfer.files[0];
    if (file && (file.name.endsWith('.parquet') || file.name.endsWith('.csv'))) {
      setFormData(prev => ({
        ...prev,
        datasetFile: file
      }));
    } else {
      alert('Please drop a valid .parquet or .csv file');
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsProcessing(true);
    setError(null);

    try {
      // Create FormData for file upload
      const formDataToSend = new FormData();
      formDataToSend.append('file', formData.datasetFile);
      formDataToSend.append('model_id', formData.modelId);
      formDataToSend.append('prediction_name', formData.predictionName);
      formDataToSend.append('confidence_interval', formData.confidenceInterval);

      const response = await fetch('http://localhost:5002/api/predict', {
        method: 'POST',
        body: formDataToSend
      });

      const data = await response.json();

      if (data.status === 'success') {
        // Navigate to predictions page with success message
        navigate('/predictions', { 
          state: { 
            message: 'Prediction started successfully! Results will appear when processing completes.',
            predictionId: data.prediction_id
          }
        });
      } else {
        setError(data.message || 'Failed to start prediction');
      }
    } catch (err) {
      setError('Failed to connect to server');
      console.error('Error making prediction:', err);
    } finally {
      setIsProcessing(false);
    }
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div className="page-container">
      <h1 className="page-title">Make New Prediction</h1>
      <p className="page-subtitle">
        Select a trained model and dataset to generate new predictions
      </p>

      {error && (
        <div className="mt-3" style={{
          padding: '1rem',
          backgroundColor: '#f8d7da',
          color: '#721c24',
          border: '1px solid #f5c6cb',
          borderRadius: '8px',
          marginBottom: '1rem'
        }}>
          {error}
        </div>
      )}

      {loading ? (
        <div className="text-center mt-3">
          <div className="loading">Loading trained models...</div>
        </div>
      ) : (
        <form onSubmit={handleSubmit} className="mt-3">
          <div className="form-group">
            <label htmlFor="predictionName" className="form-label">
              Prediction Name *
            </label>
            <input
              type="text"
              id="predictionName"
              name="predictionName"
              value={formData.predictionName}
              onChange={handleInputChange}
              className="form-input"
              placeholder="Enter a name for this prediction batch"
              required
            />
          </div>

          <div className="form-group">
            <label htmlFor="modelId" className="form-label">
              Select Model *
            </label>
            <select
              id="modelId"
              name="modelId"
              value={formData.modelId}
              onChange={handleInputChange}
              className="form-select"
              required
            >
              <option value="">Choose a trained model...</option>
              {availableModels.map(model => (
                <option key={model.id} value={model.id}>
                  {model.name} (Status: {model.status})
                </option>
              ))}
            </select>
            {availableModels.length === 0 && (
              <p style={{ color: '#666', marginTop: '0.5rem', fontSize: '0.9rem' }}>
                No trained models available. Please train a model first.
              </p>
            )}
          </div>

        <div className="form-group">
          <label className="form-label">
            Dataset File *
          </label>
          <div
            className={`form-file ${dragActive ? 'drag-active' : ''}`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            onClick={() => document.getElementById('datasetInput').click()}
            style={{
              borderColor: dragActive ? '#5a67d8' : '#667eea',
              backgroundColor: dragActive ? '#f0f2ff' : '#f8f9ff'
            }}
          >
            <input
              type="file"
              id="datasetInput"
              accept=".parquet,.csv"
              onChange={handleFileSelect}
              style={{ display: 'none' }}
            />
            
            <div style={{ fontSize: '2rem', marginBottom: '0.5rem' }}>📊</div>
            
            {formData.datasetFile ? (
              <div>
                <p style={{ fontSize: '1.1rem', fontWeight: 'bold', color: '#333' }}>
                  {formData.datasetFile.name}
                </p>
                <p style={{ color: '#666' }}>
                  Size: {formatFileSize(formData.datasetFile.size)}
                </p>
              </div>
            ) : (
              <div>
                <p style={{ fontSize: '1.1rem', fontWeight: 'bold', color: '#333' }}>
                  Drop your dataset here
                </p>
                <p style={{ color: '#666' }}>
                  Supports .parquet and .csv files
                </p>
              </div>
            )}
          </div>
        </div>

        <div className="form-group">
          <label htmlFor="confidenceInterval" className="form-label">
            Confidence Interval
          </label>
          <select
            id="confidenceInterval"
            name="confidenceInterval"
            value={formData.confidenceInterval}
            onChange={handleInputChange}
            className="form-select"
          >
            <option value={0.90}>90%</option>
            <option value={0.95}>95%</option>
            <option value={0.99}>99%</option>
          </select>
        </div>

        <div className="text-center">
          <button
            type="submit"
            className="btn btn-primary"
            disabled={isProcessing || !formData.modelId || !formData.datasetFile || !formData.predictionName.trim() || availableModels.length === 0}
            style={{ marginRight: '1rem' }}
          >
            {isProcessing ? 'Processing...' : 'Generate Predictions'}
          </button>
          <button
            type="button"
            className="btn btn-secondary"
            onClick={() => navigate('/predictions')}
          >
            Cancel
          </button>
        </div>
        </form>
      )}

      {isProcessing && (
        <div className="mt-3 text-center">
          <div className="loading">
            Generating predictions... This may take a few minutes depending on dataset size.
          </div>
        </div>
      )}

      <div className="mt-3">
        <h3 style={{ color: '#333', marginBottom: '1rem' }}>Dataset Requirements:</h3>
        <ul style={{ color: '#666', lineHeight: '1.6' }}>
          <li>File format: .parquet or .csv</li>
          <li>Maximum file size: 500MB</li>
          <li>Must contain the same features used to train the selected model</li>
          <li>No target column required (predictions only)</li>
        </ul>
      </div>
    </div>
  );
};

export default MakePrediction;
