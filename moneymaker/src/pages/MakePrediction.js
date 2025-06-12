import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';

const MakePrediction = () => {
  const navigate = useNavigate();
  const [formData, setFormData] = useState({
    modelId: '',
    datasetFile: null
  });
  const [isProcessing, setIsProcessing] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const [availableModels, setAvailableModels] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [predictionResults, setPredictionResults] = useState(null);

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
      setError(null);
      setPredictionResults(null);
    } else {
      setError('Please select a valid .parquet or .csv file');
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
      setError(null);
      setPredictionResults(null);
    } else {
      setError('Please drop a valid .parquet or .csv file');
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsProcessing(true);
    setError(null);
    setPredictionResults(null);

    try {
      const formDataToSend = new FormData();
      formDataToSend.append('file', formData.datasetFile);
      formDataToSend.append('model_id', formData.modelId);
      
      // Generate prediction name from dataset filename
      const datasetName = formData.datasetFile.name.replace(/\.[^/.]+$/, ''); // Remove extension
      formDataToSend.append('prediction_name', datasetName);

      const response = await fetch('http://localhost:5002/api/predict', {
        method: 'POST',
        body: formDataToSend,
      });

      if (response.ok) {
        // Get the filename from the response headers
        const contentDisposition = response.headers.get('Content-Disposition');
        const filename = contentDisposition 
          ? contentDisposition.split('filename=')[1].replace(/"/g, '')
          : 'predictions.parquet';

        // Create blob and download
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.style.display = 'none';
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);

        // Show prediction results
        await showPredictionStats(blob, filename);

      } else {
        const errorData = await response.json();
        setError(errorData.message || 'Failed to generate predictions');
      }
    } catch (error) {
      console.error('Error making prediction:', error);
      setError('Failed to generate predictions. Please check your connection.');
    } finally {
      setIsProcessing(false);
    }
  };

  const showPredictionStats = async (blob, filename) => {
    try {
      const selectedModel = availableModels.find(m => m.id === formData.modelId);
      
      setPredictionResults({
        filename: filename,
        fileSize: formatFileSize(blob.size),
        downloadTime: new Date().toLocaleString(),
        model: selectedModel?.name || 'Unknown Model',
        datasetName: formData.datasetFile.name,
        datasetSize: formatFileSize(formData.datasetFile.size),
        success: true
      });
    } catch (error) {
      console.error('Error generating stats:', error);
    }
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const resetForm = () => {
    setFormData({
      modelId: '',
      datasetFile: null
    });
    setPredictionResults(null);
    setError(null);
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

        <div className="text-center">
          <button
            type="submit"
            className="btn btn-primary"
            disabled={isProcessing || !formData.modelId || !formData.datasetFile || availableModels.length === 0}
            style={{ marginRight: '1rem' }}
          >
            {isProcessing ? 'Processing...' : 'Generate Predictions'}
          </button>
          <button
            type="button"
            className="btn btn-secondary"
            onClick={resetForm}
            disabled={isProcessing}
          >
            Reset
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

      {predictionResults && (
        <div className="mt-3" style={{
          padding: '1.5rem',
          backgroundColor: '#d4edda',
          color: '#155724',
          border: '1px solid #c3e6cb',
          borderRadius: '8px'
        }}>
          <h3 style={{ color: '#155724', marginBottom: '1rem', textAlign: 'center' }}>
            ✅ Prediction Complete!
          </h3>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
            <div>
              <p><strong>Download File:</strong> {predictionResults.filename}</p>
              <p><strong>File Size:</strong> {predictionResults.fileSize}</p>
              <p><strong>Generated:</strong> {predictionResults.downloadTime}</p>
            </div>
            <div>
              <p><strong>Model Used:</strong> {predictionResults.model}</p>
              <p><strong>Dataset:</strong> {predictionResults.datasetName}</p>
              <p><strong>Dataset Size:</strong> {predictionResults.datasetSize}</p>
            </div>
          </div>
          <div className="text-center mt-3">
            <button
              type="button"
              className="btn btn-primary"
              onClick={resetForm}
              style={{ marginRight: '1rem' }}
            >
              Make Another Prediction
            </button>
            <button
              type="button"
              className="btn btn-secondary"
              onClick={() => navigate('/')}
            >
              Back to Home
            </button>
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
