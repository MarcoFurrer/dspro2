import React, { useState, useEffect, useRef } from 'react';
import { useLocation, Link } from 'react-router-dom';
import axios from 'axios';

const Home = () => {
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [successMessage, setSuccessMessage] = useState('');
  const location = useLocation();
  const intervalRef = useRef(null);

  // Backend API URL
  const API_BASE_URL = 'http://localhost:5002/api';

  useEffect(() => {
    loadModels();
    
    // Check for success message from navigation
    if (location.state?.message) {
      setSuccessMessage(location.state.message);
      setTimeout(() => {
        setSuccessMessage('');
        // Clear the message from history
        window.history.replaceState({}, document.title);
      }, 5000);
    }
    
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [location.state]);

  const loadModels = async () => {
    try {
      setLoading(true);
      setError('');
      
      const response = await axios.get(`${API_BASE_URL}/models`);
      
      if (response.data.status === 'success') {
        setModels(response.data.models);
        
        // Clear existing interval
        if (intervalRef.current) {
          clearInterval(intervalRef.current);
        }
        
        // Set up auto-refresh only if there are training models
        const hasTrainingModels = response.data.models.some(model => model.status === 'training');
        if (hasTrainingModels) {
          intervalRef.current = setInterval(() => {
            loadModels();
          }, 5000);
        }
      } else {
        setError('Failed to load models');
      }
    } catch (error) {
      console.error('Error loading models:', error);
      setError('Failed to connect to backend. Make sure the API server is running.');
      
      // Fallback to demo data if backend is not available
      setModels([
        {
          id: 'demo-1',
          name: 'Demo Advanced Neural Network v1',
          model_type: 'Advanced',
          optimizer: 'Adam',
          status: 'trained',
          created_at: '2024-01-15T10:30:00',
          metrics: {
            rmse: 0.2156,
            mae: 0.1834
          }
        },
        {
          id: 'demo-2', 
          name: 'Demo Deep Learning Model v2',
          model_type: 'Deep',
          optimizer: 'RMSprop',
          status: 'training',
          created_at: '2024-01-20T14:22:00',
          metrics: {
            rmse: 0.1834,
            mae: 0.1456
          }
        }
      ]);
    } finally {
      setLoading(false);
    }
  };

  const getStatusBadge = (status) => {
    switch (status) {
      case 'trained':
        return <span className="status-badge status-success">Trained</span>;
      case 'training':
        return <span className="status-badge status-warning">Training</span>;
      case 'error':
        return <span className="status-badge status-error">Error</span>;
      default:
        return <span className="status-badge status-warning">Created</span>;
    }
  };

  const getPerformanceColor = (rmse) => {
    if (rmse === undefined || rmse === null) return '#6b7280'; // Gray for undefined
    if (rmse < 0.15) return '#10b981'; // Green for excellent
    if (rmse < 0.25) return '#f59e0b'; // Yellow for good
    return '#ef4444'; // Red for needs improvement
  };

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleDateString();
  };

  const deleteModel = async (modelId) => {
    if (!window.confirm('Are you sure you want to delete this model?')) {
      return;
    }
    
    try {
      await axios.delete(`${API_BASE_URL}/models/${modelId}`);
      setModels(models.filter(model => model.id !== modelId));
      setSuccessMessage('Model deleted successfully');
      setTimeout(() => setSuccessMessage(''), 3000);
    } catch (error) {
      console.error('Error deleting model:', error);
      setError('Failed to delete model');
    }
  };

  if (loading) {
    return (
      <div className="page-container">
        <div className="loading">Loading models...</div>
      </div>
    );
  }

  return (
    <div className="page-container">
      <h1 className="page-title">ML Model Dashboard</h1>
      <p className="page-subtitle">
        Monitor and manage your machine learning models with real-time performance metrics
      </p>

      {successMessage && (
        <div className="alert alert-success">
          {successMessage}
        </div>
      )}

      {error && (
        <div className="alert alert-error">
          {error}
        </div>
      )}

      <div className="card-grid">
        {models.map((model) => (
          <div key={model.id} className="card">
            <div className="card-header">
              <div className="card-title">{model.name}</div>
              <button 
                className="btn-delete"
                onClick={() => deleteModel(model.id)}
                title="Delete model"
              >
                ×
              </button>
            </div>
            
            <div className="card-details">
              <p><strong>Type:</strong> {model.model_type}</p>
              <p><strong>Optimizer:</strong> {model.optimizer}</p>
              
              {model.metrics && model.metrics.rmse !== undefined && model.metrics.mae !== undefined ? (
                <>
                  <p><strong>RMSE:</strong> 
                    <span style={{ color: getPerformanceColor(model.metrics.rmse) }}>
                      {model.metrics.rmse.toFixed(4)}
                    </span>
                  </p>
                  <p><strong>MAE:</strong> {model.metrics.mae.toFixed(4)}</p>
                </>
              ) : model.status === 'training' ? (
                <p><strong>Status:</strong> Training in progress...</p>
              ) : model.status === 'error' ? (
                <p><strong>Status:</strong> Training failed</p>
              ) : (
                <p><strong>Status:</strong> Waiting for training to complete</p>
              )}
              
              <p><strong>Created:</strong> {formatDate(model.created_at)}</p>
              
              <div className="mt-2">
                {getStatusBadge(model.status)}
              </div>
            </div>
          </div>
        ))}
      </div>

      {models.length === 0 && !loading && (
        <div className="text-center mt-3">
          <p>No models found. Start by creating your first model!</p>
          <Link to="/create" className="btn btn-primary mt-2">
            Create New Model
          </Link>
        </div>
      )}
    </div>
  );
};

export default Home;