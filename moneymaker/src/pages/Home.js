import React, { useState, useEffect } from 'react';

const Home = () => {
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Simulate loading models data
    // In a real app, this would fetch from an API
    setTimeout(() => {
      setModels([
        {
          id: 1,
          name: 'Advanced Neural Network v1',
          optimizer: 'Adam',
          performanceScore: 0.8947,
          rmse: 0.2156,
          mae: 0.1834,
          createdAt: '2024-01-15',
          status: 'trained'
        },
        {
          id: 2,
          name: 'Deep Learning Model v2',
          optimizer: 'RMSprop',
          performanceScore: 0.9123,
          rmse: 0.1834,
          mae: 0.1456,
          createdAt: '2024-01-20',
          status: 'trained'
        },
        {
          id: 3,
          name: 'CNN Classifier',
          optimizer: 'SGD',
          performanceScore: 0.8756,
          rmse: 0.2401,
          mae: 0.1923,
          createdAt: '2024-01-22',
          status: 'training'
        },
        {
          id: 4,
          name: 'LSTM Time Series',
          optimizer: 'Adam',
          performanceScore: 0.9345,
          rmse: 0.1567,
          mae: 0.1234,
          createdAt: '2024-01-25',
          status: 'trained'
        }
      ]);
      setLoading(false);
    }, 1000);
  }, []);

  const getStatusBadge = (status) => {
    switch (status) {
      case 'trained':
        return <span className="status-badge status-success">Trained</span>;
      case 'training':
        return <span className="status-badge status-warning">Training</span>;
      case 'error':
        return <span className="status-badge status-error">Error</span>;
      default:
        return <span className="status-badge status-warning">Unknown</span>;
    }
  };

  const getPerformanceColor = (score) => {
    if (score >= 0.9) return '#28a745';
    if (score >= 0.8) return '#ffc107';
    return '#dc3545';
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

      <div className="card-grid">
        {models.map((model) => (
          <div key={model.id} className="card">
            <div className="card-title">{model.name}</div>
            <div 
              className="card-score"
              style={{ color: getPerformanceColor(model.performanceScore) }}
            >
              Performance: {(model.performanceScore * 100).toFixed(2)}%
            </div>
            <div className="card-details">
              <p><strong>Optimizer:</strong> {model.optimizer}</p>
              <p><strong>RMSE:</strong> {model.rmse.toFixed(4)}</p>
              <p><strong>MAE:</strong> {model.mae.toFixed(4)}</p>
              <p><strong>Created:</strong> {model.createdAt}</p>
              <div className="mt-2">
                {getStatusBadge(model.status)}
              </div>
            </div>
          </div>
        ))}
      </div>

      {models.length === 0 && (
        <div className="text-center mt-3">
          <p>No models found. Start by creating your first model!</p>
          <button className="btn btn-primary mt-2">
            Create New Model
          </button>
        </div>
      )}
    </div>
  );
};

export default Home;