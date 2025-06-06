import React, { useState, useEffect } from 'react';

const ViewPredictions = () => {
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [currentPage, setCurrentPage] = useState(1);
  const [itemsPerPage] = useState(10);
  const [sortConfig, setSortConfig] = useState({ key: null, direction: 'asc' });
  const [filterModel, setFilterModel] = useState('all');

  useEffect(() => {
    // Simulate loading predictions data
    setTimeout(() => {
      setPredictions([
        {
          id: 1,
          prediction: 0.8534,
          actual: 0.8901,
          timestamp: '2025-06-06 10:30:00',
          model_id: 'Advanced_v1',
          error: 0.0367,
          accuracy: 95.89
        },
        {
          id: 2,
          prediction: 0.7234,
          actual: 0.7001,
          timestamp: '2025-06-06 10:31:00',
          model_id: 'Advanced_v1',
          error: -0.0233,
          accuracy: 96.77
        },
        {
          id: 3,
          prediction: 0.9123,
          actual: 0.9234,
          timestamp: '2025-06-06 10:32:00',
          model_id: 'LSTM_v2',
          error: 0.0111,
          accuracy: 98.79
        },
        {
          id: 4,
          prediction: 0.6789,
          actual: 0.6543,
          timestamp: '2025-06-06 10:33:00',
          model_id: 'CNN_v1',
          error: -0.0246,
          accuracy: 96.37
        },
        {
          id: 5,
          prediction: 0.8901,
          actual: 0.8876,
          timestamp: '2025-06-06 10:34:00',
          model_id: 'Advanced_v1',
          error: -0.0025,
          accuracy: 99.72
        },
        {
          id: 6,
          prediction: 0.5432,
          actual: 0.5678,
          timestamp: '2025-06-06 10:35:00',
          model_id: 'LSTM_v2',
          error: 0.0246,
          accuracy: 95.67
        },
        {
          id: 7,
          prediction: 0.7890,
          actual: 0.7654,
          timestamp: '2025-06-06 10:36:00',
          model_id: 'CNN_v1',
          error: -0.0236,
          accuracy: 97.01
        },
        {
          id: 8,
          prediction: 0.9456,
          actual: 0.9501,
          timestamp: '2025-06-06 10:37:00',
          model_id: 'Advanced_v1',
          error: 0.0045,
          accuracy: 99.53
        },
        {
          id: 9,
          prediction: 0.3210,
          actual: 0.3456,
          timestamp: '2025-06-06 10:38:00',
          model_id: 'LSTM_v2',
          error: 0.0246,
          accuracy: 92.88
        },
        {
          id: 10,
          prediction: 0.8765,
          actual: 0.8543,
          timestamp: '2025-06-06 10:39:00',
          model_id: 'CNN_v1',
          error: -0.0222,
          accuracy: 97.47
        },
        {
          id: 11,
          prediction: 0.6543,
          actual: 0.6789,
          timestamp: '2025-06-06 10:40:00',
          model_id: 'Advanced_v1',
          error: 0.0246,
          accuracy: 96.38
        },
        {
          id: 12,
          prediction: 0.4321,
          actual: 0.4567,
          timestamp: '2025-06-06 10:41:00',
          model_id: 'LSTM_v2',
          error: 0.0246,
          accuracy: 94.61
        }
      ]);
      setLoading(false);
    }, 1000);
  }, []);

  const handleSort = (key) => {
    let direction = 'asc';
    if (sortConfig.key === key && sortConfig.direction === 'asc') {
      direction = 'desc';
    }
    setSortConfig({ key, direction });
  };

  const sortedPredictions = React.useMemo(() => {
    let sortableItems = [...predictions];
    
    // Filter by model
    if (filterModel !== 'all') {
      sortableItems = sortableItems.filter(item => item.model_id === filterModel);
    }
    
    // Sort
    if (sortConfig.key) {
      sortableItems.sort((a, b) => {
        if (a[sortConfig.key] < b[sortConfig.key]) {
          return sortConfig.direction === 'asc' ? -1 : 1;
        }
        if (a[sortConfig.key] > b[sortConfig.key]) {
          return sortConfig.direction === 'asc' ? 1 : -1;
        }
        return 0;
      });
    }
    
    return sortableItems;
  }, [predictions, sortConfig, filterModel]);

  const uniqueModels = [...new Set(predictions.map(p => p.model_id))];

  // Pagination
  const indexOfLastItem = currentPage * itemsPerPage;
  const indexOfFirstItem = indexOfLastItem - itemsPerPage;
  const currentItems = sortedPredictions.slice(indexOfFirstItem, indexOfLastItem);
  const totalPages = Math.ceil(sortedPredictions.length / itemsPerPage);

  const paginate = (pageNumber) => setCurrentPage(pageNumber);

  const getSortIcon = (columnName) => {
    if (sortConfig.key === columnName) {
      return sortConfig.direction === 'asc' ? ' ↑' : ' ↓';
    }
    return ' ↕';
  };

  const getAccuracyColor = (accuracy) => {
    if (accuracy >= 98) return '#28a745';
    if (accuracy >= 95) return '#ffc107';
    return '#dc3545';
  };

  if (loading) {
    return (
      <div className="page-container">
        <div className="loading">Loading predictions...</div>
      </div>
    );
  }

  return (
    <div className="page-container">
      <h1 className="page-title">View Predictions</h1>
      <p className="page-subtitle">
        Analyze and compare model predictions with actual values
      </p>

      <div className="mb-3" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <label htmlFor="modelFilter" className="form-label" style={{ marginRight: '0.5rem' }}>
            Filter by Model:
          </label>
          <select
            id="modelFilter"
            value={filterModel}
            onChange={(e) => setFilterModel(e.target.value)}
            className="form-select"
            style={{ width: '200px', display: 'inline-block' }}
          >
            <option value="all">All Models</option>
            {uniqueModels.map(model => (
              <option key={model} value={model}>{model}</option>
            ))}
          </select>
        </div>
        
        <div style={{ color: '#666' }}>
          Showing {currentItems.length} of {sortedPredictions.length} predictions
        </div>
      </div>

      <div className="table-container">
        <table className="data-table">
          <thead>
            <tr>
              <th 
                onClick={() => handleSort('id')}
                style={{ cursor: 'pointer' }}
              >
                ID{getSortIcon('id')}
              </th>
              <th 
                onClick={() => handleSort('model_id')}
                style={{ cursor: 'pointer' }}
              >
                Model{getSortIcon('model_id')}
              </th>
              <th 
                onClick={() => handleSort('prediction')}
                style={{ cursor: 'pointer' }}
              >
                Prediction{getSortIcon('prediction')}
              </th>
              <th 
                onClick={() => handleSort('actual')}
                style={{ cursor: 'pointer' }}
              >
                Actual{getSortIcon('actual')}
              </th>
              <th 
                onClick={() => handleSort('error')}
                style={{ cursor: 'pointer' }}
              >
                Error{getSortIcon('error')}
              </th>
              <th 
                onClick={() => handleSort('accuracy')}
                style={{ cursor: 'pointer' }}
              >
                Accuracy{getSortIcon('accuracy')}
              </th>
              <th 
                onClick={() => handleSort('timestamp')}
                style={{ cursor: 'pointer' }}
              >
                Timestamp{getSortIcon('timestamp')}
              </th>
            </tr>
          </thead>
          <tbody>
            {currentItems.map((prediction) => (
              <tr key={prediction.id}>
                <td>{prediction.id}</td>
                <td>
                  <span className="status-badge status-success">
                    {prediction.model_id}
                  </span>
                </td>
                <td>{prediction.prediction.toFixed(4)}</td>
                <td>{prediction.actual.toFixed(4)}</td>
                <td style={{ 
                  color: prediction.error >= 0 ? '#dc3545' : '#28a745',
                  fontWeight: 'bold'
                }}>
                  {prediction.error >= 0 ? '+' : ''}{prediction.error.toFixed(4)}
                </td>
                <td style={{ 
                  color: getAccuracyColor(prediction.accuracy),
                  fontWeight: 'bold'
                }}>
                  {prediction.accuracy.toFixed(2)}%
                </td>
                <td>{prediction.timestamp}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="text-center mt-3">
          <div style={{ display: 'inline-flex', gap: '0.5rem' }}>
            <button
              onClick={() => paginate(currentPage - 1)}
              disabled={currentPage === 1}
              className="btn btn-secondary"
              style={{ padding: '0.5rem 1rem' }}
            >
              Previous
            </button>
            
            {[...Array(totalPages)].map((_, index) => (
              <button
                key={index + 1}
                onClick={() => paginate(index + 1)}
                className={`btn ${currentPage === index + 1 ? 'btn-primary' : 'btn-secondary'}`}
                style={{ padding: '0.5rem 1rem' }}
              >
                {index + 1}
              </button>
            ))}
            
            <button
              onClick={() => paginate(currentPage + 1)}
              disabled={currentPage === totalPages}
              className="btn btn-secondary"
              style={{ padding: '0.5rem 1rem' }}
            >
              Next
            </button>
          </div>
        </div>
      )}

      {predictions.length === 0 && (
        <div className="text-center mt-3">
          <p>No predictions found. Upload a parquet file to view predictions.</p>
        </div>
      )}
    </div>
  );
};

export default ViewPredictions;