import React, { useState } from 'react';

const ImportPredictions = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState(null);
  const [dragActive, setDragActive] = useState(false);

  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    if (file && file.name.endsWith('.parquet')) {
      setSelectedFile(file);
      setUploadStatus(null);
    } else {
      setUploadStatus({
        type: 'error',
        message: 'Please select a valid .parquet file'
      });
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
    if (file && file.name.endsWith('.parquet')) {
      setSelectedFile(file);
      setUploadStatus(null);
    } else {
      setUploadStatus({
        type: 'error',
        message: 'Please drop a valid .parquet file'
      });
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) return;

    setUploading(true);
    setUploadStatus(null);

    // Simulate file upload
    setTimeout(() => {
      setUploading(false);
      setUploadStatus({
        type: 'success',
        message: `Successfully uploaded ${selectedFile.name}. Predictions are now available in the View Predictions tab.`
      });
      
      // In a real app, you would:
      // 1. Upload file to server
      // 2. Process the parquet file
      // 3. Store predictions in database
      // 4. Update application state
    }, 2000);
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
      <h1 className="page-title">Import Predictions</h1>
      <p className="page-subtitle">
        Upload parquet files containing model predictions for analysis and visualization
      </p>

      <div className="mt-3">
        <div
          className={`form-file ${dragActive ? 'drag-active' : ''}`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          onClick={() => document.getElementById('fileInput').click()}
          style={{
            borderColor: dragActive ? '#5a67d8' : '#667eea',
            backgroundColor: dragActive ? '#f0f2ff' : '#f8f9ff'
          }}
        >
          <input
            type="file"
            id="fileInput"
            accept=".parquet"
            onChange={handleFileSelect}
            style={{ display: 'none' }}
          />
          
          <div style={{ fontSize: '3rem', marginBottom: '1rem' }}>📁</div>
          
          {selectedFile ? (
            <div>
              <p style={{ fontSize: '1.2rem', fontWeight: 'bold', color: '#333' }}>
                {selectedFile.name}
              </p>
              <p style={{ color: '#666' }}>
                Size: {formatFileSize(selectedFile.size)}
              </p>
            </div>
          ) : (
            <div>
              <p style={{ fontSize: '1.2rem', fontWeight: 'bold', color: '#333' }}>
                Drop your parquet file here
              </p>
              <p style={{ color: '#666' }}>
                or click to browse and select a file
              </p>
            </div>
          )}
        </div>

        {selectedFile && (
          <div className="text-center mt-3">
            <button
              onClick={handleUpload}
              disabled={uploading}
              className="btn btn-primary"
              style={{ marginRight: '1rem' }}
            >
              {uploading ? 'Uploading...' : 'Upload File'}
            </button>
            <button
              onClick={() => {
                setSelectedFile(null);
                setUploadStatus(null);
              }}
              className="btn btn-secondary"
            >
              Clear Selection
            </button>
          </div>
        )}

        {uploadStatus && (
          <div 
            className={`mt-3 p-3 rounded ${
              uploadStatus.type === 'success' 
                ? 'status-success' 
                : 'status-error'
            }`}
            style={{
              backgroundColor: uploadStatus.type === 'success' ? '#d4edda' : '#f8d7da',
              color: uploadStatus.type === 'success' ? '#155724' : '#721c24',
              border: `1px solid ${uploadStatus.type === 'success' ? '#c3e6cb' : '#f5c6cb'}`,
              borderRadius: '8px'
            }}
          >
            {uploadStatus.message}
          </div>
        )}

        <div className="mt-3">
          <h3 style={{ color: '#333', marginBottom: '1rem' }}>File Requirements:</h3>
          <ul style={{ color: '#666', lineHeight: '1.6' }}>
            <li>File format must be .parquet</li>
            <li>Maximum file size: 100MB</li>
            <li>Required columns: prediction, actual (optional), timestamp (optional)</li>
            <li>Supported data types: numeric predictions, categorical labels</li>
          </ul>
        </div>

        <div className="mt-3">
          <h3 style={{ color: '#333', marginBottom: '1rem' }}>Sample Data Structure:</h3>
          <div className="table-container">
            <table className="data-table">
              <thead>
                <tr>
                  <th>prediction</th>
                  <th>actual</th>
                  <th>timestamp</th>
                  <th>model_id</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>0.8534</td>
                  <td>0.8901</td>
                  <td>2025-06-06 10:30:00</td>
                  <td>model_v1</td>
                </tr>
                <tr>
                  <td>0.7234</td>
                  <td>0.7001</td>
                  <td>2025-06-06 10:31:00</td>
                  <td>model_v1</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ImportPredictions;