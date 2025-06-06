import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import './App.css';
import Home from './pages/Home';
import ModelCreation from './pages/ModelCreation';
import ImportPredictions from './pages/ImportPredictions';
import ViewPredictions from './pages/ViewPredictions';

function App() {
  return (
    <Router>
      <div className="App">
        <nav className="navbar">
          <div className="nav-container">
            <Link to="/" className="nav-brand">ML Model Creator</Link>
            <ul className="nav-menu">
              <li className="nav-item">
                <Link to="/" className="nav-link">Home</Link>
              </li>
              <li className="nav-item">
                <Link to="/create" className="nav-link">Create Model</Link>
              </li>
              <li className="nav-item">
                <Link to="/import" className="nav-link">Import Predictions</Link>
              </li>
              <li className="nav-item">
                <Link to="/predictions" className="nav-link">View Predictions</Link>
              </li>
            </ul>
          </div>
        </nav>
        
        <main className="main-content">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/create" element={<ModelCreation />} />
            <Route path="/import" element={<ImportPredictions />} />
            <Route path="/predictions" element={<ViewPredictions />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;
