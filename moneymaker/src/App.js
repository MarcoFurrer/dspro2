import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import './App.css';
import Home from './pages/Home';
import ModelCreation from './pages/ModelCreation';
import MakePrediction from './pages/MakePrediction';

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
                <Link to="/make-prediction" className="nav-link">Make Prediction</Link>
              </li>
            </ul>
          </div>
        </nav>
        
        <main className="main-content">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/create" element={<ModelCreation />} />
            <Route path="/make-prediction" element={<MakePrediction />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;
