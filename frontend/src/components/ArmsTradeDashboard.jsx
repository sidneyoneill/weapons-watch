import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { Link } from 'react-router-dom';
import * as d3 from 'd3';
import YearSlider from '../components/YearSlider';
import ChordDiagram from '../components/ChordDiagram';
import '../index.css';

const API_BASE_URL = 'http://localhost:8000';

const ArmsTradeDashboard = () => {
  const [selectedYear, setSelectedYear] = useState(2021);
  const [tradeData, setTradeData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  // Fetch arms trade data for the selected year
  useEffect(() => {
    const fetchArmsTradeData = async () => {
      try {
        setLoading(true);
        const response = await axios.get(`${API_BASE_URL}/arms_trade_matrix/${selectedYear}`);
        setTradeData(response.data);
        setLoading(false);
      } catch (err) {
        console.error('Error fetching arms trade data:', err);
        setError('Failed to load arms trade data. Please try again later.');
        setLoading(false);
      }
    };

    fetchArmsTradeData();
  }, [selectedYear]);

  const handleYearChange = (year) => {
    setSelectedYear(year);
  };

  return (
    <div className="dashboard-container">
      <header className="dashboard-header">
        <h1>Arms Trade Dashboard</h1>
        <nav className="dashboard-nav">
          <Link to="/" className="nav-link">
            Military Expenditure Dashboard
          </Link>
        </nav>
      </header>

      <div className="controls-container">
        <YearSlider 
          selectedYear={selectedYear} 
          onYearChange={handleYearChange} 
          minYear={2015} 
          maxYear={2021} 
        />
      </div>
      
      {error && <div className="error-message">{error}</div>}
      
      <div className="visualization-container">
        {loading ? (
          <div className="loading">Loading data...</div>
        ) : (
          tradeData && (
            <ChordDiagram 
              data={tradeData} 
              selectedYear={selectedYear}
            />
          )
        )}
      </div>
    </div>
  );
};

export default ArmsTradeDashboard; 