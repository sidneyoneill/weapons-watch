// src/App.jsx
import React, { useState, useEffect } from "react";
import MapComponent from "./components/MapComponent";
import ChartComponent from "./components/ChartComponent";
import axios from "axios";
import CountryExpenditureComponent from "./components/CountryExpidentureComponent";
import ExpenditureMapComponent from "./components/ExpidentureMapComponent";
import GlobeComponent from "./components/GlobeComponent";
import DataModeToggle from "./components/DataModeToggle";

// Define consistent theme colors
const theme = {
  background: '#000000',
  secondaryBg: '#111111',
  accent: '#ea580c', // Orange accent matching the globe
  text: {
    primary: '#ffffff',
    secondary: '#cccccc',
  },
  borders: '#333333',
};

function App() {
  const [selectedCountry, setSelectedCountry] = useState(null);
  const [timeSeriesData, setTimeSeriesData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [dataMode, setDataMode] = useState('total'); // Default to 'total' mode

  // Listen for custom events from the GlobeComponent
  useEffect(() => {
    const handleDataModeChange = (event) => {
      setDataMode(event.detail);
    };

    window.addEventListener('setDataMode', handleDataModeChange);
    
    return () => {
      window.removeEventListener('setDataMode', handleDataModeChange);
    };
  }, []);

  // Handler to be called when a country is selected from the map
  const handleCountrySelect = (country) => {
    setSelectedCountry(country);
    setError(null);
    setLoading(true);

    // Fallback mock data in case the API fails
    const mockData = [
      { Year: 2015, Expenditure: 100 },
      { Year: 2016, Expenditure: 120 },
      { Year: 2017, Expenditure: 140 },
      { Year: 2018, Expenditure: 130 },
      { Year: 2019, Expenditure: 150 },
    ];

    axios
      .get(`http://localhost:8000/expenditure/${encodeURIComponent(country)}?mode=${dataMode}`)
      .then((response) => {
        console.log("Received data:", response.data);
        if (
          response.data &&
          response.data.time_series &&
          response.data.time_series.length > 0
        ) {
          setTimeSeriesData(response.data.time_series);
        } else {
          console.warn(
            "Empty or invalid data received from API, using mock data"
          );
          setTimeSeriesData(mockData);
          setError(
            "Limited data available for " + country + " (using sample data)"
          );
        }
        setLoading(false);
      })
      .catch((error) => {
        console.error("Error fetching expenditure data:", error);
        setError(
          "Could not fetch data for " + country + " (using sample data)"
        );
        setTimeSeriesData(mockData);
        setLoading(false);
      });
  };

  // Handler for data mode toggle
  const handleDataModeChange = (mode) => {
    setDataMode(mode);
    console.log(`Data mode changed to: ${mode}`);
    
    // If a country is selected, refetch its data with the new mode
    if (selectedCountry) {
      handleCountrySelect(selectedCountry);
    }
  };

  return (
    <div className="App" style={{ 
      padding: "0",
      margin: "0",
      minHeight: "100vh",
      width: "100vw",
      backgroundColor: theme.background,
      color: theme.text.primary,
      fontFamily: "'Arial', sans-serif",
      position: "relative",
      overflow: "hidden"
    }}>
      {/* Header Bar - Full Width */}
      <header style={{
        position: "fixed",
        top: "0",
        left: "0",
        width: "100vw",
        padding: "15px 20px",
        backgroundColor: "rgba(17, 17, 17, 0.8)",
        backdropFilter: "blur(10px)",
        borderBottom: `1px solid ${theme.borders}`,
        zIndex: 1000,
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        boxSizing: "border-box", // Make sure padding is included in the width
      }}>
        <h1 style={{ 
          margin: "0",
          fontSize: "1.8rem",
          fontWeight: "600",
          color: theme.text.primary,
          textShadow: "0 2px 4px rgba(0, 0, 0, 0.5)"
        }}>
          <span style={{ color: theme.accent }}>Arms Trade</span> Dashboard
        </h1>
        
        {/* Data Mode Toggle styled to match theme */}
        <div style={{ 
          backgroundColor: "rgba(51, 51, 51, 0.7)",
          padding: "8px 12px",
          borderRadius: "8px",
          border: `1px solid ${theme.borders}`,
          boxShadow: "0 4px 6px rgba(0, 0, 0, 0.1)"
        }}>
          <DataModeToggle 
            currentMode={dataMode} 
            onModeChange={handleDataModeChange} 
          />
        </div>
      </header>
      
      {/* Error message with improved styling */}
      {error && (
        <div style={{ 
          position: "absolute",
          top: "80px",
          left: "50%",
          transform: "translateX(-50%)",
          backgroundColor: "rgba(255, 0, 0, 0.2)",
          color: "#ff8080",
          padding: "10px 20px",
          borderRadius: "8px",
          backdropFilter: "blur(5px)",
          border: "1px solid #ff3333",
          zIndex: 1001,
          boxShadow: "0 4px 12px rgba(255, 0, 0, 0.2)"
        }}>
          {error}
        </div>
      )}
      
      {/* 3D Interactive Globe - Fill the entire viewport */}
      <div style={{ 
        position: "absolute", 
        top: 0, 
        left: 0, 
        width: "100vw", 
        height: "100vh",
        overflow: "hidden"
      }}>
        <GlobeComponent dataMode={dataMode} />
      </div>
      
      {/* Version info and credits */}
      <div style={{
        position: "absolute",
        bottom: "10px",
        left: "10px",
        color: theme.text.secondary,
        fontSize: "0.7rem",
        opacity: 0.7,
        zIndex: 900
      }}>
        v1.0 • SIPRI Data Visualization
      </div>
      
      {/* Commented out original components for reference
      <div style={{ marginBottom: "20px" }}>
        <MapComponent onCountrySelect={handleCountrySelect} />
      </div>
      {loading && <div>Loading data...</div>}
      {selectedCountry && timeSeriesData.length > 0 && !loading && (
        <ChartComponent
          timeSeriesData={timeSeriesData}
          selectedCountry={selectedCountry}
          dataMode={dataMode}
        />
      )}
      <CountryExpenditureComponent />
      <ExpenditureMapComponent dataMode={dataMode} />
      */}
    </div>
  );
}

export default App;
