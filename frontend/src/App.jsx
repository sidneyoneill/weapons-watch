// src/App.jsx
import React, { useState } from "react";
import MapComponent from "./components/MapComponent";
import ChartComponent from "./components/ChartComponent";
import axios from "axios";
import CountryExpenditureComponent from "./components/CountryExpidentureComponent";
import ExpenditureMapComponent from "./components/ExpidentureMapComponent";
import DataModeToggle from "./components/DataModeToggle";

function App() {
  const [selectedCountry, setSelectedCountry] = useState(null);
  const [timeSeriesData, setTimeSeriesData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [dataMode, setDataMode] = useState('total'); // Default to 'total' mode

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
    <div className="App" style={{ padding: "20px" }}>
      <h1>Arms Trade Dashboard</h1>
      
      {/* Data Mode Toggle */}
      <DataModeToggle 
        currentMode={dataMode} 
        onModeChange={handleDataModeChange} 
      />
      
      {error && (
        <div style={{ color: "red", marginBottom: "10px" }}>{error}</div>
      )}
      
      {/* <div style={{ padding: "20px" }}> */}
        {/* <div style={{ marginBottom: "20px" }}>
          <MapComponent onCountrySelect={handleCountrySelect} />
        </div>
        {loading && <div>Loading data...</div>}
        {selectedCountry && timeSeriesData.length > 0 && !loading && (
          <ChartComponent
            timeSeriesData={timeSeriesData}
            selectedCountry={selectedCountry}
            dataMode={dataMode}
          />
        )} */}
        {/* <CountryExpenditureComponent /> */}
        <ExpenditureMapComponent dataMode={dataMode} />
      {/* </div> */}
    </div>
  );
}

export default App;
